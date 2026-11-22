"""Opt-in rewrite of the FP32 Mimi codebook search emitted by the ONNX exporter."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rewrite_codec_search(path):
    """Preserve the encoder interface and recheck near-ties with the original distance."""
    model = onnx.load(path)
    graph = model.graph
    producers = {value: node for node in graph.node for value in node.output}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    searches = [node for node in graph.node if node.op_type == 'ArgMin']
    if len(searches) != 16:
        raise ValueError('Expected 16 reference encoder codebook searches')

    def source(value, op_type):
        node = producers.get(value)
        if node is None or node.op_type != op_type:
            raise ValueError(f'Unsupported codec search graph: expected {op_type} for {value}')
        return node

    def constant(value):
        initializer = initializers.get(value)
        if initializer is None:
            attributes = source(value, 'Constant').attribute
            initializer = next((a.t for a in attributes if a.name == 'value'), None)
        if initializer is None:
            raise ValueError(f'Expected a tensor constant in codec search graph: {value}')
        return numpy_helper.to_array(initializer)

    def attribute(node, key, default):
        return next((helper.get_attribute_value(a) for a in node.attribute if a.name == key), default)

    # Match the legacy torch.cdist lowering before replacing any part of it.
    matches = {}
    for i, argmin in enumerate(searches):
        gather = source(argmin.input[0], 'Gather')
        root = source(gather.input[0], 'Pow')
        reduction = source(root.input[0], 'ReduceSum')
        square = source(reduction.input[0], 'Pow')
        diff = source(square.input[0], 'Sub')
        queries = source(diff.input[0], 'Unsqueeze')
        codebook = source(diff.input[1], 'Unsqueeze')
        exponent = constant(root.input[1])
        if (attribute(argmin, 'axis', 0) != -1 or attribute(argmin, 'keepdims', 1) != 0
                or attribute(argmin, 'select_last_index', 0) != 0
                or attribute(gather, 'axis', 0) != 0 or constant(gather.input[1]).tolist() != 0
                or exponent.size != 1 or not 0 < exponent.item() < 1
                or attribute(reduction, 'keepdims', 1) != 0
                or constant(reduction.input[1]).tolist() != [-1]
                or constant(square.input[1]).tolist() != 2.0
                or constant(queries.input[1]).tolist() != [2]
                or constant(codebook.input[1]).tolist() != [1]):
            raise ValueError(f'Unsupported reference encoder distance layout at {argmin.name}')
        matches[argmin.output[0]] = (i, gather, root, reduction, square, queries, codebook)

    def tensor(key, value, dtype):
        name = 'm5_codec_search_' + key
        if name in producers or name in initializers:
            raise ValueError(f'Conflicting codec search name: {name}')
        graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=dtype), name))
        return name

    axis_zero = tensor('axis_zero', [0], np.int64)
    axes_codebook = tensor('axes_codebook', [0, 1], np.int64)
    zero = tensor('zero', 0, np.int64)
    one = tensor('one', 1, np.int64)
    two = tensor('two', [2], np.int64)
    minus_two = tensor('minus_two', -2.0, np.float32)
    # Keep a margin proportional to both norms to cover cancellation in FP32 scores.
    margin_scale = tensor('margin_scale', 2e-4, np.float32)

    updated = []
    for node in graph.node:
        if node.op_type != 'ArgMin':
            updated.append(node)
            continue
        i, gather, root, reduction, square, queries, codebook = matches[node.output[0]]
        prefix = f'm5_codec_search_{i}_'

        def add(op, inputs, suffix, **attrs):
            name = prefix + suffix
            updated.append(helper.make_node(op, inputs, [name], name=name, **attrs))
            return name

        vectors = queries.input[0]  # [1, frames, dimensions]
        centers = add('Squeeze', [codebook.output[0], axes_codebook], 'centers')
        transposed = add('Transpose', [centers], 'transposed', perm=[1, 0])
        axes = reduction.input[1]
        norm = add('ReduceSum', [add('Mul', [centers, centers], 'centers_sq'), axes], 'norm', keepdims=0)
        dot = add('MatMul', [vectors, transposed], 'dot')
        # Squared distance is norm + query_norm - 2 * dot. The query norm
        # is shared by all candidates, so it is only needed for the margin.
        scores = add('Add', [add('Mul', [dot, minus_two], 'scaled_dot'), norm], 'scores')
        scores = add('Gather', [scores, zero], 'frame_scores', axis=0)
        approx = add('ArgMin', [scores], 'winner', axis=-1, keepdims=0)
        best = prefix + 'best_two'
        updated.append(helper.make_node('TopK', [scores, two], [best, prefix + 'indices'],
                                        name=best, axis=-1, largest=0, sorted=1))
        low = add('Gather', [best, zero], 'best', axis=-1)
        high = add('Gather', [best, one], 'second', axis=-1)
        gap = add('Sub', [high, low], 'gap')
        query_sq = add('Mul', [vectors, vectors], 'query_sq')
        query_norm = add('ReduceSum', [query_sq, axes], 'query_norm', keepdims=0)
        query_norm = add('Gather', [query_norm, zero], 'frame_norm', axis=0)
        max_norm = add('ReduceMax', [norm], 'max_norm', keepdims=0)
        scale = add('Add', [query_norm, max_norm], 'scale')
        margin = add('Mul', [scale, margin_scale], 'margin')
        close = add('LessOrEqual', [gap, margin], 'close')
        rows = add('Squeeze', [add('NonZero', [close], 'indices_to_check'), axis_zero], 'rows')

        # Guard the empty selection: older ORT versions can lose the rank of
        # an empty ArgMin output, which ScatterElements rejects.
        count = add('Size', [rows], 'count')
        needs_recheck = add('Greater', [count, zero], 'needs_recheck')
        branch_start = len(updated)
        selected = add('Gather', [queries.output[0], rows], 'selected', axis=1)
        delta = add('Sub', [selected, codebook.output[0]], 'delta')
        squared = add('Pow', [delta, square.input[1]], 'squared')
        total = add('ReduceSum', [squared, axes], 'total', keepdims=0)
        distance = add('Pow', [total, root.input[1]], 'distance')
        distance = add('Gather', [distance, gather.input[1]], 'frame_distance', axis=0)
        rechecked = add('ArgMin', [distance], 'rechecked', axis=-1, keepdims=0)
        corrected = add('ScatterElements', [approx, rows, rechecked], 'corrected', axis=0)
        then_graph = helper.make_graph(
            updated[branch_start:], prefix + 'recheck', [],
            [helper.make_tensor_value_info(corrected, TensorProto.INT64, ['frames'])])
        del updated[branch_start:]
        unchanged = prefix + 'unchanged'
        else_graph = helper.make_graph(
            [helper.make_node('Identity', [approx], [unchanged])], prefix + 'keep', [],
            [helper.make_tensor_value_info(unchanged, TensorProto.INT64, ['frames'])])
        updated.append(helper.make_node('If', [needs_recheck], list(node.output), name=prefix + 'result',
                                        then_branch=then_graph, else_branch=else_graph))

    # Remove the old full-frame distance path and its now-unused constants.
    live = {output.name for output in graph.output}
    kept = []

    def dependencies(node):
        needed = set(node.input)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                subgraph = attribute.g
                local = {v.name for v in subgraph.input} | {v.name for v in subgraph.initializer}
                local.update(value for child in subgraph.node for value in child.output)
                needed.update(v.name for v in subgraph.output if v.name not in local)
                for child in subgraph.node:
                    needed.update(dependencies(child) - local)
        return needed

    for node in reversed(updated):
        if live.intersection(node.output):
            kept.append(node)
            live.update(dependencies(node))
    del graph.node[:]
    graph.node.extend(reversed(kept))
    for i in reversed(range(len(graph.initializer))):
        if graph.initializer[i].name not in live:
            del graph.initializer[i]

    onnx.checker.check_model(model)
    onnx.save(model, path)
