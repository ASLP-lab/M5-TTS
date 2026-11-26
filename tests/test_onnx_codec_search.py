import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'onnx'))
from codec_search import rewrite_codec_search


def encoder_graph(centers):
    # The shape and operators match the legacy exporter, with a tiny codebook.
    initializer = [
        numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        for name, value, dtype in (
            ('c', [centers], np.float32), ('axis_q', [2], np.int64),
            ('axis_c', [1], np.int64), ('last', [-1], np.int64),
            ('batch_zero', 0, np.int64), ('square', 2, np.float32),
            ('root', [0.4999997615814209], np.float32),
        )
    ]
    nodes = [
        helper.make_node('Unsqueeze', ['x', 'axis_q'], ['queries']),
        helper.make_node('Unsqueeze', ['c', 'axis_c'], ['centers']),
        helper.make_node('Sub', ['queries', 'centers'], ['difference']),
        helper.make_node('Pow', ['difference', 'square'], ['squared']),
        helper.make_node('ReduceSum', ['squared', 'last'], ['sum'], keepdims=0),
        helper.make_node('Pow', ['sum', 'root'], ['distances_with_batch']),
        helper.make_node('Gather', ['distances_with_batch', 'batch_zero'], ['distances'], axis=0),
    ]
    outputs = []
    for i in range(16):
        output = f'codes_{i}'
        nodes.append(helper.make_node('ArgMin', ['distances'], [output], axis=-1, keepdims=0))
        outputs.append(helper.make_tensor_value_info(output, TensorProto.INT64, ['frames']))
    graph = helper.make_graph(nodes, 'encoder', [
        helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 'frames', 2]),
    ], outputs, initializer)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


class CodecSearchTest(unittest.TestCase):
    def compare(self, centers, examples):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.onnx'
            onnx.save(encoder_graph(centers), path)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            base = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
            rewrite_codec_search(path)
            fast = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
            self.assertEqual([s.name for s in fast.get_outputs()], [s.name for s in base.get_outputs()])
            for vectors, expected in examples:
                feed = {'x': np.array([vectors], dtype=np.float32)}
                np.testing.assert_array_equal(base.run(None, feed), [expected] * 16)
                np.testing.assert_array_equal(fast.run(None, feed), [expected] * 16)

    def test_dynamic_frames_without_near_ties(self):
        self.compare([[0, 0], [4, 0], [0, 4]], [
            ([[0.1, 0]], [0]),
            ([[0.1, 0], [3.9, 0], [0, 3.9]], [0, 1, 2]),
        ])

    def test_recheck_corrects_cancellation_and_preserves_ties(self):
        # The FP32 dot-product scores tie at these large common coordinates.
        self.compare([[10000, 0], [10000, 0.5], [10000, 2]], [
            ([[10000, 0.49], [10000, 0.25]], [1, 0]),
        ])


if __name__ == '__main__':
    unittest.main()
