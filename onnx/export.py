"""Export a complete FP32 ONNX bundle for infer_onnx.py on CPU or CUDA.

Run from the repository root: python onnx/export.py --model ./hf_model --output ./onnx_export
The onnx/ scripts directory deliberately has no __init__.py, to avoid shadowing PyPI onnx.
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import MethodType
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from checkpoint import CODEC_FILES, DEFAULT_CODEC, load_checkpoint, load_codec, resolve, sha256
from graphs import Backbone, CodecEncoder, CodecTransformer, DepthFrame, causal_extra_padding, export_graph
from cached_codec import export_cached_codec

TTS_FILES = ('backbone.onnx', 'depth_frame.onnx', 'text_embeddings.npy', 'audio_embeddings.npy',
             'tokenizer.json', 'manifest.json')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_graph(path, module, cases, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    errors = []
    with torch.inference_mode():
        for inputs in cases:
            expected = module(*inputs)
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            feed = {spec.name: value.detach().cpu().numpy() for spec, value in zip(session.get_inputs(), inputs)}
            if len(feed) != len(inputs):
                raise ValueError(f'Unexpected exported input contract: {path.name}')
            actual = session.run(None, feed)
            if len(actual) != len(expected):
                raise ValueError(f'Unexpected output count: {path.name}')
            for index, (reference, value) in enumerate(zip(expected, actual)):
                reference = reference.detach().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError(f'{path.name} returned non-finite output')
                if np.issubdtype(reference.dtype, np.integer) or reference.dtype == np.bool_:
                    np.testing.assert_array_equal(value, reference, err_msg=f'{path.name} output {index}')
                else:
                    np.testing.assert_allclose(value, reference, rtol=3e-3, atol=2e-4,
                                               err_msg=f'{path.name} output {index}')
                    errors.append(float(np.max(np.abs(value-reference))) if value.size else 0.0)
    result = dict(graph=path.name, dynamic_test_cases=len(cases), max_absolute_error=max(errors, default=0.0),
                  provider='CPUExecutionProvider', rtol=3e-3, atol=2e-4, integer_outputs='exact')
    print('Validated', result, flush=True)
    return result


def export_tts(model, tokenizer, checkpoint, output, threads):
    backbone, depth = Backbone(model).eval(), DepthFrame(model).eval()
    np.save(output/'text_embeddings.npy', model.text_embeddings.weight.detach().numpy())
    np.save(output/'audio_embeddings.npy', model.audio_embeddings.weight.detach().numpy())
    tokenizer.backend_tokenizer.save(str(output/'tokenizer.json'))
    dim = model.backbone.config.hidden_size
    past_shape = (len(model.backbone.layers), 2, 1, backbone.core.kv_heads, 3, backbone.core.dim)
    hidden = torch.randn(1, 4, dim)
    export_graph(backbone, (hidden, torch.arange(3,7)[None], torch.zeros(past_shape)),
                 output/'backbone.onnx', ['hidden','positions','past'], ['last_hidden','present'],
                 {'hidden':{1:'query'}, 'positions':{1:'query'}, 'past':{4:'past_length'}, 'present':{4:'total_length'}})
    export_graph(depth, (hidden[:,-1], torch.ones(16,1,2051), torch.tensor([True])),
                 output/'depth_frame.onnx', ['hidden','noise','allow_eos'], ['codes','next_hidden'], {})
    cases = []
    for query, cache in [(5,0),(1,7),(3,2)]:
        shape = (*past_shape[:4],cache,past_shape[-1])
        cases.append((torch.randn(1,query,dim), torch.arange(cache,cache+query)[None], torch.zeros(shape)))
    checks = [validate_graph(output/'backbone.onnx',backbone,cases,threads)]
    cases = [(torch.randn(1,dim),torch.empty(16,1,2051).exponential_().clamp_min(1e-8),torch.tensor([flag]))
             for flag in (False,True)]
    checks.append(validate_graph(output/'depth_frame.onnx',depth,cases,threads))
    meta = dict(format_version=1,architecture='m5-llama-rq16-qwen3-12hz',layers=len(model.backbone.layers),
                kv_heads=backbone.core.kv_heads,head_dim=backbone.core.dim,hidden_size=dim,
                codebooks=16,audio_vocab=2051,sample_rate=24000,hop_length=1920,topk=10,temperature=.9,
                min_frames=6,max_sequence=2048,checkpoint='model.safetensors',checkpoint_sha256=sha256(checkpoint),
                fused_projections=False,fast_sample=False,depth_sdpa=False)
    save(output/'manifest.json',meta)
    return checks


def export_codec(codec, output, threads):
    checks = []
    cached = export_cached_codec(codec, output)
    cases = []
    for frames, offset, length in [(1,0,0),(7,3,3),(4,80,cached.window-1)]:
        past = torch.zeros(cached.layers,2,1,cached.kv_heads,length,cached.dim)
        cases.append((torch.randint(0,2048,(1,16,frames)),torch.tensor([offset]),past,*cached.empty_states()))
    checks.append(validate_graph(output/'codec_decoder_cached.onnx',cached,cases,threads))
    del cached, cases
    for module in codec.decoder.modules():
        if type(module).__name__ == 'Qwen3TTSTokenizerV2CausalConvNet':
            module._get_extra_padding_for_conv1d = MethodType(causal_extra_padding,module)
    codec.decoder.pre_transformer.config._attn_implementation = 'eager'
    codec.encoder.encoder_transformer.config._attn_implementation = 'eager'
    codec.decoder.pre_transformer = CodecTransformer(codec.decoder.pre_transformer,True)
    codec.encoder.encoder_transformer = CodecTransformer(codec.encoder.encoder_transformer,False)
    export_graph(codec.decoder,(torch.zeros(1,16,12,dtype=torch.long),),output/'codec_decoder.onnx',
                 ['codes'],['audio'],{'codes':{2:'frames'},'audio':{2:'samples'}})
    checks.append(validate_graph(output/'codec_decoder.onnx',codec.decoder,
                                 [(torch.randint(0,2048,(1,16,n)),) for n in (1,9,16)],threads))
    encoder = CodecEncoder(codec)
    export_graph(encoder,(torch.zeros(1,24000),),output/'codec_encoder.onnx',
                 ['audio'],['codes'],{'audio':{1:'samples'},'codes':{2:'frames'}})
    cases = [(torch.sin(torch.arange(n,dtype=torch.float32)*(.037 if n%2 else .061))[None]*.1,) for n in (19200,30721)]
    checks.append(validate_graph(output/'codec_encoder.onnx',encoder,cases,threads))
    save(output/'codec_identity.json',dict(encoder_sha256=sha256(output/'codec_encoder.onnx')))
    return checks


def copy_codec(source, output):
    metadata = json.loads((source/'manifest.json').read_text())
    for key, value in {'sample_rate':24000,'hop_length':1920,'codebooks':16}.items():
        if metadata.get(key) != value:
            raise ValueError(f'Incompatible reused codec: {key}')
    checks = json.loads((source/'checksums.json').read_text()) if (source/'checksums.json').is_file() else None
    for name in CODEC_FILES:
        path = source/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing reused codec asset: {path}')
        checksum = sha256(path)
        if checks is not None and (path.stat().st_size!=checks[name]['bytes'] or checksum!=checks[name]['sha256']):
            raise ValueError(f'Corrupt reused codec asset: {name}')
        shutil.copy2(path,output/name)
        if sha256(output/name) != checksum:
            raise ValueError(f'Codec copy mismatch: {name}')
        if name.endswith('.onnx'):
            onnx.checker.check_model(str(output/name))
    identity = json.loads((output/'codec_identity.json').read_text())
    if identity['encoder_sha256'] != sha256(output/'codec_encoder.onnx'):
        raise ValueError('Reused codec encoder identity mismatch')


def validate_bundle(output, threads):
    from infer_onnx import M5Onnx
    with tempfile.TemporaryDirectory(prefix='.smoke-',dir=output) as directory:
        import soundfile as sf
        reference = Path(directory)/'reference.wav'
        sf.write(reference,np.zeros(24000,dtype=np.float32),24000)
        for mode in ('cached','window'):
            model = M5Onnx(output,threads=threads,codec_mode=mode,local_files_only=True,verify=True,device='cpu')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Generation reached max_frames without EOS',category=RuntimeWarning)
                audio = model.infer('你好',reference,'测试',seed=1234,max_frames=8,stream=True)
            if audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError(f'Invalid full pipeline smoke output in {mode}')
            del model
            gc.collect()
    return 'cached and window, 8-frame synthetic-reference structural smoke only; not an audio quality test'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='M5 HF repository ID or directory with model.safetensors, config.json, conditioning.json and tokenizer/')
    parser.add_argument('--config',type=Path,help='Optional architecture JSON override for legacy bundles')
    codec = parser.add_mutually_exclusive_group()
    codec.add_argument('--codec',default=None,help=f'Qwen 12Hz codec checkpoint directory or HF ID (default: {DEFAULT_CODEC})')
    codec.add_argument('--reuse-codec',help='Existing complete ONNX bundle directory/HF ID; copy its codec instead of re-exporting it')
    parser.add_argument('--output',type=Path,required=True,help='New directory; never overwrites an existing bundle')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--codec-revision',default='main',help='Version of the codec checkpoint or reused ONNX repository')
    parser.add_argument('--cache-dir')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    if args.threads<1:
        parser.error('--threads must be positive')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    if not hasattr(onnx,'load'):
        raise ImportError('Install the export dependencies with pip install -r requirements.txt')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    settings = dict(cache_dir=args.cache_dir,local_files_only=args.local_files_only)
    bundle = resolve(args.model,revision=args.revision,patterns=['model.safetensors','config.json','conditioning.json','tokenizer/*'],**settings)
    if args.reuse_codec:
        codec_bundle = resolve(args.reuse_codec,revision=args.codec_revision,
                               patterns=[*CODEC_FILES,'manifest.json','checksums.json'],**settings)
    else:
        codec_bundle = resolve(args.codec or DEFAULT_CODEC,revision=args.codec_revision,patterns=['*.json','*.safetensors'],**settings)
    model, tokenizer = load_checkpoint(bundle,args.config)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-',dir=output.parent) as tmp:
        stage = Path(tmp)
        with torch.inference_mode():
            checks = export_tts(model,tokenizer,bundle/'model.safetensors',stage,args.threads)
            del model, tokenizer
            gc.collect()
            if args.reuse_codec:
                copy_codec(codec_bundle,stage)
            else:
                codec_model = load_codec(codec_bundle)
                checks.extend(export_codec(codec_model,stage,args.threads))
                del codec_model
                gc.collect()
        files = TTS_FILES+CODEC_FILES
        save(stage/'checksums.json',{name:dict(bytes=(stage/name).stat().st_size,sha256=sha256(stage/name)) for name in files})
        smoke = validate_bundle(stage,args.threads)
        save(stage/'export_info.json',dict(format_version=1,precision='fp32',opset=17,dynamo=False,
                                           codec_reused=bool(args.reuse_codec),numerical_validation=checks,pipeline_smoke=smoke,
                                           versions={n:importlib.metadata.version(n) for n in ('torch','transformers','onnx','onnxruntime','tokenizers')},
                                           note='Explicit ONNX attention and FP32 sampling follow the ONNX reference implementation. This is not a FlashAttention/BF16 bitwise equivalence claim.'))
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    print(f'Complete ONNX bundle: {output}',flush=True)


if __name__=='__main__':
    main()
