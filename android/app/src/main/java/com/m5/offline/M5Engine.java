package com.m5.offline;

import ai.onnxruntime.*;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** CPU-only, non-streaming port of m5_onnx/runtime.py. Single worker owns an instance. */
public final class M5Engine implements AutoCloseable {
    public interface Progress { void update(String message); }
    public interface NoiseSource { void fill(FloatBuffer buffer) throws IOException; }
    public static final class Voice {
        public final String name, referenceText;
        final long[][] codes;
        Voice(JSONObject json) throws JSONException {
            name = json.getString("name"); referenceText = json.getString("reference_text");
            JSONArray rows = json.getJSONArray("codes");
            if (rows.length() < 1 || rows.length() > 1000 || referenceText.trim().isEmpty())
                throw new JSONException("参考音色数据无效");
            codes = new long[rows.length()][16];
            for (int i = 0; i < codes.length; i++) {
                JSONArray row = rows.getJSONArray(i);
                if (row.length() != 16) throw new JSONException("参考音色码本数量错误");
                for (int j = 0; j < 16; j++) {
                    Object raw = row.get(j); long v = row.getLong(j);
                    if (!(raw instanceof Number) || ((Number) raw).doubleValue() != v || v < 0 || v > 2047)
                        throw new JSONException("参考音色 token 错误");
                    codes[i][j] = v;
                }
            }
        }
        public String toString() { return name; }
    }
    public static final class Synthesis {
        public final double elapsedSeconds, audioSeconds;
        public final int segments;
        Synthesis(double elapsed, double audio, int count) { elapsedSeconds = elapsed; audioSeconds = audio; segments = count; }
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession backbone, depth, codec;
    private NpyMatrix textEmbedding, audioEmbedding;
    private Tokenizer tokenizer;
    private JSONObject meta, codecMeta;
    private final List<Voice> voices = new ArrayList<>();

    public M5Engine(File root, int threads, Progress progress) throws Exception {
        if (threads < 1 || threads > 8) throw new IllegalArgumentException("线程数需要为 1–8");
        try {
            meta = json(new File(root, "manifest.json")); codecMeta = json(new File(root, "codec_cached.json"));
            if (!"m5-llama-rq16-qwen3-12hz".equals(meta.getString("architecture")) ||
                meta.getInt("layers") != 8 || meta.getInt("kv_heads") != 2 || meta.getInt("head_dim") != 64 ||
                meta.getInt("hidden_size") != 512 || meta.getInt("codebooks") != 16 ||
                meta.getInt("sample_rate") != 24000 || meta.getInt("hop_length") != 1920 ||
                meta.getInt("audio_vocab") != 2051 || codecMeta.getInt("layers") != 8 ||
                codecMeta.getInt("kv_heads") != 16 || codecMeta.getInt("head_dim") != 64)
                throw new IOException("模型架构与此版本应用不兼容");
            progress.update("正在加载文本分词器…"); tokenizer = new Tokenizer(new File(root, "tokenizer.bin"));
            textEmbedding = new NpyMatrix(new File(root, "text_embeddings.npy"));
            audioEmbedding = new NpyMatrix(new File(root, "audio_embeddings.npy"));
            if (textEmbedding.columns != 512 || textEmbedding.rows != 129992 ||
                audioEmbedding.columns != 512 || audioEmbedding.rows != 16 * 2051)
                throw new IOException("Embedding 维度与模型不匹配");
            JSONArray list = json(new File(root, "voices.json")).getJSONArray("voices");
            for (int i = 0; i < list.length(); i++) voices.add(new Voice(list.getJSONObject(i)));
            if (voices.isEmpty()) throw new IOException("模型包没有音色");
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(threads); options.setInterOpNumThreads(1);
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                options.addConfigEntry("session.intra_op.allow_spinning", "0");
                options.addConfigEntry("session.inter_op.allow_spinning", "0");
                progress.update("正在加载语音生成模型…");
                backbone = env.createSession(new File(root, "backbone.onnx").getPath(), options);
                depth = env.createSession(new File(root, "depth_frame.onnx").getPath(), options);
                progress.update("正在加载声音解码器…");
                codec = env.createSession(new File(root, "codec_decoder_cached.onnx").getPath(), options);
            }
        } catch (Exception | OutOfMemoryError e) { close(); throw e; }
    }

    public List<Voice> voices() { return Collections.unmodifiableList(voices); }
    public static JSONObject json(File path) throws IOException, JSONException {
        return new JSONObject(new String(Files.readAllBytes(path.toPath()), java.nio.charset.StandardCharsets.UTF_8));
    }
    public static String conditioned(String reference, String text) {
        String combined = reference.trim() + text.trim();
        for (String token : new String[]{"<|caption_begin|>", "<|caption_end|>", "<|accent_begin|>",
            "<|accent_end|>", "<|speaker_start|>", "<|speaker_end|>", "<|lang_begin|>", "<|lang_end|>"})
            if (combined.contains(token)) throw new IllegalArgumentException("文本包含保留的控制标记");
        return "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>" +
            "<|speaker_start|>None<|speaker_end|><|lang_begin|>Chinese<|lang_end|>" + combined;
    }

    public Synthesis synthesize(String text, int voiceIndex, long seed, File output,
                                AtomicBoolean cancelled, Progress progress) throws Exception {
        if (voiceIndex < 0 || voiceIndex >= voices.size()) throw new IllegalArgumentException("请选择音色");
        return synthesize(text, voices.get(voiceIndex), seed, output, cancelled, progress);
    }
    public Synthesis synthesize(String text, Voice voice, long seed, File output,
                                AtomicBoolean cancelled, Progress progress) throws Exception {
        Random random = new Random(seed);
        return synthesize(text, voice, output, cancelled, progress, buffer -> {
            for (int i = 0; i < buffer.capacity(); i++)
                buffer.put(i, Math.max((float) -Math.log(1.0 - random.nextDouble()), Float.MIN_NORMAL));
        });
    }

    public Synthesis synthesize(String text, int voiceIndex, File output, AtomicBoolean cancelled,
                                Progress progress, NoiseSource noiseSource) throws Exception {
        if (voiceIndex < 0 || voiceIndex >= voices.size()) throw new IllegalArgumentException("请选择音色");
        return synthesize(text, voices.get(voiceIndex), output, cancelled, progress, noiseSource);
    }
    public Synthesis synthesize(String text, Voice voice, File output, AtomicBoolean cancelled,
                                Progress progress, NoiseSource noiseSource) throws Exception {
        if (voice == null) throw new IllegalArgumentException("请选择音色");
        List<String> segments = splitText(text);
        long start = System.nanoTime(); double duration;
        boolean success = false;
        try (WavWriter wav = new WavWriter(output)) {
            for (int i = 0; i < segments.size(); i++) {
                check(cancelled);
                final String prefix = "第 " + (i + 1) + "/" + segments.size() + " 段：";
                long[][] generated = generate(segments.get(i), voice, cancelled,
                    message -> progress.update(prefix + message), noiseSource);
                progress.update(prefix + "正在转换成声音…");
                decode(voice.codes, generated, wav, cancelled);
            }
            duration = wav.duration();
            if (duration <= 0) throw new IOException("模型没有生成有效音频");
            success = true;
        } finally { if (!success) Files.deleteIfExists(output.toPath()); }
        return new Synthesis((System.nanoTime() - start) / 1e9, duration, segments.size());
    }

    private long[][] generate(String text, Voice voice, AtomicBoolean cancelled,
                              Progress progress, NoiseSource source) throws Exception {
        long[] ids = tokenizer.encode(conditioned(voice.referenceText, text));
        int maxFrames = 500, length = ids.length + voice.codes.length;
        if (length + maxFrames > meta.getInt("max_sequence")) throw new IOException("文本或参考音色过长，请缩短后重试");
        FloatBuffer prefix = floats(length * 512);
        for (long id : ids) textEmbedding.copyRow((int) id, prefix);
        for (long[] row : voice.codes) {
            float[] sum = new float[512];
            for (int j = 0; j < 16; j++) audioEmbedding.addRow((int) row[j] + 1 + j * 2051, sum);
            prefix.put(sum);
        }
        prefix.rewind();
        List<long[]> generated = new ArrayList<>();
        OrtSession.Result previousBackbone = null, previousDepth = null;
        FloatBuffer noise = floats(16 * 2051);
        try (OnnxTensor initialHidden = OnnxTensor.createTensor(env, prefix, new long[]{1, length, 512});
             OnnxTensor initialPast = zeros(new long[]{8, 2, 1, 2, 0, 64});
             OnnxTensor noiseTensor = OnnxTensor.createTensor(env, noise, new long[]{16, 1, 2051})) {
            for (int step = 0; step < maxFrames; step++) {
                check(cancelled);
                long[] positions = step == 0 ? new long[length] : new long[]{length + step - 1L};
                if (step == 0) for (int i = 0; i < length; i++) positions[i] = i;
                try (OnnxTensor position = longs(positions, new long[]{1, positions.length})) {
                    Map<String, OnnxTensor> feed = new HashMap<>();
                    feed.put("hidden", previousDepth == null ? initialHidden : tensor(previousDepth, "next_hidden"));
                    feed.put("past", previousBackbone == null ? initialPast : tensor(previousBackbone, "present"));
                    feed.put("positions", position);
                    OrtSession.Result next = backbone.run(feed);
                    if (previousBackbone != null) previousBackbone.close(); previousBackbone = next;
                }
                if (previousDepth != null) { previousDepth.close(); previousDepth = null; }
                source.fill(noise);
                try (OnnxTensor allowEos = OnnxTensor.createTensor(env, new boolean[]{step > meta.getInt("min_frames")})) {
                    Map<String, OnnxTensor> feed = new HashMap<>();
                    feed.put("hidden", tensor(previousBackbone, "last_hidden"));
                    feed.put("noise", noiseTensor); feed.put("allow_eos", allowEos);
                    previousDepth = depth.run(feed);
                }
                LongBuffer codes = tensor(previousDepth, "codes").getLongBuffer();
                long[] frame = new long[16]; boolean eos = true;
                for (int j = 0; j < 16; j++) { long v = codes.get(j); eos &= v == 0; frame[j] = Math.max(0, Math.min(2047, v - 1)); }
                if (eos) {
                    if (generated.isEmpty()) throw new IOException("模型提前结束，没有生成语音");
                    return generated.toArray(new long[0][]);
                }
                generated.add(frame);
                if (step % 8 == 0) progress.update(String.format(Locale.CHINA, "已生成 %.1f 秒语音", generated.size() * 0.08));
            }
            throw new IOException("此段超过 40 秒生成上限，请缩短文本后重试");
        } finally {
            if (previousBackbone != null) previousBackbone.close();
            if (previousDepth != null) previousDepth.close();
        }
    }

    private void decode(long[][] reference, long[][] generated, WavWriter wav, AtomicBoolean cancelled) throws Exception {
        try (CodecState state = new CodecState()) {
            for (int start = 0; start < reference.length; start += 16) {
                check(cancelled); state.run(reference, start, Math.min(16, reference.length - start));
            }
            int remaining = generated.length * 1920 - 2400; // Same 100 ms tail trim as Python.
            for (int start = 0; start < generated.length; start += 32) {
                check(cancelled);
                FloatBuffer pcm = state.run(generated, start, Math.min(32, generated.length - start));
                int count = Math.max(0, Math.min(remaining, pcm.remaining()));
                wav.append(pcm, count); remaining -= count;
            }
            if (remaining > 0) throw new IOException("声音解码器返回的长度不足");
        }
    }

    private final class CodecState implements AutoCloseable {
        private final List<OnnxTensor> initial = new ArrayList<>();
        private OrtSession.Result previous;
        private int offset;
        CodecState() throws Exception {
            try {
                initial.add(zeros(new long[]{8, 2, 1, 16, 0, 64}));
                JSONArray states = codecMeta.getJSONArray("states");
                for (int i = 0; i < states.length(); i++) {
                    JSONArray a = states.getJSONObject(i).getJSONArray("shape"); long[] shape = new long[a.length()];
                    for (int j = 0; j < shape.length; j++) shape[j] = a.getLong(j);
                    initial.add(zeros(shape));
                }
            } catch (Exception e) { close(); throw e; }
        }
        FloatBuffer run(long[][] frames, int start, int count) throws Exception {
            long[] flat = new long[16 * count];
            for (int c = 0; c < 16; c++) for (int i = 0; i < count; i++) flat[c * count + i] = frames[start + i][c];
            try (OnnxTensor codes = longs(flat, new long[]{1, 16, count});
                 OnnxTensor position = longs(new long[]{offset}, new long[]{1})) {
                Map<String, OnnxTensor> feed = new HashMap<>();
                feed.put("codes", codes); feed.put("offset", position);
                feed.put("past", previous == null ? initial.get(0) : tensor(previous, "present"));
                for (int i = 1; i < initial.size(); i++)
                    feed.put("conv_" + (i - 1), previous == null ? initial.get(i) : tensor(previous, "next_conv_" + (i - 1)));
                OrtSession.Result next = codec.run(feed);
                if (previous != null) previous.close(); previous = next; offset += count;
                return tensor(previous, "audio").getFloatBuffer();
            }
        }
        public void close() { if (previous != null) previous.close(); for (OnnxTensor t : initial) t.close(); }
    }

    private static OnnxTensor tensor(OrtSession.Result result, String name) throws IOException {
        OnnxValue value = result.get(name).orElseThrow(() -> new IOException("模型缺少输出：" + name));
        if (!(value instanceof OnnxTensor)) throw new IOException("模型输出类型错误：" + name);
        return (OnnxTensor) value;
    }
    private static FloatBuffer floats(int count) { return ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer(); }
    private OnnxTensor zeros(long[] shape) throws OrtException {
        long count = 1; for (long d : shape) count *= d;
        return OnnxTensor.createTensor(env, floats(Math.toIntExact(count)), shape);
    }
    private OnnxTensor longs(long[] values, long[] shape) throws OrtException {
        LongBuffer buffer = ByteBuffer.allocateDirect(values.length * 8).order(ByteOrder.nativeOrder()).asLongBuffer();
        buffer.put(values).rewind(); return OnnxTensor.createTensor(env, buffer, shape);
    }
    private static void check(AtomicBoolean cancelled) throws InterruptedIOException {
        if (cancelled.get() || Thread.currentThread().isInterrupted()) throw new InterruptedIOException("已停止生成");
    }
    public static List<String> splitText(String text) {
        if (text == null || text.trim().isEmpty()) throw new IllegalArgumentException("请输入要合成的文字");
        if (text.codePointCount(0, text.length()) > 2000) throw new IllegalArgumentException("一次最多输入 2000 字");
        List<String> result = new ArrayList<>(); StringBuilder part = new StringBuilder(); int count = 0;
        for (int offset = 0; offset < text.length();) {
            int cp = text.codePointAt(offset); offset += Character.charCount(cp); part.appendCodePoint(cp); count++;
            if (count >= 80 || "。！？!?；;\n".indexOf(cp) >= 0) {
                if (!part.toString().trim().isEmpty()) result.add(part.toString().trim()); part.setLength(0); count = 0;
            }
        }
        if (!part.toString().trim().isEmpty()) result.add(part.toString().trim());
        if (result.isEmpty()) throw new IllegalArgumentException("请输入有效文字"); return result;
    }
    public void close() {
        for (OrtSession session : new OrtSession[]{codec, depth, backbone})
            if (session != null) try { session.close(); } catch (OrtException ignored) { }
        for (NpyMatrix matrix : new NpyMatrix[]{textEmbedding, audioEmbedding})
            if (matrix != null) try { matrix.close(); } catch (IOException ignored) { }
        codec = null; depth = null; backbone = null; textEmbedding = null; audioEmbedding = null; tokenizer = null;
    }
}
