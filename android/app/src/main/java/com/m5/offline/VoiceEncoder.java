package com.m5.offline;

import ai.onnxruntime.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

final class VoiceEncoder {
    static final String SHA256 = "ae0364957dd52df283e88baea8ea09d1f1f34b579cd53d2806b38290ebc03b57";
    static final long MODEL_BYTES = 190632314L;
    static long[][] encode(File encoder, float[] audio, AtomicBoolean cancelled, M5Engine.Progress progress) throws Exception {
        if (audio.length < 1 || audio.length > 30 * 24000) throw new IOException("参考录音处理后的长度需要为 30 秒以内");
        if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(4); options.setInterOpNumThreads(1); options.setCPUArenaAllocator(false);
            options.addConfigEntry("session.intra_op.allow_spinning", "0");
            options.addConfigEntry("session.inter_op.allow_spinning", "0");
            progress.update("正在加载音色编码器…");
            try (OrtSession session = env.createSession(encoder.getPath(), options);
                    OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), new long[]{1, audio.length})) {
                progress.update("正在从录音创建音色…");
                if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                try (OrtSession.Result output = session.run(Collections.singletonMap("audio", input))) {
                    if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                    OnnxTensor tensor = (OnnxTensor) output.get(0); long[] shape = tensor.getInfo().getShape();
                    int frames = (audio.length + 1919) / 1920;
                    if (shape.length != 3 || shape[0] != 1 || shape[1] != 16 || shape[2] < frames)
                        throw new IOException("音色编码结果维度不正确");
                    LongBuffer values = tensor.getLongBuffer(); long[][] codes = new long[frames][16];
                    for (int t = 0; t < frames; t++) for (int book = 0; book < 16; book++) {
                        long value = values.get((int) (book * shape[2] + t));
                        if (value < 0 || value > 2047) throw new IOException("音色编码结果无效");
                        codes[t][book] = value;
                    }
                    return codes;
                }
            }
        }
    }
}
