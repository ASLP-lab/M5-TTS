package com.m5.offline;

import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** PCM handling shared with host checks. Native-rate WAV input has no Android dependency. */
final class ReferenceAudio {
    static final int TARGET_RATE = 24000, MAX_SECONDS = 30, MAX_FILE_BYTES = 20 * 1024 * 1024;
    static final class Samples {
        final float[] mono; final int sampleRate;
        Samples(float[] mono, int rate) { this.mono = mono; sampleRate = rate; }
    }
    static boolean isWav(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(file))) {
            byte[] header = new byte[12]; if (in.read(header) != 12) return false;
            return header[0]=='R' && header[1]=='I' && header[2]=='F' && header[3]=='F'
                && header[8]=='W' && header[9]=='A' && header[10]=='V' && header[11]=='E';
        }
    }
    static Samples readWav(File file) throws IOException {
        try (RandomAccessFile in = new RandomAccessFile(file, "r")) {
            in.seek(12); int format = 0, channels = 0, rate = 0, bits = 0, align = 0;
            long dataOffset = -1, dataBytes = 0;
            while (in.getFilePointer() + 8 <= in.length()) {
                byte[] tag = new byte[4]; in.readFully(tag); long size = Integer.toUnsignedLong(Integer.reverseBytes(in.readInt()));
                long offset = in.getFilePointer(), next = offset + size + (size & 1);
                if (size > in.length() - offset) throw new IOException("WAV 文件下载不完整或已损坏");
                String name = new String(tag, java.nio.charset.StandardCharsets.US_ASCII);
                if (name.equals("fmt ")) {
                    if (size < 16 || size > 4096) throw new IOException("WAV 格式头无效");
                    format = Short.toUnsignedInt(Short.reverseBytes(in.readShort()));
                    channels = Short.toUnsignedInt(Short.reverseBytes(in.readShort()));
                    rate = Integer.reverseBytes(in.readInt()); in.skipBytes(4);
                    align = Short.toUnsignedInt(Short.reverseBytes(in.readShort())); bits = Short.toUnsignedInt(Short.reverseBytes(in.readShort()));
                    if (format == 65534 && size >= 40) { in.seek(offset + 24); format = Short.toUnsignedInt(Short.reverseBytes(in.readShort())); }
                } else if (name.equals("data")) { dataOffset = offset; dataBytes = size; }
                in.seek(next);
            }
            if (dataOffset < 0 || channels < 1 || channels > 8 || rate < 8000 || rate > 192000
                    || !((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32))
                    || align != channels * bits / 8 || dataBytes % align != 0 || dataBytes > MAX_FILE_BYTES)
                throw new IOException("不支持此 WAV 编码，请使用 PCM WAV、M4A 或 MP3");
            long count = dataBytes / align;
            if (count < rate || count > (long) MAX_SECONDS * rate) throw new IOException("参考录音需要为 1–30 秒");
            byte[] bytes = new byte[(int) dataBytes]; in.seek(dataOffset); in.readFully(bytes);
            float[] mono = new float[(int) count]; ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int frame = 0; frame < mono.length; frame++) {
                float sum = 0;
                for (int channel = 0; channel < channels; channel++) {
                    if (format == 3) sum += data.getFloat();
                    else if (bits == 8) sum += ((data.get() & 255) - 128) / 128f;
                    else if (bits == 16) sum += data.getShort() / 32768f;
                    else if (bits == 24) { int value = (data.get() & 255) | ((data.get() & 255) << 8) | (data.get() << 16); sum += value / 8388608f; }
                    else sum += data.getInt() / 2147483648f;
                }
                mono[frame] = sum / channels;
            }
            return new Samples(mono, rate);
        }
    }
    static float[] prepare(Samples source, AtomicBoolean cancelled) throws IOException {
        if (source.sampleRate < 8000 || source.sampleRate > 192000 || source.mono.length < source.sampleRate
                || source.mono.length > (long) MAX_SECONDS * source.sampleRate) throw new IOException("参考录音需要为 1–30 秒");
        for (float value : source.mono) if (!Float.isFinite(value)) throw new IOException("录音包含无效音频数据");
        float[] audio = source.sampleRate == TARGET_RATE ? source.mono.clone() : resample(source.mono, source.sampleRate, cancelled);
        float peak = 0; for (float value : audio) peak = Math.max(peak, Math.abs(value));
        if (peak < 1e-6) throw new IOException("录音没有有效声音，请选择有清晰人声的录音");
        float divisor = peak + 1e-8f; for (int i = 0; i < audio.length; i++) audio[i] /= divisor;
        float threshold = audio.length > 10 * TARGET_RATE ? 0.02f : 0.015f;
        List<int[]> ranges = new ArrayList<>(); int first = -1, previous = -1;
        for (int i = 0; i < audio.length; i++) if (Math.abs(audio[i]) > threshold) {
            if (first < 0) first = i;
            else if (i - previous > 3600) { ranges.add(new int[]{Math.max(0, first - 1200), Math.min(audio.length, previous + 1201)}); first = i; }
            previous = i;
        }
        if (first < 0) throw new IOException("录音没有有效声音");
        int finalEnd = audio.length - previous - 1 >= 3600 ? previous + 1 : audio.length;
        ranges.add(new int[]{Math.max(0, first - 1200), Math.min(audio.length, finalEnd + 1200)});
        int count = 0; for (int[] range : ranges) count += range[1] - range[0];
        float[] result = new float[count]; int offset = 0;
        for (int[] range : ranges) { int n = range[1] - range[0]; System.arraycopy(audio, range[0], result, offset, n); offset += n; }
        if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
        return result;
    }
    private static float[] resample(float[] source, int rate, AtomicBoolean cancelled) throws IOException {
        double ratio = (double) TARGET_RATE / rate, cutoff = Math.min(1, ratio) * 0.94;
        int half = (int) Math.ceil(16 / cutoff), width = half * 2, phases = 1024;
        float[][] kernel = new float[phases][width];
        for (int phase = 0; phase < phases; phase++) {
            double fraction = (double) phase / phases, total = 0;
            for (int j = 0; j < width; j++) {
                double x = j - half + 1 - fraction, z = Math.PI * x * cutoff;
                double sinc = Math.abs(z) < 1e-12 ? 1 : Math.sin(z) / z;
                double window = Math.abs(x) >= half ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * x / half);
                kernel[phase][j] = (float) (cutoff * sinc * window); total += kernel[phase][j];
            }
            for (int j = 0; j < width; j++) kernel[phase][j] /= total;
        }
        float[] output = new float[(int) ((source.length * (long) TARGET_RATE + rate - 1) / rate)];
        for (int i = 0; i < output.length; i++) {
            if ((i & 16383) == 0 && cancelled.get()) throw new InterruptedIOException("已取消新增音色");
            double position = (double) i * rate / TARGET_RATE; int center = (int) position;
            float[] weights = kernel[Math.min(phases - 1, (int) ((position - center) * phases))];
            double sum = 0;
            for (int j = 0; j < width; j++) { int index = center + j - half + 1; if (index >= 0 && index < source.length) sum += source[index] * weights[j]; }
            output[i] = (float) sum;
        }
        return output;
    }
}
