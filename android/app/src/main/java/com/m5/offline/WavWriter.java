package com.m5.offline;

import java.io.*;
import java.nio.FloatBuffer;

final class WavWriter implements AutoCloseable {
    private final RandomAccessFile file;
    private long samples;
    WavWriter(File path) throws IOException { file = new RandomAccessFile(path, "rw"); file.setLength(0); file.write(new byte[44]); }
    void append(FloatBuffer pcm, int length) throws IOException {
        byte[] bytes = new byte[length * 2];
        for (int i = 0; i < length; i++) {
            float value = pcm.get(i);
            if (!Float.isFinite(value)) throw new IOException("生成音频出现无效数值");
            int sample = (int) Math.floor(Math.max(-1, Math.min(32767.0 / 32768, value)) * 32768.0);
            bytes[i * 2] = (byte) sample; bytes[i * 2 + 1] = (byte) (sample >> 8);
        }
        file.write(bytes); samples += length;
    }
    double duration() { return samples / 24000.0; }
    private void le32(long v) throws IOException { file.writeInt(Integer.reverseBytes((int) v)); }
    private void le16(int v) throws IOException { file.writeShort(Short.reverseBytes((short) v)); }
    public void close() throws IOException {
        try {
            file.seek(0); file.writeBytes("RIFF"); le32(36 + samples * 2); file.writeBytes("WAVEfmt ");
            le32(16); le16(1); le16(1); le32(24000); le32(48000); le16(2); le16(16);
            file.writeBytes("data"); le32(samples * 2);
        } finally { file.close(); }
    }
}
