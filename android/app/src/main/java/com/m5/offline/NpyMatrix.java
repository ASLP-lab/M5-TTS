package com.m5.offline;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.regex.*;

/** Maps FP32 embeddings without copying hundreds of MB into the Java heap. */
final class NpyMatrix implements AutoCloseable {
    private final RandomAccessFile file;
    private final FloatBuffer data;
    final int rows, columns;

    NpyMatrix(File path) throws IOException {
        file = new RandomAccessFile(path, "r");
        byte[] magic = new byte[6]; file.readFully(magic);
        if (!java.util.Arrays.equals(magic, new byte[]{(byte)147,78,85,77,80,89})) throw new IOException("无效 embedding 文件");
        int version = file.readUnsignedByte(); file.readUnsignedByte();
        int headerLength;
        if (version == 1) headerLength = file.readUnsignedByte() | file.readUnsignedByte() << 8;
        else if (version == 2) headerLength = Integer.reverseBytes(file.readInt());
        else throw new IOException("不支持的 NPY 版本");
        if (headerLength < 1 || headerLength > 65536) throw new IOException("无效 NPY 头");
        byte[] bytes = new byte[headerLength]; file.readFully(bytes);
        String header = new String(bytes, StandardCharsets.US_ASCII);
        Matcher shape = Pattern.compile("'shape':\\s*\\((\\d+),\\s*(\\d+)\\)").matcher(header);
        if (!header.contains("'<f4'") || !header.contains("'fortran_order': False") || !shape.find())
            throw new IOException("需要 C-order FP32 二维 embedding");
        rows = Integer.parseInt(shape.group(1)); columns = Integer.parseInt(shape.group(2));
        long offset = file.getFilePointer(), length = (long) rows * columns * 4;
        if (length <= 0 || file.length() != offset + length) throw new IOException("Embedding 尺寸错误");
        data = file.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, length).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    void copyRow(int row, FloatBuffer target) {
        if (row < 0 || row >= rows) throw new IllegalArgumentException("Token 超出词表");
        int start = row * columns; for (int i = 0; i < columns; i++) target.put(data.get(start + i));
    }
    void addRow(int row, float[] target) {
        if (row < 0 || row >= rows) throw new IllegalArgumentException("音频 token 超出词表");
        int start = row * columns; for (int i = 0; i < columns; i++) target[i] += data.get(start + i);
    }
    public void close() throws IOException { file.close(); }
}
