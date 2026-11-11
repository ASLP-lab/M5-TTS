package com.m5.offline;

import android.media.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Decode local phone recordings; no recording permission or network connection is used. */
final class AndroidAudioReader {
    static ReferenceAudio.Samples read(File file, AtomicBoolean cancelled) throws IOException {
        if (ReferenceAudio.isWav(file)) return ReferenceAudio.readWav(file);
        MediaExtractor extractor = new MediaExtractor(); MediaCodec decoder = null;
        try {
            extractor.setDataSource(file.getPath()); MediaFormat format = null;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat candidate = extractor.getTrackFormat(i); String mime = candidate.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("audio/")) { extractor.selectTrack(i); format = candidate; break; }
            }
            if (format == null) throw new IOException("文件中没有音频，请选择 WAV、M4A 或 MP3 录音");
            if (format.containsKey(MediaFormat.KEY_DURATION) && format.getLong(MediaFormat.KEY_DURATION) > 30500000L)
                throw new IOException("参考录音需要为 1–30 秒");
            int rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE), channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            int encoding = AudioFormat.ENCODING_PCM_16BIT;
            decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
            decoder.configure(format, null, null, 0); decoder.start();
            boolean endedInput = false, endedOutput = false; long deadline = System.nanoTime() + 90_000_000_000L;
            float[] samples = new float[65536]; int count = 0; MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            while (!endedOutput) {
                if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                if (System.nanoTime() > deadline) throw new IOException("读取录音超时，请换一段较短的 WAV 录音");
                if (!endedInput) {
                    int index = decoder.dequeueInputBuffer(10000);
                    if (index >= 0) {
                        ByteBuffer buffer = decoder.getInputBuffer(index);
                        if (buffer == null) throw new IOException("无法读取音频解码缓冲区");
                        buffer.clear(); int size = extractor.readSampleData(buffer, 0);
                        if (size < 0) { decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); endedInput = true; }
                        else { decoder.queueInputBuffer(index, 0, size, Math.max(0, extractor.getSampleTime()), 0); extractor.advance(); }
                    }
                }
                int index = decoder.dequeueOutputBuffer(info, 10000);
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat decoded = decoder.getOutputFormat(); rate = decoded.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    channels = decoded.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                    encoding = decoded.containsKey(MediaFormat.KEY_PCM_ENCODING) ? decoded.getInteger(MediaFormat.KEY_PCM_ENCODING) : AudioFormat.ENCODING_PCM_16BIT;
                } else if (index >= 0) {
                    try {
                        if (rate < 8000 || rate > 192000 || channels < 1 || channels > 8) throw new IOException("录音采样参数不支持");
                        int bytes = encoding == AudioFormat.ENCODING_PCM_FLOAT ? 4 : encoding == AudioFormat.ENCODING_PCM_16BIT ? 2 : 0;
                        if (bytes == 0) throw new IOException("此录音编码暂不支持，请转换为 PCM WAV");
                        if (info.size > 0 && (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                            ByteBuffer buffer = decoder.getOutputBuffer(index);
                            if (buffer == null || info.size % (channels * bytes) != 0) throw new IOException("录音解码数据不完整");
                            buffer.position(info.offset); buffer.limit(info.offset + info.size); buffer.order(ByteOrder.LITTLE_ENDIAN);
                            int frames = info.size / (channels * bytes);
                            if (count + frames > (long) rate * ReferenceAudio.MAX_SECONDS) throw new IOException("参考录音需要为 1–30 秒");
                            if (count + frames > samples.length) samples = Arrays.copyOf(samples, Math.max(count + frames, samples.length * 2));
                            for (int frame = 0; frame < frames; frame++) {
                                float sum = 0;
                                for (int channel = 0; channel < channels; channel++) sum += bytes == 4 ? buffer.getFloat() : buffer.getShort() / 32768f;
                                samples[count++] = sum / channels;
                            }
                        }
                        endedOutput = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    } finally { decoder.releaseOutputBuffer(index, false); }
                }
            }
            if (count < rate) throw new IOException("参考录音需要为 1–30 秒");
            return new ReferenceAudio.Samples(Arrays.copyOf(samples, count), rate);
        } catch (IllegalArgumentException | IllegalStateException e) {
            throw new IOException("录音无法解码，请使用 WAV、M4A 或 MP3", e);
        } finally {
            if (decoder != null) { try { decoder.stop(); } catch (IllegalStateException ignored) { } decoder.release(); }
            extractor.release();
        }
    }
}
