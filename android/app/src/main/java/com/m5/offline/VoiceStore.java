package com.m5.offline;

import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** User voices live beside the model directory and never rewrite model files. */
final class VoiceStore {
    static final int MAX_BYTES = 512 * 1024;
    static final class Entry {
        final String id; final M5Engine.Voice voice; final File file;
        Entry(String id, M5Engine.Voice voice, File file) { this.id = id; this.voice = voice; this.file = file; }
        boolean custom() { return file != null; }
        @Override public String toString() { return custom() ? "自定义 · " + voice.name : voice.name; }
    }
    static List<Entry> load(File model, File directory) throws Exception {
        return load(model, directory, null);
    }
    static List<Entry> load(File model, File directory, JSONObject replacement) throws Exception {
        JSONObject builtin = M5Engine.json(new File(model, "voices.json"));
        String identity = builtin.getString("encoder_sha256");
        if (replacement != null) {
            if (!"m5-builtin-voices-v1".equals(replacement.optString("format"))
                    || !identity.equals(replacement.optString("encoder_sha256"))
                    || replacement.optInt("sample_rate") != 24000 || replacement.optInt("hop_length") != 1920)
                throw new IOException("内置音色与当前模型不兼容");
            builtin = replacement;
        }
        List<Entry> entries = new ArrayList<>(); JSONArray list = builtin.getJSONArray("voices");
        Set<String> ids = new HashSet<>();
        for (int i = 0; i < list.length(); i++) {
            JSONObject row = list.getJSONObject(i);
            String id = replacement == null ? "builtin-" + i : row.getString("id");
            if (id.isEmpty() || !ids.add(id)) throw new IOException("内置音色编号重复或为空");
            M5Engine.Voice voice = new M5Engine.Voice(row);
            if (replacement != null) {
                validateLabel(voice.name, voice.referenceText);
                if (voice.codes.length > 375) throw new IOException("内置音色超过 30 秒");
            }
            entries.add(new Entry(id, voice, null));
        }
        if (replacement != null && (entries.isEmpty() || !entries.get(0).id.equals(replacement.getString("default_voice_id"))))
            throw new IOException("内置音色缺少默认项");
        File[] files = directory.listFiles((d, name) -> name.endsWith(".m5voice"));
        if (files != null) {
            Arrays.sort(files, Comparator.comparing(File::getName));
            for (File file : files) {
                try { entries.add(new Entry(file.getName(), decode(read(file), identity), file)); }
                catch (IOException | JSONException ignored) { /* One bad voice must not hide the built-in voices. */ }
            }
        }
        return entries;
    }
    static JSONObject bundled(InputStream in) throws IOException, JSONException {
        return new JSONObject(new String(read(in, 8 * 1024 * 1024), StandardCharsets.UTF_8));
    }
    static M5Engine.Voice decode(byte[] bytes, String identity) throws IOException, JSONException {
        if (bytes.length == 0 || bytes.length > MAX_BYTES) throw new IOException("音色文件为空或过大，请选择 .m5voice 音色包");
        JSONObject json = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
        if (!"m5-voice-v1".equals(json.optString("format"))) throw new IOException("请选择配套的 .m5voice 音色包");
        if (!identity.equals(json.optString("encoder_sha256"))) throw new IOException("此音色与当前模型不兼容");
        if (json.optInt("sample_rate") != 24000 || json.optInt("hop_length") != 1920)
            throw new IOException("音色采样参数不兼容");
        String name = json.getString("name").trim(), text = json.getString("reference_text").trim();
        validateLabel(name, text);
        M5Engine.Voice voice = new M5Engine.Voice(json.put("name", name).put("reference_text", text));
        if (voice.codes.length > 375) throw new IOException("参考音色最长为 30 秒");
        return voice;
    }
    static void validateLabel(String name, String text) throws IOException {
        if (name.trim().isEmpty() || name.codePointCount(0, name.length()) > 40) throw new IOException("请输入音色名称，最多 40 字");
        if (text.trim().isEmpty() || text.codePointCount(0, text.length()) > 500) throw new IOException("请填写录音原文，最多 500 字");
        try { M5Engine.conditioned(text, "测试"); } catch (IllegalArgumentException e) { throw new IOException(e.getMessage(), e); }
    }
    static JSONObject document(String name, String text, long[][] codes, String identity) throws JSONException, IOException {
        validateLabel(name, text);
        JSONArray rows = new JSONArray();
        for (long[] row : codes) { JSONArray values = new JSONArray(); for (long value : row) values.put(value); rows.put(values); }
        return new JSONObject().put("format", "m5-voice-v1").put("encoder_sha256", identity)
            .put("sample_rate", 24000).put("hop_length", 1920).put("name", name.trim())
            .put("reference_text", text.trim()).put("codes", rows);
    }
    static File save(File directory, byte[] bytes, String identity) throws Exception {
        M5Engine.Voice voice = decode(bytes, identity);
        byte[] canonical = document(voice.name, voice.referenceText, voice.codes, identity).toString().getBytes(StandardCharsets.UTF_8);
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("无法创建音色目录");
        File target = new File(directory, UUID.randomUUID() + ".m5voice"), temporary = new File(directory, target.getName() + ".part");
        try {
            try (FileOutputStream out = new FileOutputStream(temporary)) { out.write(canonical); out.getFD().sync(); }
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE);
        } finally { Files.deleteIfExists(temporary.toPath()); }
        return target;
    }
    static byte[] read(File file) throws IOException {
        try (InputStream in = new FileInputStream(file)) { return read(in); }
    }
    static byte[] read(InputStream in) throws IOException {
        return read(in, MAX_BYTES);
    }
    private static byte[] read(InputStream in, int maximum) throws IOException {
        if (in == null) throw new IOException("无法读取音色文件");
        ByteArrayOutputStream out = new ByteArrayOutputStream(); byte[] block = new byte[8192]; int n;
        while ((n = in.read(block)) != -1) {
            if (out.size() + n > maximum) throw new IOException("音色包过大，请选择 .m5voice 文件，录音请使用“新增音色”");
            out.write(block, 0, n);
        }
        return out.toByteArray();
    }
}
