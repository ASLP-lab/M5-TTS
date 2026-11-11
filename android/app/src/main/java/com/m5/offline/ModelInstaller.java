package com.m5.offline;

import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.*;

/** Extracts only the known bundle files, verifies hashes, then activates atomically. */
final class ModelInstaller {
    static final Set<String> REQUIRED = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
        "backbone.onnx", "depth_frame.onnx", "codec_decoder_cached.onnx", "codec_cached.json",
        "text_embeddings.npy", "audio_embeddings.npy", "tokenizer.bin", "voices.json", "manifest.json")));
    static File install(File zipPath, File parent, AtomicBoolean cancelled, M5Engine.Progress progress) throws Exception {
        File stage = new File(parent, "m5-staging-" + UUID.randomUUID());
        if (!stage.mkdir()) throw new IOException("无法创建模型目录");
        try (ZipFile zip = new ZipFile(zipPath)) {
            Map<String, ZipEntry> entries = bundleEntries(zip);
            ZipEntry bundleEntry = entries.get("bundle.json");
            // Some ZIP readers do not expose an entry size. Bound the actual bytes below.
            if (bundleEntry.getSize() == 0 || bundleEntry.getSize() > 65536)
                throw new IOException("模型包清单 bundle.json 的大小不正确，请重新下载模型包");
            byte[] manifestBytes;
            try (InputStream in = zip.getInputStream(bundleEntry)) { manifestBytes = limited(in, 65536); }
            JSONObject bundle = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
            if (!"m5-android-offline-v1".equals(bundle.getString("format")) || !"fp32".equals(bundle.getString("precision")))
                throw new IOException("不兼容的模型包版本");
            JSONObject spec = bundle.getJSONObject("files");
            Set<String> declared = new HashSet<>();
            for (Iterator<String> keys = spec.keys(); keys.hasNext();) declared.add(keys.next());
            if (!declared.equals(REQUIRED)) throw new IOException("模型包文件列表不完整");
            long total = 0;
            for (String name : REQUIRED) {
                long size = spec.getJSONObject(name).getLong("size");
                if (size < 1 || size > 600L * 1024 * 1024) throw new IOException("模型文件尺寸不合理");
                total += size;
            }
            if (total > 1600L * 1024 * 1024 || parent.getUsableSpace() < total + 128L * 1024 * 1024)
                throw new IOException("存储空间不足，请清理至少 3GB 空间后重试");
            long copied = 0, lastProgress = 0; byte[] block = new byte[1024 * 1024];
            for (String name : new TreeSet<>(REQUIRED)) {
                if (cancelled.get()) throw new InterruptedIOException("已取消导入");
                JSONObject item = spec.getJSONObject(name); long expected = item.getLong("size"), count = 0;
                ZipEntry entry = entries.get(name);
                if (entry == null || (entry.getSize() >= 0 && entry.getSize() != expected))
                    throw new IOException("模型文件缺失或尺寸错误：" + name + "，请重新下载模型包");
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                try (InputStream in = zip.getInputStream(entry); OutputStream out = new FileOutputStream(new File(stage, name))) {
                    int n;
                    while ((n = in.read(block)) != -1) {
                        if (cancelled.get()) throw new InterruptedIOException("已取消导入");
                        count += n; copied += n;
                        if (count > expected) throw new IOException("模型文件超出声明大小");
                        digest.update(block, 0, n); out.write(block, 0, n);
                        long now = System.nanoTime();
                        if (now - lastProgress > 250000000L) {
                            progress.update("正在校验并导入模型… " + copied / 1048576 + "/" + total / 1048576 + " MB"); lastProgress = now;
                        }
                    }
                }
                if (count != expected || !hex(digest.digest()).equals(item.getString("sha256")))
                    throw new IOException("模型校验失败：" + name + "，请重新下载模型包");
            }
            Files.write(new File(stage, "bundle.json").toPath(), manifestBytes);
            if (cancelled.get()) throw new InterruptedIOException("已取消导入");
            File current = new File(parent, "m5-model"), backup = new File(parent, "m5-old-" + UUID.randomUUID());
            boolean hadPrevious = current.exists();
            if (hadPrevious && !current.renameTo(backup)) throw new IOException("无法替换旧模型");
            if (!stage.renameTo(current)) {
                if (hadPrevious) backup.renameTo(current);
                throw new IOException("无法启用模型");
            }
            if (hadPrevious) deleteTree(backup);
            return current;
        } catch (ZipException e) {
            throw new IOException("模型 ZIP 已损坏或下载未完成，请重新下载模型包", e);
        } finally { if (stage.exists()) deleteTree(stage); }
    }
    /** Record the actual selected archive, without extracting any file. */
    static JSONObject inspect(File path) throws IOException, JSONException {
        JSONObject details = new JSONObject().put("bytes", path.length());
        try (ZipFile zip = new ZipFile(path)) {
            JSONArray entries = new JSONArray();
            Enumeration<? extends ZipEntry> all = zip.entries();
            while (all.hasMoreElements() && entries.length() < 20) {
                ZipEntry entry = all.nextElement(); String name = entry.getName();
                entries.put(new JSONObject().put("name", name.substring(0, Math.min(name.length(), 160)))
                    .put("bytes", entry.getSize()));
            }
            return details.put("entry_count", zip.size()).put("entries", entries);
        } catch (ZipException e) {
            throw new IOException("模型 ZIP 已损坏或下载未完成，请重新下载模型包", e);
        }
    }
    private static Map<String, ZipEntry> bundleEntries(ZipFile zip) throws IOException {
        List<? extends ZipEntry> all = Collections.list(zip.entries());
        String prefix = null;
        for (ZipEntry entry : all) {
            String name = entry.getName();
            if (name.equals("bundle.json")) {
                if (prefix != null) throw new IOException("模型包包含多个 bundle.json 清单");
                prefix = "";
            } else if (name.endsWith("/bundle.json")) {
                String folder = name.substring(0, name.length() - "/bundle.json".length());
                if (!folder.isEmpty() && !folder.equals(".") && !folder.equals("..")
                        && folder.indexOf('/') < 0 && folder.indexOf('\\') < 0 && folder.indexOf('\0') < 0) {
                    if (prefix != null) throw new IOException("模型包包含多个 bundle.json 清单");
                    prefix = folder + "/";
                }
            }
        }
        if (prefix == null) throw new IOException("所选 ZIP 中缺少 bundle.json 模型清单，请重新下载配套模型包");
        Map<String, ZipEntry> entries = new HashMap<>();
        for (ZipEntry entry : all) {
            String raw = entry.getName();
            if (!prefix.isEmpty() && raw.equals(prefix) && entry.isDirectory()) continue;
            if (!raw.startsWith(prefix)) throw new IOException("模型包包含未知文件：" + raw);
            String name = raw.substring(prefix.length());
            if ((!REQUIRED.contains(name) && !name.equals("bundle.json")) || entries.put(name, entry) != null)
                throw new IOException("模型包包含未知或重复文件：" + raw);
        }
        return entries;
    }
    private static byte[] limited(InputStream in, int limit) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(); byte[] b = new byte[4096]; int n;
        while ((n = in.read(b)) != -1) { if (out.size() + n > limit) throw new IOException("文件过大"); out.write(b, 0, n); }
        return out.toByteArray();
    }
    private static String hex(byte[] bytes) {
        StringBuilder s = new StringBuilder(); for (byte b : bytes) s.append(String.format(Locale.ROOT, "%02x", b & 255)); return s.toString();
    }
    static void deleteTree(File file) throws IOException {
        File[] children = file.listFiles(); if (children != null) for (File child : children) deleteTree(child);
        Files.deleteIfExists(file.toPath());
    }
}
