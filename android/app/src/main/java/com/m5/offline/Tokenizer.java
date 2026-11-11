package com.m5.offline;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;

/** Exact byte-level BPE pipeline exported from the checkpoint tokenizer. */
public final class Tokenizer {
    private final Map<String, Integer> vocab = new HashMap<>(180000);
    private final Trie specials = new Trie();
    private final PairTable merges;
    private final Pattern split;
    private final int bos, eos;
    private final char[] byteChars = new char[256];
    private final int[] byteIds = new int[256];

    public Tokenizer(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            byte[] magic = new byte[5]; in.readFully(magic);
            if (!Arrays.equals(magic, "M5TK1".getBytes(StandardCharsets.US_ASCII)))
                throw new IOException("不支持的分词文件");
            bos = in.readInt(); eos = in.readInt();
            // Spell out Unicode White_Space so desktop Java and Android ICU agree.
            String whitespace = "[\\t\\n\\x0B\\f\\r\\x{85}\\p{Z}]";
            String regex = readString(in).replace("\\s", whitespace)
                .replace("\\S", "[^\\t\\n\\x0B\\f\\r\\x{85}\\p{Z}]");
            split = Pattern.compile(regex);
            int count = bounded(in.readInt(), 1, 200000);
            for (int i = 0; i < count; i++) vocab.put(readString(in), in.readInt());
            for (int i = 0; i < 256; i++) { byteChars[i] = (char) in.readInt(); byteIds[i] = in.readInt(); }
            int mergeCount = bounded(in.readInt(), 1, 1000000);
            merges = new PairTable(mergeCount);
            for (int rank = 0; rank < mergeCount; rank++)
                merges.put(in.readInt(), in.readInt(), rank, in.readInt());
            int specialCount = bounded(in.readInt(), 1, 10000);
            for (int i = 0; i < specialCount; i++) specials.put(readString(in), in.readInt());
            if (in.read() != -1) throw new IOException("分词文件包含多余数据");
        }
    }

    public long[] encode(String text) {
        List<Long> out = new ArrayList<>(); out.add((long) bos);
        int plain = 0, at = 0;
        while (at < text.length()) {
            Trie node = specials; int end = -1, id = -1;
            for (int j = at; j < text.length(); j++) {
                node = node.children.get(text.charAt(j)); if (node == null) break;
                if (node.id >= 0) { end = j + 1; id = node.id; }
            }
            if (end >= 0) {
                encodePlain(text.substring(plain, at), out); out.add((long) id);
                at = end; plain = end;
            } else at++;
        }
        encodePlain(text.substring(plain), out); out.add((long) eos);
        long[] result = new long[out.size()];
        for (int i = 0; i < result.length; i++) result[i] = out.get(i);
        return result;
    }

    private void encodePlain(String text, List<Long> out) {
        Matcher matcher = split.matcher(text); int end = 0;
        while (matcher.find()) {
            if (matcher.start() > end) encodePiece(text.substring(end, matcher.start()), out);
            encodePiece(matcher.group(), out); end = matcher.end();
        }
        if (end < text.length()) encodePiece(text.substring(end), out);
    }

    private void encodePiece(String piece, List<Long> out) {
        if (piece.isEmpty()) return;
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        char[] encoded = new char[bytes.length]; int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            int b = bytes[i] & 255; encoded[i] = byteChars[b]; ids[i] = byteIds[b];
        }
        Integer direct = vocab.get(new String(encoded));
        if (direct != null) { out.add(direct.longValue()); return; } // ignore_merges=true
        int size = ids.length;
        while (size > 1) {
            int best = -1, rank = Integer.MAX_VALUE, mergedId = -1;
            for (int i = 0; i < size - 1; i++) {
                long value = merges.get(ids[i], ids[i + 1]);
                if (value >= 0 && (int) (value >>> 32) < rank) {
                    best = i; rank = (int) (value >>> 32); mergedId = (int) value;
                }
            }
            if (best < 0) break;
            ids[best] = mergedId;
            System.arraycopy(ids, best + 2, ids, best + 1, size - best - 2); size--;
        }
        for (int i = 0; i < size; i++) out.add((long) ids[i]);
    }

    private static int bounded(int value, int min, int max) throws IOException {
        if (value < min || value > max) throw new IOException("分词文件尺寸错误"); return value;
    }
    private static String readString(DataInputStream in) throws IOException {
        byte[] data = new byte[bounded(in.readInt(), 0, 100000)]; in.readFully(data);
        return new String(data, StandardCharsets.UTF_8);
    }
    private static final class Trie {
        final Map<Character, Trie> children = new HashMap<>(); int id = -1;
        void put(String value, int tokenId) {
            Trie node = this;
            for (int i = 0; i < value.length(); i++) node = node.children.computeIfAbsent(value.charAt(i), k -> new Trie());
            node.id = tokenId;
        }
    }
    private static final class PairTable {
        final long[] keys, values; final int mask;
        PairTable(int n) { int cap = 1; while (cap < n * 2) cap <<= 1; keys = new long[cap]; values = new long[cap]; mask = cap - 1; }
        int slot(long key) { long h = key * 0x9E3779B97F4A7C15L; h ^= h >>> 32; return (int) h & mask; }
        void put(int a, int b, int rank, int id) {
            long key = (((long) a << 32) | (b & 0xffffffffL)) + 1; int s = slot(key);
            while (keys[s] != 0 && keys[s] != key) s = (s + 1) & mask;
            // Hugging Face keeps the first rank if a pair occurs more than once.
            if (keys[s] == 0) { keys[s] = key; values[s] = ((long) rank << 32) | (id & 0xffffffffL); }
        }
        long get(int a, int b) {
            long key = (((long) a << 32) | (b & 0xffffffffL)) + 1; int s = slot(key);
            while (keys[s] != 0) { if (keys[s] == key) return values[s]; s = (s + 1) & mask; }
            return -1;
        }
    }
}
