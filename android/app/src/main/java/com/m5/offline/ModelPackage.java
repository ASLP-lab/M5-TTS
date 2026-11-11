package com.m5.offline;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** The download wrapper avoids browser ZIP previews; the enclosed ZIP is unchanged. */
final class ModelPackage {
    static final byte[] HEADER = "M5MODEL-ZIP-V1\r\n".getBytes(StandardCharsets.US_ASCII);

    /** Leave the stream positioned at the original ZIP and return the wrapper bytes consumed. */
    static int unwrap(PushbackInputStream in) throws IOException {
        byte[] prefix = new byte[HEADER.length]; int count = 0, n;
        while (count < prefix.length && (n = in.read(prefix, count, prefix.length - count)) != -1) count += n;
        if (count == HEADER.length && Arrays.equals(prefix, HEADER)) return HEADER.length;
        if (count >= 4 && prefix[0] == 'P' && prefix[1] == 'K'
                && ((prefix[2] == 3 && prefix[3] == 4) || (prefix[2] == 5 && prefix[3] == 6))) {
            in.unread(prefix, 0, count);
            return 0;
        }
        throw new IOException("请选择下载的 .m5model 模型文件或原始 ZIP，不要选择解压后的文件");
    }
}
