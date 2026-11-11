package com.m5.offline;

import android.app.*;
import android.content.*;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Insets;
import android.graphics.Typeface;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.*;
import android.provider.OpenableColumns;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final int IMPORT_MODEL = 10, SAVE_AUDIO = 11, ADD_VOICE = 12, IMPORT_VOICE = 13, EXPORT_VOICE = 14;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final JSONObject diagnostics = new JSONObject();
    private TextView status, modelStatus, result;
    private TextView tnPreview;
    private CheckBox tnEnabled;
    private Spinner numberMode;
    private TextNormalizer normalizer;
    private String normalizedPreview = "";
    private EditText text;
    private Spinner voices, threads;
    private Button importButton, generateButton, cancelButton, playButton, saveButton;
    private Button addVoiceButton, importVoiceButton, exportVoiceButton, deleteVoiceButton;
    private List<VoiceStore.Entry> voiceEntries = new ArrayList<>();
    private byte[] pendingVoiceExport;
    private ProgressBar progress;
    private volatile boolean busy, destroyed;
    private M5Engine engine;
    private int loadedThreads;
    private MediaPlayer player;
    private long lastPssSample;
    private long peakPssKb;
    private File modelRoot, audioFile, voiceRoot;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        modelRoot = new File(getFilesDir(), "m5-model"); audioFile = new File(getFilesDir(), "m5-last.wav");
        voiceRoot = new File(getFilesDir(), "m5-user-voices");
        LinearLayout content = new LinearLayout(this); content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(20), dp(20), dp(20), dp(28));
        ScrollView scroll = new ScrollView(this); scroll.setFillViewport(true); scroll.addView(content);
        scroll.setBackgroundColor(Color.rgb(246,248,251));
        scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                Insets safe = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            } else view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });
        setContentView(scroll);
        label(content, "M5 离线语音", 28, true);
        label(content, "文字在手机上合成为语音 · 无需联网", 14, false);
        gap(content, 18);
        importButton = button(content, "导入离线模型", v -> pickModel());
        modelStatus = label(content, "首次使用，请导入下载的 .m5model 模型文件。", 13, false);
        gap(content, 12); label(content, "音色", 15, true);
        voices = new Spinner(this); content.addView(voices);
        voices.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) { updateVoiceButtons(); }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) { updateVoiceButtons(); }
        });
        LinearLayout voiceActions = new LinearLayout(this); content.addView(voiceActions);
        addVoiceButton = button(voiceActions, "新增音色", v -> pickVoice(ADD_VOICE));
        importVoiceButton = button(voiceActions, "导入音色包", v -> pickVoice(IMPORT_VOICE));
        LinearLayout voiceTools = new LinearLayout(this); content.addView(voiceTools);
        exportVoiceButton = button(voiceTools, "导出音色", v -> exportVoice());
        deleteVoiceButton = button(voiceTools, "删除音色", v -> deleteVoice());
        label(content, "新增音色：选择 1–30 秒的单人录音，填写录音原文，在手机上离线创建。", 12, false);
        label(content, "生成速度", 15, true);
        threads = new Spinner(this);
        threads.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
            new String[]{"2 线程 · 较省电", "4 线程 · 默认", "8 线程 · 尝试更快"}));
        threads.setSelection(1); content.addView(threads);
        gap(content, 12); label(content, "要合成的文字", 15, true);
        text = new EditText(this); text.setTextSize(17); text.setMinLines(4); text.setMaxLines(10);
        text.setGravity(Gravity.TOP); text.setHint("输入中文，最多 2000 字");
        text.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        text.setText(state == null ? "你好，这是在手机上运行的离线语音合成测试。" : state.getString("text", ""));
        content.addView(text, new LinearLayout.LayoutParams(-1, -2));
        tnEnabled = new CheckBox(this); tnEnabled.setText("文本规范化 TN");
        tnEnabled.setChecked(state == null || state.getBoolean("tn", true)); content.addView(tnEnabled);
        numberMode = new Spinner(this);
        numberMode.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item,
            new String[]{"数字读法 · 自动", "普通数字 · 按数值", "普通数字 · 逐位读"}));
        numberMode.setSelection(state == null ? 0 : state.getInt("number_mode", 0)); content.addView(numberMode);
        label(content, "自动模式：普通数字按数值读，验证码和号码逐位读。日期、金额和单位使用对应读法。", 12, false);
        tnPreview = label(content, "正在准备规范化预览…", 13, false); tnPreview.setMaxLines(5);
        tnPreview.setEllipsize(android.text.TextUtils.TruncateAt.END);
        tnPreview.setOnClickListener(v -> new AlertDialog.Builder(this).setTitle("实际合成文本")
            .setMessage(normalizedPreview).setPositiveButton("关闭", null).show());
        tnEnabled.setOnCheckedChangeListener((v, enabled) -> updateTnPreview());
        numberMode.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> p, View v, int i, long id) { updateTnPreview(); }
            @Override public void onNothingSelected(android.widget.AdapterView<?> p) { }
        });
        text.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) { }
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) { }
            @Override public void afterTextChanged(Editable value) { updateTnPreview(); }
        });
        generateButton = button(content, "生成语音", v -> generate());
        cancelButton = button(content, "停止", v -> { cancelled.set(true); updateStatus("正在停止，请稍候…"); });
        progress = new ProgressBar(this); content.addView(progress);
        status = label(content, "准备就绪", 15, false);
        result = label(content, "", 14, false);
        LinearLayout actions = new LinearLayout(this); content.addView(actions);
        playButton = button(actions, "播放", v -> play());
        saveButton = button(actions, "保存 WAV", v -> {
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("audio/wav")
                .addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE, "m5-" + System.currentTimeMillis() + ".wav");
            startActivityForResult(intent, SAVE_AUDIO);
        });
        button(content, "复制诊断信息", v -> {
            ((ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("M5 诊断", diagnosticText()));
            Toast.makeText(this, "已复制，可粘贴到聊天中反馈", Toast.LENGTH_SHORT).show();
        });
        label(content, "生成时请保持此页打开。首次生成需要加载模型，请耐心等待。", 12, false);
        populateVoices(); setBusy(false);
        try {
            diagnostics.put("app_version", BuildConfig.VERSION_NAME); diagnostics.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            diagnostics.put("android_sdk", Build.VERSION.SDK_INT); diagnostics.put("abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)));
            diagnostics.put("runtime", "onnxruntime-1.23.2-cpu-fp32");
            ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
            ((ActivityManager)getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(info);
            diagnostics.put("total_memory_mib", info.totalMem / 1048576);
        } catch (JSONException ignored) { }
        try (InputStream in = getAssets().open("tn/rules.json")) {
            normalizer = new TextNormalizer(in); updateTnPreview();
        } catch (Exception e) { failure(e); }
    }

    private String numberMode() { return new String[]{"auto", "cardinal", "digits"}[numberMode.getSelectedItemPosition()]; }
    private void updateTnPreview() {
        if (normalizer == null || tnPreview == null) return;
        try {
            normalizedPreview = normalizer.normalize(text.getText().toString(), tnEnabled.isChecked(), numberMode(), "Chinese").text;
            tnPreview.setText("实际合成文本（点击查看全文）：\n" + normalizedPreview);
        } catch (IllegalArgumentException e) { normalizedPreview = ""; tnPreview.setText(e.getMessage()); }
    }

    private void populateVoices() {
        String selected = selectedVoice() == null ? null : selectedVoice().id;
        populateVoices(selected);
    }
    private void populateVoices(String selected) {
        voiceEntries = new ArrayList<>(); List<String> names = new ArrayList<>();
        try {
            voiceEntries = VoiceStore.load(modelRoot, voiceRoot);
            int custom = 0;
            for (VoiceStore.Entry entry : voiceEntries) { names.add(entry.toString()); if (entry.custom()) custom++; }
            modelStatus.setText("模型已导入 · " + (names.size() - custom) + " 个内置音色 · " + custom + " 个自定义音色 · 完全离线");
        } catch (Exception e) {
            names.add("请先导入模型"); modelStatus.setText("首次使用，请导入下载的 .m5model 模型文件。");
        }
        voices.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, names));
        if (selected != null) for (int i = 0; i < voiceEntries.size(); i++) if (voiceEntries.get(i).id.equals(selected)) { voices.setSelection(i); break; }
        updateVoiceButtons();
    }
    private VoiceStore.Entry selectedVoice() {
        int index = voices == null ? -1 : voices.getSelectedItemPosition();
        return index >= 0 && index < voiceEntries.size() ? voiceEntries.get(index) : null;
    }
    private void updateVoiceButtons() {
        if (addVoiceButton == null || importVoiceButton == null || exportVoiceButton == null || deleteVoiceButton == null) return;
        boolean ready = !busy && new File(modelRoot, "bundle.json").isFile();
        VoiceStore.Entry selected = selectedVoice();
        addVoiceButton.setEnabled(ready); importVoiceButton.setEnabled(ready);
        exportVoiceButton.setEnabled(ready && selected != null);
        deleteVoiceButton.setEnabled(ready && selected != null && selected.custom());
    }
    private void pickVoice(int request) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, request);
    }
    private String voiceIdentity() throws Exception {
        return M5Engine.json(new File(modelRoot, "voices.json")).getString("encoder_sha256");
    }
    private void voiceDialog(Uri uri) {
        LinearLayout fields = new LinearLayout(this); fields.setOrientation(LinearLayout.VERTICAL); fields.setPadding(dp(20), dp(8), dp(20), dp(8));
        label(fields, "音色名称", 15, true);
        EditText name = new EditText(this); name.setSingleLine(true); name.setHint("例如：我的声音"); fields.addView(name);
        label(fields, "录音中说的原文", 15, true);
        EditText transcript = new EditText(this); transcript.setMinLines(3); transcript.setGravity(Gravity.TOP);
        transcript.setHint("准确填写录音原文，不是要合成的新文字");
        transcript.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE); fields.addView(transcript);
        label(fields, "选择清晰、无背景音乐的单人录音。创建完成后即可离线使用，原模型无需重新导入。", 13, false);
        AlertDialog dialog = new AlertDialog.Builder(this).setTitle("从录音新增音色").setView(fields)
            .setNegativeButton("取消", null).setPositiveButton("离线创建", null).create();
        dialog.setOnShowListener(v -> dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(button -> {
            final String label = name.getText().toString().trim(), words = transcript.getText().toString().trim();
            try { VoiceStore.validateLabel(label, words); }
            catch (IOException e) { Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show(); return; }
            dialog.dismiss(); createVoice(uri, label, words);
        }));
        dialog.show();
    }
    private void createVoice(Uri uri, String name, String transcript) {
        begin(); worker.execute(() -> {
            File temporary = new File(getCacheDir(), "m5-reference.audio");
            try {
                closeEngine(); System.gc(); updateStatus("正在读取参考录音…");
                String identity = voiceIdentity();
                if (!VoiceEncoder.SHA256.equals(identity)) throw new IOException("当前模型与内置音色编码器不兼容");
                try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(temporary)) {
                    if (in == null) throw new IOException("无法读取参考录音");
                    byte[] block = new byte[65536]; int count = 0, n;
                    while ((n = in.read(block)) != -1) {
                        if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                        count += n; if (count > ReferenceAudio.MAX_FILE_BYTES) throw new IOException("参考录音文件不能超过 20 MB");
                        out.write(block, 0, n);
                    }
                }
                updateStatus("正在解码并处理参考录音…");
                ReferenceAudio.Samples decoded = AndroidAudioReader.read(temporary, cancelled);
                float[] audio = ReferenceAudio.prepare(decoded, cancelled);
                synchronized (diagnostics) {
                    diagnostics.put("voice_source_rate", decoded.sampleRate);
                    diagnostics.put("voice_reference_s", audio.length / 24000.0);
                }
                long[][] codes = VoiceEncoder.encode(encoderFile(), audio, cancelled, this::updateStatus);
                byte[] document = VoiceStore.document(name, transcript, codes, identity).toString().getBytes(StandardCharsets.UTF_8);
                if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                File stored = VoiceStore.save(voiceRoot, document, identity);
                synchronized (diagnostics) { diagnostics.put("voice_frames", codes.length); diagnostics.put("voice_operation", "created_offline"); }
                ui(() -> { populateVoices(stored.getName()); status.setText("音色已新增并选中，可以生成语音。原模型无需重新导入。"); });
            } catch (Exception | OutOfMemoryError e) { failure(e); }
            finally { try { Files.deleteIfExists(temporary.toPath()); } catch (IOException ignored) { } System.gc(); finishWork(); }
        });
    }
    private File encoderFile() throws Exception {
        File directory = new File(getFilesDir(), "m5-voice-tools");
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("无法创建音色工具目录");
        File file = new File(directory, "encoder-" + VoiceEncoder.SHA256 + ".onnx");
        if (file.isFile() && file.length() == VoiceEncoder.MODEL_BYTES) return file;
        File temporary = new File(directory, "encoder.part");
        updateStatus("首次使用：正在准备离线音色工具…");
        MessageDigest digest = MessageDigest.getInstance("SHA-256"); long count = 0;
        try {
            try (InputStream in = getAssets().open("voice-tools/codec_encoder.onnx"); OutputStream out = new FileOutputStream(temporary)) {
                byte[] block = new byte[1048576]; int n;
                while ((n = in.read(block)) != -1) {
                    if (cancelled.get()) throw new InterruptedIOException("已取消新增音色");
                    out.write(block, 0, n); digest.update(block, 0, n); count += n;
                }
            }
            StringBuilder hash = new StringBuilder(); for (byte b : digest.digest()) hash.append(String.format(Locale.ROOT, "%02x", b & 255));
            if (count != VoiceEncoder.MODEL_BYTES || !VoiceEncoder.SHA256.equals(hash.toString())) throw new IOException("离线音色工具校验失败，请重新安装 APK");
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING);
            return file;
        } finally { Files.deleteIfExists(temporary.toPath()); }
    }
    private void importVoice(Uri uri) {
        begin(); worker.execute(() -> {
            try (InputStream in = getContentResolver().openInputStream(uri)) {
                updateStatus("正在导入音色包…"); byte[] bytes = VoiceStore.read(in);
                if (cancelled.get()) throw new InterruptedIOException("已取消导入音色");
                File file = VoiceStore.save(voiceRoot, bytes, voiceIdentity());
                ui(() -> { populateVoices(file.getName()); status.setText("音色包已导入并选中。"); });
            } catch (Exception e) { failure(e); } finally { finishWork(); }
        });
    }
    private void exportVoice() {
        VoiceStore.Entry entry = selectedVoice(); if (entry == null) return;
        try {
            pendingVoiceExport = VoiceStore.document(entry.voice.name, entry.voice.referenceText, entry.voice.codes, voiceIdentity()).toString().getBytes(StandardCharsets.UTF_8);
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/octet-stream")
                .addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE, "m5-voice-" + System.currentTimeMillis() + ".m5voice");
            startActivityForResult(intent, EXPORT_VOICE);
        } catch (Exception e) { failure(e); }
    }
    private void deleteVoice() {
        VoiceStore.Entry entry = selectedVoice(); if (entry == null || !entry.custom()) return;
        new AlertDialog.Builder(this).setTitle("删除音色").setMessage("删除“" + entry.voice.name + "”？")
            .setNegativeButton("取消", null).setPositiveButton("删除", (d, w) -> {
                begin(); worker.execute(() -> {
                    try { Files.delete(entry.file.toPath()); ui(() -> { populateVoices(); status.setText("音色已删除。"); }); }
                    catch (IOException e) { failure(e); } finally { finishWork(); }
                });
            }).show();
    }
    private void pickModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, IMPORT_MODEL);
    }
    @Override protected void onActivityResult(int request, int code, Intent intent) {
        super.onActivityResult(request, code, intent);
        if (code != RESULT_OK || intent == null || intent.getData() == null || busy) return;
        Uri uri = intent.getData();
        if (request == ADD_VOICE) { voiceDialog(uri); }
        else if (request == IMPORT_VOICE) { importVoice(uri); }
        else if (request == EXPORT_VOICE) {
            final byte[] bytes = pendingVoiceExport; pendingVoiceExport = null;
            if (bytes == null) { status.setText("请重新选择音色后导出。"); return; }
            begin(); worker.execute(() -> {
                try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                    if (out == null) throw new IOException("无法保存音色包"); out.write(bytes); updateStatus("音色包已导出。");
                } catch (Exception e) { failure(e); } finally { finishWork(); }
            });
        } else if (request == IMPORT_MODEL) {
            begin();
            worker.execute(() -> {
                File temporary = new File(getCacheDir(), "m5-import.zip");
                try {
                    closeEngine(); System.gc(); updateStatus("正在读取模型包…");
                    JSONObject source = describeModelSource(uri);
                    synchronized (diagnostics) {
                        diagnostics.remove("error"); diagnostics.remove("import_zip"); diagnostics.remove("import_sha256");
                        diagnostics.remove("import_format"); diagnostics.remove("import_zip_bytes");
                        diagnostics.put("import_source", source); diagnostics.put("import_copied_bytes", 0);
                    }
                    long copied = 0, lastProgress = 0;
                    MessageDigest digest = MessageDigest.getInstance("SHA-256");
                    InputStream selected = getContentResolver().openInputStream(uri);
                    if (selected == null) throw new IOException("无法读取选择的文件");
                    try (PushbackInputStream in = new PushbackInputStream(new BufferedInputStream(selected), ModelPackage.HEADER.length);
                            OutputStream out = new FileOutputStream(temporary)) {
                        copied = ModelPackage.unwrap(in);
                        if (copied > 0) digest.update(ModelPackage.HEADER);
                        synchronized (diagnostics) { diagnostics.put("import_format", copied > 0 ? "m5model-v1" : "zip"); }
                        byte[] b = new byte[1048576]; int n;
                        while ((n = in.read(b)) != -1) {
                            if (cancelled.get()) throw new InterruptedIOException("已取消导入");
                            copied += n;
                            if (copied > 1700L * 1048576) throw new IOException("模型包过大，请选择配套文件");
                            out.write(b, 0, n); digest.update(b, 0, n);
                            long now = System.nanoTime();
                            if (now - lastProgress > 250000000L) {
                                synchronized (diagnostics) { diagnostics.put("import_copied_bytes", copied); }
                                updateStatus("正在读取模型包… " + copied / 1048576 + " MiB"); lastProgress = now;
                            }
                        }
                    }
                    StringBuilder hash = new StringBuilder();
                    for (byte value : digest.digest()) hash.append(String.format(Locale.ROOT, "%02x", value & 255));
                    synchronized (diagnostics) {
                        diagnostics.put("import_copied_bytes", copied); diagnostics.put("import_sha256", hash.toString());
                        diagnostics.put("import_zip_bytes", temporary.length());
                    }
                    if (copied == 0) throw new IOException("选中的模型包为空，请下载完成后重试");
                    long reported = source.optLong("reported_bytes", -1);
                    if (reported > 0 && copied != reported)
                        throw new IOException("文件读取不完整（读取 " + copied + " / " + reported + " 字节），请下载完成后重新选择模型包");
                    updateStatus("模型包已读取 " + copied / 1048576 + " MiB，正在检查 ZIP…");
                    JSONObject archive = ModelInstaller.inspect(temporary);
                    synchronized (diagnostics) { diagnostics.put("import_zip", archive); }
                    ModelInstaller.install(temporary, getFilesDir(), cancelled, this::updateStatus);
                    ui(() -> { populateVoices(); status.setText("模型导入完成，可以开始生成。"); });
                } catch (Exception | OutOfMemoryError e) { failure(e); }
                finally {
                    try { Files.deleteIfExists(temporary.toPath()); } catch (IOException ignored) { }
                    finishWork();
                }
            });
        } else if (request == SAVE_AUDIO) {
            begin(); worker.execute(() -> {
                try (InputStream in = new FileInputStream(audioFile); OutputStream out = getContentResolver().openOutputStream(uri)) {
                    if (out == null) throw new IOException("无法保存到选择的位置");
                    byte[] bytes = new byte[65536]; int n; while ((n = in.read(bytes)) != -1) out.write(bytes, 0, n);
                    updateStatus("WAV 已保存。");
                } catch (Exception e) { failure(e); } finally { finishWork(); }
            });
        }
    }
    private JSONObject describeModelSource(Uri uri) {
        JSONObject info = new JSONObject();
        try (Cursor cursor = getContentResolver().query(uri,
                new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int name = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME), size = cursor.getColumnIndex(OpenableColumns.SIZE);
                if (name >= 0 && !cursor.isNull(name)) info.put("name", cursor.getString(name));
                if (size >= 0 && !cursor.isNull(size)) info.put("reported_bytes", cursor.getLong(size));
            }
        } catch (Exception ignored) { }
        return info;
    }
    private void generate() {
        final String value = text.getText().toString();
        try { M5Engine.splitText(value); } catch (IllegalArgumentException e) { status.setText(e.getMessage()); return; }
        if (normalizer == null) { status.setText("TN 规则未加载，请重新安装 APK。"); return; }
        final boolean useTn = tnEnabled.isChecked(); final String mode = numberMode();
        final long normalizeStart = System.nanoTime(); final TextNormalizer.Result prepared;
        try {
            prepared = normalizer.normalize(value, useTn, mode, "Chinese");
            if (prepared.text.codePointCount(0, prepared.text.length()) > 2000) throw new IllegalArgumentException("TN 展开后超过 2000 字，请缩短文本。");
        } catch (IllegalArgumentException e) { status.setText(e.getMessage()); return; }
        final double normalizeMs = (System.nanoTime() - normalizeStart) / 1e6;
        final VoiceStore.Entry voice = selectedVoice();
        if (voice == null) { status.setText("请先导入模型并选择音色。"); return; }
        final int threadCount = new int[]{2,4,8}[threads.getSelectedItemPosition()];
        releasePlayer(); begin(); result.setText("");
        worker.execute(() -> {
            File pending = new File(getFilesDir(), "m5-generating.wav");
            try {
                synchronized (diagnostics) { diagnostics.put("threads", threadCount); diagnostics.remove("error"); }
                synchronized (diagnostics) {
                    diagnostics.put("tn_enabled", useTn); diagnostics.put("tn_mode", mode); diagnostics.put("tn_changed", prepared.changed);
                    diagnostics.put("tn_version", prepared.version); diagnostics.put("tn_elapsed_ms", normalizeMs);
                }
                if (engine != null && loadedThreads != threadCount) { closeEngine(); System.gc(); }
                if (engine == null) { engine = new M5Engine(modelRoot, threadCount, this::updateStatus); loadedThreads = threadCount; }
                M5Engine.Synthesis synthesis = engine.synthesize(prepared.text, voice.voice, 1234, pending, cancelled, this::updateStatus);
                Files.move(pending.toPath(), audioFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                synchronized (diagnostics) {
                    diagnostics.put("elapsed_s", synthesis.elapsedSeconds); diagnostics.put("audio_s", synthesis.audioSeconds);
                    diagnostics.put("rtf", synthesis.elapsedSeconds / synthesis.audioSeconds); diagnostics.put("segments", synthesis.segments);
                }
                ui(() -> {
                    result.setText(String.format(Locale.CHINA, "语音 %.1f 秒 · 合成耗时 %.1f 秒", synthesis.audioSeconds, synthesis.elapsedSeconds));
                    status.setText("生成完成，可以播放或保存。");
                });
            } catch (Exception | OutOfMemoryError e) { failure(e); closeEngine(); System.gc(); }
            finally { try { Files.deleteIfExists(pending.toPath()); } catch (IOException ignored) { } finishWork(); }
        });
    }
    private void play() {
        releasePlayer();
        try {
            player = new MediaPlayer(); player.setDataSource(audioFile.getPath()); player.prepare(); player.start();
            player.setOnCompletionListener(p -> { releasePlayer(); status.setText("播放结束。"); });
            status.setText("正在播放…");
        } catch (IOException e) { failure(e); }
    }
    private void releasePlayer() { if (player != null) { player.release(); player = null; } }
    private void closeEngine() { if (engine != null) { engine.close(); engine = null; } }
    private void begin() {
        releasePlayer(); cancelled.set(false);
        synchronized (diagnostics) { diagnostics.remove("error"); }
        setBusy(true);
    }
    private void finishWork() { ui(() -> setBusy(false)); }
    private void setBusy(boolean value) {
        busy = value; importButton.setEnabled(!value); text.setEnabled(!value); voices.setEnabled(!value); threads.setEnabled(!value);
        tnEnabled.setEnabled(!value); numberMode.setEnabled(!value);
        updateVoiceButtons();
        generateButton.setEnabled(!value && new File(modelRoot, "bundle.json").isFile());
        playButton.setEnabled(!value && audioFile.isFile()); saveButton.setEnabled(!value && audioFile.isFile());
        cancelButton.setVisibility(value ? View.VISIBLE : View.GONE); progress.setVisibility(value ? View.VISIBLE : View.GONE);
        if (value) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
    private void updateStatus(String message) {
        long now = System.nanoTime();
        if (now - lastPssSample > 1000000000L) { peakPssKb = Math.max(peakPssKb, Debug.getPss()); lastPssSample = now; }
        synchronized (diagnostics) { try { diagnostics.put("stage", message); diagnostics.put("sampled_peak_pss_mib", peakPssKb / 1024.0); } catch (JSONException ignored) { } }
        ui(() -> status.setText(message));
    }
    private void failure(Throwable e) {
        StringWriter trace = new StringWriter(); e.printStackTrace(new PrintWriter(trace));
        synchronized (diagnostics) { try { diagnostics.put("error", trace.toString()); } catch (JSONException ignored) { } }
        android.util.Log.e("M5Offline", "Operation failed", e);
        String message = e instanceof OutOfMemoryError ? "内存不足，请关闭其他应用后重试。" : e.getMessage();
        if (message == null || message.isEmpty()) message = e.getClass().getSimpleName();
        final String shown = message;
        ui(() -> status.setText(shown + (e instanceof InterruptedIOException ? "" : "\n可点击“复制诊断信息”反馈。")));
    }
    private String diagnosticText() { synchronized (diagnostics) { try { return diagnostics.toString(2); } catch (JSONException e) { return diagnostics.toString(); } } }
    private void ui(Runnable runnable) { runOnUiThread(() -> { if (!destroyed) runnable.run(); }); }
    private int dp(int value) { return (int) (value * getResources().getDisplayMetrics().density + 0.5f); }
    private void gap(LinearLayout parent, int height) { View view = new View(this); parent.addView(view, new LinearLayout.LayoutParams(1, dp(height))); }
    private TextView label(LinearLayout parent, String value, int size, boolean bold) {
        TextView label = new TextView(this); label.setText(value); label.setTextSize(size); label.setTextColor(Color.rgb(30,42,60));
        if (bold) label.setTypeface(Typeface.DEFAULT, Typeface.BOLD); label.setPadding(0, dp(6), 0, dp(6));
        parent.addView(label, new LinearLayout.LayoutParams(-1, -2)); return label;
    }
    private Button button(LinearLayout parent, String value, View.OnClickListener listener) {
        Button button = new Button(this); button.setText(value); button.setAllCaps(false); button.setOnClickListener(listener);
        parent.addView(button, new LinearLayout.LayoutParams(parent.getOrientation() == LinearLayout.HORIZONTAL ? 0 : -1, -2,
            parent.getOrientation() == LinearLayout.HORIZONTAL ? 1 : 0)); return button;
    }
    @Override protected void onSaveInstanceState(Bundle state) {
        state.putString("text", text.getText().toString()); state.putBoolean("tn", tnEnabled.isChecked());
        state.putInt("number_mode", numberMode.getSelectedItemPosition()); super.onSaveInstanceState(state);
    }
    @Override protected void onDestroy() {
        destroyed = true; cancelled.set(true); releasePlayer(); worker.execute(this::closeEngine); worker.shutdown(); super.onDestroy();
    }
}
