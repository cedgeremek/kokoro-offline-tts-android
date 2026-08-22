package com.local.kokorotts

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import kotlin.math.roundToInt

/** Engine settings, voice selection, and an end-to-end Android TTS self-test. */
@SuppressLint("SetTextI18n")
class MainActivity : Activity() {
    companion object {
        private const val TTS_SETTINGS_ACTION = "com.android.settings.TTS_SETTINGS"
        private const val SAMPLE_TEXT = "This is an example of speech synthesis in English."
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val catalog by lazy { VoiceCatalog.all() }

    private lateinit var voiceSpinner: Spinner
    private lateinit var deliverySpeedSlider: SeekBar
    private lateinit var deliverySpeedView: TextView
    private lateinit var testButton: Button
    private lateinit var statusView: TextView

    private var textToSpeech: TextToSpeech? = null
    private var engineReady = false
    private var engineGeneration = 0
    private var selectedVoiceId = ""
    private var currentUtteranceId: String? = null

    @Volatile private var engineState = "Binding to the Kokoro Android TTS service..."
    @Volatile private var testState = "Self-test has not been run."
    @Volatile private var engineInitMs = -1L
    @Volatile private var testStartedMs = -1L
    @Volatile private var synthesisBeganMs = -1L
    @Volatile private var firstAudioMs = -1L
    @Volatile private var playbackStartedMs = -1L
    @Volatile private var testFinishedMs = -1L
    @Volatile private var receivedAudioBytes = 0L

    private val openedAtMs = SystemClock.elapsedRealtime()

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        selectedVoiceId = VoiceCatalog.preferredDefaultId(applicationContext)
        setContentView(buildContentView())
        refreshStatus()
        initializeEngine()
    }

    override fun onDestroy() {
        engineGeneration++
        currentUtteranceId = null
        engineReady = false
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        super.onDestroy()
    }

    private fun buildContentView(): ScrollView {
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(24), dp(24), dp(32))
        }

        content.addView(TextView(this).apply {
            text = "Kokoro Offline TTS"
            textSize = 26f
            setTypeface(typeface, Typeface.BOLD)
        }, matchWrap())

        content.addView(TextView(this).apply {
            text = "Fully offline Android speech with all 28 English Kokoro voices. " +
                "Choose a default voice, then run the same system TTS path used by Samsung Settings and TTSReader."
            textSize = 16f
            setPadding(0, dp(10), 0, dp(20))
        }, matchWrap())

        content.addView(TextView(this).apply {
            text = "Default voice"
            textSize = 16f
            setTypeface(typeface, Typeface.BOLD)
        }, matchWrap())

        voiceSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                catalog.map(VoiceCatalog::displayName),
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(catalog.indexOfFirst { it.id == selectedVoiceId }.coerceAtLeast(0), false)
        }
        content.addView(voiceSpinner, matchWrap())

        content.addView(TextView(this).apply {
            text = "Experimental delivery speed"
            textSize = 16f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(20), 0, dp(4))
        }, matchWrap())

        deliverySpeedView = TextView(this).apply { textSize = 14f }
        content.addView(deliverySpeedView, matchWrap())

        deliverySpeedSlider = SeekBar(this).apply {
            max = ((ExpressionSettings.MAX_DELIVERY_SPEED - ExpressionSettings.MIN_DELIVERY_SPEED) * 100).toInt()
            progress = ((ExpressionSettings.deliverySpeed(applicationContext) -
                ExpressionSettings.MIN_DELIVERY_SPEED) * 100).roundToInt()
            contentDescription = "Experimental delivery speed"
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                    val multiplier = ExpressionSettings.MIN_DELIVERY_SPEED + progress / 100f
                    ExpressionSettings.saveDeliverySpeed(applicationContext, multiplier)
                    updateDeliverySpeedLabel()
                }
                override fun onStartTrackingTouch(seekBar: SeekBar) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar) = Unit
            })
        }
        content.addView(deliverySpeedSlider, matchWrap())

        content.addView(TextView(this).apply {
            text = "This changes only Kokoro's existing sentence speed input (85–115%) and is saved. " +
                "Voice choice remains the available style control. Current Kokoro v1 has no safe independent " +
                "intonation or pitch input, so this app does not fake one with temperature, noise, pitch shifting, or pauses."
            textSize = 14f
        }, matchWrap())

        testButton = Button(this).apply {
            text = "Test selected voice"
            isEnabled = false
            setOnClickListener { runSelfTest() }
        }
        content.addView(testButton, matchWrap(topMargin = 12))

        content.addView(Button(this).apply {
            text = "Open Android TTS settings"
            setOnClickListener { openTtsSettings() }
        }, matchWrap(topMargin = 8))

        content.addView(Button(this).apply {
            text = "Retry accelerator"
            setOnClickListener { retryAccelerator() }
        }, matchWrap(topMargin = 8))

        content.addView(Button(this).apply {
            text = "Copy diagnostics"
            setOnClickListener { copyDiagnostics() }
        }, matchWrap(topMargin = 8))

        content.addView(TextView(this).apply {
            text = "Status"
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(0, dp(24), 0, dp(8))
        }, matchWrap())

        statusView = TextView(this).apply {
            textSize = 15f
            setTextIsSelectable(true)
        }
        content.addView(statusView, matchWrap())

        content.addView(TextView(this).apply {
            text = "No model download, account, network connection, or additional phone setup is required."
            textSize = 14f
            setPadding(0, dp(20), 0, 0)
        }, matchWrap())

        voiceSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long,
            ) {
                val selected = catalog.getOrNull(position) ?: return
                selectedVoiceId = selected.id
                VoiceCatalog.savePreferredDefault(applicationContext, selected.id)
                if (engineReady) {
                    applySelectedVoice()
                    engineState = "Ready. ${VoiceCatalog.displayName(selected)} is the persisted default voice."
                }
                refreshStatus()
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }

        updateDeliverySpeedLabel()

        return ScrollView(this).apply { addView(content) }
    }

    private fun initializeEngine() {
        val generation = ++engineGeneration
        val requestedAt = SystemClock.elapsedRealtime()
        textToSpeech = TextToSpeech(applicationContext, { status ->
            // TextToSpeech may invoke its listener during construction on failure.
            mainHandler.post {
                if (isFinishing || isDestroyed || generation != engineGeneration) return@post
                engineInitMs = SystemClock.elapsedRealtime() - requestedAt
                if (status == TextToSpeech.SUCCESS && textToSpeech != null) {
                    engineReady = true
                    installProgressListener(textToSpeech!!)
                    val voiceApplied = applySelectedVoice()
                    engineState = if (voiceApplied) {
                        "Ready. Bound to this APK as a native Android TTS engine."
                    } else {
                        "Engine initialized, but Android did not expose the selected Kokoro voice."
                    }
                    testButton.isEnabled = voiceApplied
                } else {
                    engineReady = false
                    engineState = "Engine initialization failed (Android status $status)."
                    testButton.isEnabled = false
                }
                refreshStatus()
            }
        }, packageName)
    }

    private fun retryAccelerator() {
        if (!BuildConfig.KOKORO_QNN_AOT_INCLUDED) {
            testState = "This build does not contain the packaged accelerator contexts."
            refreshStatus()
            return
        }
        val clearedFailure = KokoroSynthesizer.requestQnnRetry(applicationContext)
        currentUtteranceId = null
        engineReady = false
        testButton.isEnabled = false
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        engineState = "Rebinding the Kokoro service for a fresh accelerator session..."
        testState = if (clearedFailure) {
            "Accelerator fallback was cleared. Wait for Ready, then rerun the selected voice test."
        } else {
            "No accelerator failure was stored. Rebinding anyway for a fresh uncached test."
        }
        refreshStatus()
        initializeEngine()
    }

    private fun applySelectedVoice(): Boolean {
        val engine = textToSpeech ?: return false
        val voice = engine.voices?.firstOrNull { it.name == selectedVoiceId } ?: return false
        return engine.setVoice(voice) == TextToSpeech.SUCCESS
    }

    private fun updateDeliverySpeedLabel() {
        if (!::deliverySpeedView.isInitialized) return
        val percent = (ExpressionSettings.deliverySpeed(applicationContext) * 100).roundToInt()
        deliverySpeedView.text = "$percent% of requested delivery speed"
    }

    private fun installProgressListener(engine: TextToSpeech) {
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onBeginSynthesis(
                utteranceId: String?,
                sampleRateInHz: Int,
                audioFormat: Int,
                channelCount: Int,
            ) {
                if (!isCurrentTest(utteranceId)) return
                synthesisBeganMs = SystemClock.elapsedRealtime()
                testState = "Synthesis began at ${sampleRateInHz} Hz, $channelCount channel; waiting for first audio."
                refreshStatus()
            }

            override fun onAudioAvailable(utteranceId: String?, audio: ByteArray?) {
                if (!isCurrentTest(utteranceId) || audio == null) return
                receivedAudioBytes += audio.size
                if (firstAudioMs < 0L) {
                    firstAudioMs = SystemClock.elapsedRealtime()
                    testState = "First audio buffer reached Android; playback is in progress."
                    refreshStatus()
                }
            }

            override fun onStart(utteranceId: String?) {
                if (!isCurrentTest(utteranceId)) return
                playbackStartedMs = SystemClock.elapsedRealtime()
                testState = "Android started playback."
                refreshStatus()
            }

            override fun onDone(utteranceId: String?) {
                if (!isCurrentTest(utteranceId)) return
                testFinishedMs = SystemClock.elapsedRealtime()
                testState = if (receivedAudioBytes > 0L) {
                    "PASS: Android completed playback and received $receivedAudioBytes PCM bytes."
                } else {
                    "FAIL: Android reported completion without delivering an audio buffer."
                }
                finishTest()
            }

            @Deprecated("Legacy Android callback")
            override fun onError(utteranceId: String?) {
                reportTestError(utteranceId, TextToSpeech.ERROR)
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                reportTestError(utteranceId, errorCode)
            }

            override fun onStop(utteranceId: String?, interrupted: Boolean) {
                if (!isCurrentTest(utteranceId)) return
                testFinishedMs = SystemClock.elapsedRealtime()
                testState = "STOPPED: Android ended the self-test (interrupted=$interrupted)."
                finishTest()
            }
        })
    }

    private fun runSelfTest() {
        val engine = textToSpeech
        if (!engineReady || engine == null) {
            testState = "The engine is not ready. Close and reopen this screen, then try again."
            refreshStatus()
            return
        }
        if (!applySelectedVoice()) {
            testState = "Android rejected the selected voice: $selectedVoiceId."
            refreshStatus()
            return
        }

        engine.stop()
        val now = SystemClock.elapsedRealtime()
        val utteranceId = "kokoro-self-test-$now"
        currentUtteranceId = utteranceId
        testStartedMs = now
        synthesisBeganMs = -1L
        firstAudioMs = -1L
        playbackStartedMs = -1L
        testFinishedMs = -1L
        receivedAudioBytes = 0L
        testState = "Requesting the Settings sample through Android TextToSpeech..."
        testButton.isEnabled = false
        refreshStatus()

        val accepted = engine.speak(SAMPLE_TEXT, TextToSpeech.QUEUE_FLUSH, Bundle(), utteranceId)
        if (accepted != TextToSpeech.SUCCESS) {
            testFinishedMs = SystemClock.elapsedRealtime()
            testState = "FAIL: Android rejected the speech request (status $accepted)."
            finishTest()
        }
    }

    private fun reportTestError(utteranceId: String?, errorCode: Int) {
        if (!isCurrentTest(utteranceId)) return
        testFinishedMs = SystemClock.elapsedRealtime()
        testState = "FAIL: Android reported TTS error $errorCode."
        finishTest()
    }

    private fun isCurrentTest(utteranceId: String?): Boolean =
        utteranceId != null && utteranceId == currentUtteranceId

    private fun finishTest() {
        currentUtteranceId = null
        mainHandler.post {
            if (!isFinishing && !isDestroyed) {
                testButton.isEnabled = engineReady
                refreshStatus()
            }
        }
    }

    private fun refreshStatus() {
        mainHandler.post {
            if (isFinishing || isDestroyed || !::statusView.isInitialized) return@post
            statusView.text = buildStatusText()
        }
    }

    private fun buildStatusText(): String = buildString {
        val runtime = KokoroSynthesizer.runtimeDiagnostics(applicationContext)
        append(engineState)
        append("\n\n")
        append(testState)
        append("\n\nSelected voice: ").append(selectedVoiceId)
        if (engineInitMs >= 0L) append("\nEngine bind: ").append(engineInitMs).append(" ms")
        appendElapsed("Request to synthesis", testStartedMs, synthesisBeganMs)
        appendElapsed("Request to first audio", testStartedMs, firstAudioMs)
        appendElapsed("Request to playback", testStartedMs, playbackStartedMs)
        appendElapsed("Total self-test", testStartedMs, testFinishedMs)
        append("\n\nRuntime: ").append(runtime.backend.ifBlank { "no generator run recorded" })
        runtime.bucket?.let { append(", bucket T=").append(it) }
        if (runtime.generatorRtf.isNotBlank()) append(", RTF=").append(runtime.generatorRtf)
        if (runtime.contextSource.isNotBlank()) append("\nContext: ").append(runtime.contextSource)
        if (runtime.contextHashPrefix.isNotBlank()) append(" @ ").append(runtime.contextHashPrefix)
        append("\nPackaged QNN AOT: ").append(runtime.qnnAotIncluded)
        append("; QNN disabled: ").append(runtime.qnnDisabled)
        if (runtime.failureReason.isNotBlank()) append("\nLast fallback: ").append(runtime.failureReason)
    }

    private fun StringBuilder.appendElapsed(label: String, start: Long, end: Long) {
        if (start >= 0L && end >= start) append('\n').append(label).append(": ").append(end - start).append(" ms")
    }

    private fun copyDiagnostics() {
        val runtime = KokoroSynthesizer.runtimeDiagnostics(applicationContext)
        val diagnostic = buildString {
            append("Kokoro Offline TTS diagnostic\n")
            append("version=").append(BuildConfig.VERSION_NAME).append(" (").append(BuildConfig.VERSION_CODE).append(")\n")
            append("package=").append(packageName).append('\n')
            append("device=").append(Build.MANUFACTURER).append(' ').append(Build.MODEL).append('\n')
            append("sdk=").append(Build.VERSION.SDK_INT).append('\n')
            append("soc=").append(if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unavailable").append('\n')
            append("abis=").append(Build.SUPPORTED_ABIS.joinToString()).append('\n')
            append("engineReady=").append(engineReady).append('\n')
            append("androidDefaultEngine=").append(textToSpeech?.defaultEngine ?: "unavailable").append('\n')
            append("selectedVoice=").append(selectedVoiceId).append('\n')
            append("expression.deliverySpeedMultiplier=")
                .append(ExpressionSettings.deliverySpeed(applicationContext)).append('\n')
            append("catalogVoices=").append(catalog.size).append('\n')
            append("visibleKokoroVoices=").append(
                textToSpeech?.voices?.count { VoiceCatalog.isValidId(it.name) } ?: 0,
            ).append('\n')
            append("screenOpenMs=").append(SystemClock.elapsedRealtime() - openedAtMs).append('\n')
            append("engineState=").append(engineState).append('\n')
            append("testState=").append(testState).append('\n')
            append("audioBytes=").append(receivedAudioBytes).append('\n')
            append("engine.rateMultiplier=").append(KokoroSynthesizer.ENGINE_RATE_MULTIPLIER).append('\n')
            append("runtime.backend=").append(runtime.backend).append('\n')
            append("runtime.bucket=").append(runtime.bucket ?: "unavailable").append('\n')
            append("runtime.generatorRtf=").append(runtime.generatorRtf).append('\n')
            append("runtime.contextSource=").append(runtime.contextSource).append('\n')
            append("runtime.contextHashPrefix=").append(runtime.contextHashPrefix).append('\n')
            append("runtime.failureReason=").append(runtime.failureReason).append('\n')
            append("runtime.timestampUtcMillis=").append(runtime.timestampUtcMillis).append('\n')
            append("runtime.qnnDisabled=").append(runtime.qnnDisabled).append('\n')
            append("runtime.nnapiDisabled=").append(runtime.nnapiDisabled).append('\n')
            append("qnn.aotIncluded=").append(runtime.qnnAotIncluded).append('\n')
            append("qnn.sharedIncluded=").append(BuildConfig.KOKORO_QNN_SHARED_INCLUDED).append('\n')
            append("model.frontSha256=").append(BuildConfig.KOKORO_FRONT_MODEL_SHA256).append('\n')
            append("model.generatorSha256=").append(BuildConfig.KOKORO_GENERATOR_MODEL_SHA256).append('\n')
            append("qnn.contextProducer=").append(BuildConfig.KOKORO_QNN_CONTEXT_PRODUCER).append('\n')
            append("qnn.performance=").append(KokoroSynthesizer.qnnPerformancePolicyForTesting()).append('\n')
            append("qnn.precision=").append(KokoroSynthesizer.qnnPrecisionPolicyForTesting()).append('\n')
            append("qnn.b192Asset=").append(BuildConfig.KOKORO_QNN_B192_CONTEXT_ASSET).append('\n')
            append("qnn.b192Sha256=").append(BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256).append('\n')
            append("qnn.b192HashPrefix=").append(runtime.b192ContextHashPrefix).append('\n')
            append("qnn.b256Asset=").append(BuildConfig.KOKORO_QNN_B256_CONTEXT_ASSET).append('\n')
            append("qnn.b256Sha256=").append(BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256).append('\n')
            append("qnn.b256HashPrefix=").append(runtime.b256ContextHashPrefix).append('\n')
            append("qnn.b320Asset=").append(BuildConfig.KOKORO_QNN_B320_CONTEXT_ASSET).append('\n')
            append("qnn.b320Sha256=").append(BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256).append('\n')
            append("qnn.b320HashPrefix=").append(runtime.b320ContextHashPrefix).append('\n')
            append("qnn.b384Asset=").append(BuildConfig.KOKORO_QNN_B384_CONTEXT_ASSET).append('\n')
            append("qnn.b384Sha256=").append(BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256).append('\n')
            append("qnn.b384HashPrefix=").append(runtime.b384ContextHashPrefix).append('\n')
            append("qnn.retryGeneration=").append(runtime.qnnRetryGeneration).append('\n')
            append("timing.engineBindMs=").append(engineInitMs).append('\n')
            append("timing.requestToSynthesisMs=").append(deltaOrMissing(testStartedMs, synthesisBeganMs)).append('\n')
            append("timing.requestToFirstAudioMs=").append(deltaOrMissing(testStartedMs, firstAudioMs)).append('\n')
            append("timing.requestToPlaybackMs=").append(deltaOrMissing(testStartedMs, playbackStartedMs)).append('\n')
            append("timing.totalMs=").append(deltaOrMissing(testStartedMs, testFinishedMs))
        }
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("Kokoro TTS diagnostic", diagnostic))
        Toast.makeText(this, "Diagnostics copied", Toast.LENGTH_SHORT).show()
    }

    private fun deltaOrMissing(start: Long, end: Long): String =
        if (start >= 0L && end >= start) (end - start).toString() else "unavailable"

    private fun openTtsSettings() {
        try {
            startActivity(Intent(TTS_SETTINGS_ACTION).addCategory(Intent.CATEGORY_DEFAULT))
        } catch (_: ActivityNotFoundException) {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()

    private fun matchWrap(topMargin: Int = 0): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            this.topMargin = dp(topMargin)
        }
}
