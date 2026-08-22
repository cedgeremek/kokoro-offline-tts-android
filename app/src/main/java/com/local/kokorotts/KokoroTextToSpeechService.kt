package com.local.kokorotts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.util.Locale
import java.util.concurrent.CancellationException

/** Android's system-engine adapter for fully offline Kokoro synthesis. */
class KokoroTextToSpeechService : TextToSpeechService() {
    companion object {
        private const val TAG = "KokoroTts"
        private const val CHANNEL_COUNT_MONO = 1
        private const val DEFAULT_CHUNK_BYTES = 8_192
        private const val INITIAL_PREPARE_WAIT_MILLIS = 5_000L
    }

    @Volatile private var stopped = false
    private lateinit var synthesizer: KokoroSynthesizer
    private var prepareThread: Thread? = null

    override fun onCreate() {
        super.onCreate()
        synthesizer = KokoroSynthesizer(applicationContext)
        prepareThread = Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
            try {
                synthesizer.prepare()
            } catch (problem: Exception) {
                Log.w(TAG, "Background model preparation did not complete", problem)
            }
        }, "kokoro-prepare").apply { start() }
    }

    override fun onDestroy() {
        stopped = true
        synthesizer.cancel()
        prepareThread?.interrupt()
        prepareThread = null
        synthesizer.close()
        super.onDestroy()
    }

    override fun onGetLanguage(): Array<String> = arrayOf("eng", "USA", "")

    override fun onIsLanguageAvailable(lang: String, country: String?, variant: String?): Int {
        if (!lang.equals("eng", true) && !lang.equals("en", true)) return TextToSpeech.LANG_NOT_SUPPORTED
        return when {
            country.isNullOrBlank() -> TextToSpeech.LANG_AVAILABLE
            country.equals("USA", true) || country.equals("US", true) ||
                country.equals("GBR", true) || country.equals("GB", true) -> TextToSpeech.LANG_COUNTRY_AVAILABLE
            else -> TextToSpeech.LANG_AVAILABLE
        }
    }

    override fun onLoadLanguage(lang: String, country: String?, variant: String?): Int =
        onIsLanguageAvailable(lang, country, variant)

    override fun onGetVoices() = VoiceCatalog.voices()

    override fun onGetDefaultVoiceNameFor(lang: String, country: String?, variant: String?): String? {
        if (onIsLanguageAvailable(lang, country, variant) == TextToSpeech.LANG_NOT_SUPPORTED) return null
        return VoiceCatalog.preferredDefaultId(applicationContext, country)
    }

    override fun onIsValidVoiceName(voiceName: String): Int =
        if (VoiceCatalog.voices().any { it.name == voiceName }) TextToSpeech.SUCCESS else TextToSpeech.ERROR

    override fun onLoadVoice(voiceName: String): Int = onIsValidVoiceName(voiceName)

    override fun onStop() {
        stopped = true
        synthesizer.cancel()
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val text = request.charSequenceText?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            terminateWithError(callback, TextToSpeech.ERROR_INVALID_REQUEST)
            return
        }

        stopped = false
        val tid = Process.myTid()
        val previousPriority = runCatching { Process.getThreadPriority(tid) }.getOrDefault(Process.THREAD_PRIORITY_DEFAULT)
        var callbackStarted = false
        var audioDeliveryFailed = false
        var callbackFailed = false
        try {
            // A short race-to-idle burst lowers both perceived latency and total
            // awake time; ONNX workers remain bounded to four cores.
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            val voice = VoiceCatalog.find(request.voiceName)
            val deliveryMultiplier = ExpressionSettings.deliverySpeed(applicationContext)
            val speed = KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(
                KokoroSynthesizer.modelSpeedForRequest(request.speechRate),
                deliveryMultiplier,
            )
            Log.i(
                TAG,
                "stage=request_config voice=${voice.id} request_rate_percent=${request.speechRate} " +
                    "delivery_multiplier=$deliveryMultiplier model_speed=$speed",
            )
            callbackStarted = true
            if (callback.start(
                    KokoroSynthesizer.SAMPLE_RATE,
                    AudioFormat.ENCODING_PCM_16BIT,
                    CHANNEL_COUNT_MONO,
                ) != TextToSpeech.SUCCESS
            ) {
                callbackFailed = true
                return
            }

            // The normal streaming route assumes B128/B192/B208 are resident.  Starting the
            // first utterance while the background thread is still loading those contexts can
            // exhaust the opening PCM before its continuation is ready, producing a loud media
            // restart followed by an apparent volume collapse.  Wait only on a genuinely cold
            // engine; every later request observes an already-finished thread and returns here
            // immediately.  Keep the wait bounded so a provider fault cannot wedge Android TTS.
            awaitInitialPreparation()
            if (stopped) throw CancellationException("Synthesis stopped during initial preparation")

            var chunkIndex = 0
            synthesizer.synthesizeChunks(text, voice, speed) { pcm ->
                val currentChunk = chunkIndex++
                val accepted = !stopped && writeAudio(callback, pcm, currentChunk)
                if (!accepted && !stopped && !callback.hasFinished()) audioDeliveryFailed = true
                accepted
            }
            if (stopped) throw CancellationException("Synthesis stopped")
            check(!audioDeliveryFailed) { "Android rejected synthesized audio" }
            if (!callback.hasFinished()) {
                check(callback.done() == TextToSpeech.SUCCESS) { "Android rejected synthesis completion" }
            }
        } catch (_: CancellationException) {
            callbackFailed = true
        } catch (t: Throwable) {
            callbackFailed = true
            Log.e(TAG, "Offline synthesis failed", t)
            if (t is Error) throw t
        } finally {
            if (callbackFailed) {
                terminateWithError(callback, TextToSpeech.ERROR_SYNTHESIS)
            } else if (callbackStarted && !callback.hasFinished()) {
                runCatching { callback.done() }
                    .onFailure { Log.w(TAG, "Unable to finish synthesis callback", it) }
            }
            runCatching { Process.setThreadPriority(previousPriority) }
        }
    }

    private fun awaitInitialPreparation() {
        val thread = prepareThread ?: return
        if (!thread.isAlive || thread === Thread.currentThread()) return
        val started = SystemClock.elapsedRealtime()
        val deadline = started + INITIAL_PREPARE_WAIT_MILLIS
        try {
            while (thread.isAlive && !stopped) {
                val remaining = deadline - SystemClock.elapsedRealtime()
                if (remaining <= 0L) break
                thread.join(minOf(100L, remaining))
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            throw CancellationException("Interrupted while waiting for initial model preparation")
        }
        Log.i(
            TAG,
            "stage=initial_prepare_wait elapsed_ms=${SystemClock.elapsedRealtime() - started} " +
                "complete=${!thread.isAlive}",
        )
    }

    private fun terminateWithError(callback: SynthesisCallback, errorCode: Int) {
        if (callback.hasFinished()) return
        runCatching { callback.error(errorCode) }
            .onFailure { Log.w(TAG, "Unable to report synthesis callback error", it) }
        if (!callback.hasFinished()) {
            runCatching { callback.done() }
                .onFailure { Log.w(TAG, "Unable to finish failed synthesis callback", it) }
        }
    }

    private fun writeAudio(callback: SynthesisCallback, pcm: ByteArray, chunkIndex: Int): Boolean {
        val bufferSize = callback.maxBufferSize.takeIf { it > 0 } ?: DEFAULT_CHUNK_BYTES
        val started = SystemClock.elapsedRealtimeNanos()
        var callbackCount = 0
        var maxCallbackNanos = 0L
        var offset = 0
        while (offset < pcm.size && !stopped) {
            val count = minOf(bufferSize, pcm.size - offset)
            val callbackStarted = SystemClock.elapsedRealtimeNanos()
            val status = callback.audioAvailable(pcm, offset, count)
            maxCallbackNanos = maxOf(maxCallbackNanos, SystemClock.elapsedRealtimeNanos() - callbackStarted)
            callbackCount++
            if (status != TextToSpeech.SUCCESS) return false
            offset += count
        }
        Log.i(
            TAG,
            "stage=callback_chunk index=$chunkIndex bytes=$offset buffers=$callbackCount " +
                "max_block_ms=${(maxCallbackNanos / 100_000L) / 10.0} " +
                "elapsed_ms=${((SystemClock.elapsedRealtimeNanos() - started) / 100_000L) / 10.0}",
        )
        return !stopped
    }
}
