package com.local.kokorotts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import ai.onnxruntime.qnnpluginep.getEpName
import ai.onnxruntime.qnnpluginep.getLibraryPath
import android.content.Context
import android.os.Build
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.system.Os
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.EnumSet
import java.util.LinkedHashMap
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Callable
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.math.tanh

internal enum class BackendPreference { AUTO, CPU, QNN_HTP, NNAPI }
private enum class InferenceBackend { CPU, QNN_HTP, NNAPI }
internal enum class WaveformSeam {
    NONE,
    REQUEST_BOUNDARY,
    CONTINUATION,
    COMMA,
    SEMICOLON,
    COLON,
    PERIOD,
    QUESTION,
}

internal data class RuntimeDiagnosticsSnapshot(
    val backend: String,
    val bucket: Int?,
    val generatorRtf: String,
    val contextSource: String,
    val contextHashPrefix: String,
    val failureReason: String,
    val timestampUtcMillis: Long,
    val qnnDisabled: Boolean,
    val nnapiDisabled: Boolean,
    val qnnAotIncluded: Boolean,
    val b192ContextHashPrefix: String,
    val b256ContextHashPrefix: String,
    val b320ContextHashPrefix: String,
    val b384ContextHashPrefix: String,
    val qnnRetryGeneration: Long,
)

internal data class QnnCandidateCapture(
    val frames: Int,
    val samples: FloatArray,
    val elapsedMs: Long,
)

/** Debug-only evidence showing exactly what request-edge trimming removed. */
internal data class OpeningEdgeCapture(
    val frames: Int,
    val rawSamples: FloatArray,
    val trimmedSamples: FloatArray,
    val detectedActiveStartSample: Int?,
    val removedLeadingSamples: Int,
)

internal data class PcmEncodingResult(
    val pcm: ByteArray,
    val gain: Float,
    val robustPeak: Float,
    val absolutePeak: Float,
    val limitedSamples: Int,
)

/** Test/qualification evidence from the exact Android-boundary chunk sequence. */
internal data class ChunkOutputDiagnostics(
    val index: Int,
    val preStabilizerPcm: ByteArray,
    val postStabilizerPcm: ByteArray,
    val deliveredPcm: ByteArray,
    val coreSamples: Int,
    val leadingOverlapSamples: Int,
    val trailingOverlapSamples: Int,
    val activeRms: Double?,
    val referenceRms: Double?,
    val requestedGain: Float,
    val startGain: Float,
    val appliedGain: Float,
    val rampSamples: Int,
    val peak: Int,
)

/** Plan-only evidence for instrumentation; no generator or Android audio sink is opened. */
internal data class GlobalPlanInspection(
    val phonemeChars: Int,
    val initialChunks: Int,
    val finalChunks: Int,
    val globallyConditioned: Boolean,
    val coreFrames: List<Int>,
    val windowFrames: List<Int>,
    val qnnBuckets: List<Int?>,
    val leadingOverlapFrames: List<Int>,
    val trailingOverlapFrames: List<Int>,
)

internal data class GlobalDurationRefinement(
    val chunks: List<String>,
    val boundaries: List<Pair<Int, Int>>,
    val refinements: Int,
    val bridgeCoalescences: Int,
)

/**
 * An utterance-local safety controller for window-to-window generator loudness drift. It runs
 * after the PCM cache and measures the sequence that Android will actually receive.
 *
 * Raw whole-chunk RMS was a poor proxy for audible speech: a boundary click, rumble, or a few
 * loud low-frequency periods could hide a sustained quiet continuation. This meter instead uses
 * the median active 50 ms speech-band window. It preserves a +/-1 dB expressive corridor, then
 * corrects only the excess by at most 6 dB. Gain carries across boundaries and moves over 100 ms;
 * isolated peaks use the same continuous soft limiter as PCM encoding rather than turning down
 * all later speech.
 */
internal class ChunkLoudnessStabilizer {
    data class Result(
        val pcm: ByteArray,
        val activeRms: Double?,
        val referenceRms: Double?,
        val requestedGain: Float,
        val startGain: Float,
        val appliedGain: Float,
        val rampSamples: Int,
        val peak: Int,
    )

    private data class Meter(val rms: Double, val peak: Int)

    private var referenceRms: Double? = null
    private var previousGain = 1f

    fun stabilize(pcm: ByteArray): Result {
        val meter = meter(pcm)
            ?: return Result(pcm, null, referenceRms, 1f, 1f, 1f, 0, 0)
        val reference = referenceRms
        if (reference == null) {
            referenceRms = meter.rms
            return Result(pcm, meter.rms, meter.rms, 1f, 1f, 1f, 0, meter.peak)
        }

        val requested = (reference / meter.rms).toFloat()
        // Preserve the corridor itself instead of jumping all the way back to equal loudness.
        val target = when {
            requested > LOUDNESS_CORRIDOR_GAIN -> requested / LOUDNESS_CORRIDOR_GAIN
            requested < 1f / LOUDNESS_CORRIDOR_GAIN -> requested * LOUDNESS_CORRIDOR_GAIN
            else -> 1f
        }
        val applied = target.coerceIn(1f / MAX_CHUNK_GAIN_STEP, MAX_CHUNK_GAIN_STEP)
        val start = previousGain
        previousGain = applied
        if (abs(start - 1f) < 0.0001f && abs(applied - 1f) < 0.0001f) {
            return Result(pcm, meter.rms, reference, requested, 1f, 1f, 0, meter.peak)
        }

        val output = pcm.copyOf()
        val samples = output.size / Short.SIZE_BYTES
        val rampSamples = if (abs(applied - start) < 0.0001f) 0 else min(LOUDNESS_RAMP_SAMPLES, samples)
        val limiterNeeded = meter.peak * max(start, applied) > PCM_SAFE_PEAK
        for (index in 0 until samples) {
            val offset = index * Short.SIZE_BYTES
            val source = ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort().toInt()
            val gain = if (rampSamples > 0 && index < rampSamples) {
                start + (applied - start) * ((index + 1).toFloat() / rampSamples)
            } else {
                applied
            }
            val normalized = source.toDouble() * gain.toDouble() / Short.MAX_VALUE.toDouble()
            val magnitude = abs(normalized)
            val limited = if (!limiterNeeded || magnitude <= PCM_LIMITER_KNEE) {
                normalized
            } else {
                val compressed = PCM_LIMITER_KNEE +
                    (PCM_SAFE_LEVEL - PCM_LIMITER_KNEE) *
                    tanh((magnitude - PCM_LIMITER_KNEE) / (PCM_SAFE_LEVEL - PCM_LIMITER_KNEE))
                if (normalized < 0.0) -compressed else compressed
            }
            val scaled = (limited * Short.MAX_VALUE).roundToInt().coerceIn(-PCM_SAFE_PEAK, PCM_SAFE_PEAK)
            output[offset] = (scaled and 0xff).toByte()
            output[offset + 1] = (scaled shr 8).toByte()
        }
        return Result(output, meter.rms, reference, requested, start, applied, rampSamples, meter.peak)
    }

    private fun meter(pcm: ByteArray): Meter? {
        if (pcm.size < METER_WINDOW_SAMPLES * Short.SIZE_BYTES || pcm.size % Short.SIZE_BYTES != 0) return null
        var peak = 0
        val samples = pcm.size / Short.SIZE_BYTES
        val filtered = DoubleArray(samples)
        var high1Input = 0.0
        var high1Output = 0.0
        var high2Input = 0.0
        var high2Output = 0.0
        var low1 = 0.0
        var low2 = 0.0
        for (index in 0 until samples) {
            val offset = index * Short.SIZE_BYTES
            val sample = ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort().toInt()
            peak = max(peak, abs(sample))
            val high1 = HIGH_PASS_ALPHA * (high1Output + sample - high1Input)
            high1Input = sample.toDouble()
            high1Output = high1
            val high2 = HIGH_PASS_ALPHA * (high2Output + high1 - high2Input)
            high2Input = high1
            high2Output = high2
            low1 += LOW_PASS_MIX * (high2 - low1)
            low2 += LOW_PASS_MIX * (low1 - low2)
            filtered[index] = low2
        }
        if (peak < MIN_METER_PEAK) return null
        val windowCount = 1 + (samples - METER_WINDOW_SAMPLES) / METER_HOP_SAMPLES
        val windowRms = DoubleArray(windowCount)
        var windowIndex = 0
        for (start in 0..samples - METER_WINDOW_SAMPLES step METER_HOP_SAMPLES) {
            var energy = 0.0
            for (index in start until start + METER_WINDOW_SAMPLES) {
                energy += filtered[index] * filtered[index]
            }
            windowRms[windowIndex++] = sqrt(energy / METER_WINDOW_SAMPLES)
        }
        val sorted = windowRms.copyOf(windowIndex).also { it.sort() }
        val upper = sorted[((sorted.lastIndex.toLong() * 9L) / 10L).toInt()]
        val activeFloor = max(
            Short.MAX_VALUE * ACTIVE_RMS_ABSOLUTE_FULL_SCALE,
            upper * ACTIVE_RMS_RELATIVE_UPPER,
        )
        var firstActive = 0
        while (firstActive < sorted.size && sorted[firstActive] < activeFloor) firstActive++
        val activeCount = sorted.size - firstActive
        if (activeCount < MIN_ACTIVE_METER_WINDOWS) return null
        val middle = firstActive + activeCount / 2
        val median = if (activeCount % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
        return Meter(median, peak)
    }

    private companion object {
        private const val PCM_SAFE_LEVEL = 0.95
        private const val PCM_LIMITER_KNEE = 0.90
        private val PCM_SAFE_PEAK = (Short.MAX_VALUE * PCM_SAFE_LEVEL).toInt()
        private const val HIGH_PASS_ALPHA = 0.9690724263048106 // Two 120 Hz one-pole stages.
        private const val LOW_PASS_MIX = 0.8768552889298669 // Two 8 kHz one-pole stages.
        private const val ACTIVE_RMS_ABSOLUTE_FULL_SCALE = 0.0017782794100389228 // -55 dBFS
        private const val ACTIVE_RMS_RELATIVE_UPPER = 0.03162277660168379 // -30 dB
        private const val METER_WINDOW_SAMPLES = KokoroSynthesizer.SAMPLE_RATE / 20 // 50 ms
        private const val METER_HOP_SAMPLES = KokoroSynthesizer.SAMPLE_RATE / 40 // 25 ms
        private const val MIN_ACTIVE_METER_WINDOWS = 2
        private const val MIN_METER_PEAK = 16
        private const val LOUDNESS_CORRIDOR_GAIN = 1.1220185f // +/-1 dB
        private const val MAX_CHUNK_GAIN_STEP = 1.9952623f // +/-6 dB
        private const val LOUDNESS_RAMP_SAMPLES = KokoroSynthesizer.SAMPLE_RATE / 10 // 100 ms
    }
}

/**
 * Joins two renders of the same global frame interval. The previous window contributes the
 * beginning of the overlap and the next window contributes its end; the raised-cosine blend has
 * zero slope at both hand-offs. Duplicate context is collapsed, so the sentence keeps exactly
 * the global front's original sample count.
 */
internal class GlobalPcmOverlapJoiner {
    private var pendingOverlap: ByteArray? = null

    fun stitch(
        pcm: ByteArray,
        leadingHalfOverlapSamples: Int,
        trailingHalfOverlapSamples: Int,
    ): ByteArray {
        require(pcm.size % Short.SIZE_BYTES == 0) { "PCM16 chunk has an odd byte count" }
        require(leadingHalfOverlapSamples >= 0 && trailingHalfOverlapSamples >= 0)
        val leadingBytes = leadingHalfOverlapSamples * 2 * Short.SIZE_BYTES
        val pending = pendingOverlap
        val assembled = when {
            pending == null && leadingBytes == 0 -> pcm
            pending != null && leadingBytes > 0 -> {
                require(pending.size == leadingBytes) {
                    "Global overlap mismatch ${pending.size}/$leadingBytes bytes"
                }
                require(pcm.size >= leadingBytes) { "Continuation PCM is shorter than its overlap" }
                val blended = raisedCosineCrossfade(pending, pcm.copyOfRange(0, leadingBytes))
                ByteArray(blended.size + pcm.size - leadingBytes).also { output ->
                    blended.copyInto(output)
                    pcm.copyInto(output, blended.size, leadingBytes)
                }
            }
            else -> error("Unpaired global PCM overlap")
        }
        pendingOverlap = null

        val trailingBytes = trailingHalfOverlapSamples * 2 * Short.SIZE_BYTES
        if (trailingBytes == 0) return assembled
        require(assembled.size > trailingBytes) { "Global PCM core is not larger than its overlap" }
        pendingOverlap = assembled.copyOfRange(assembled.size - trailingBytes, assembled.size)
        return assembled.copyOf(assembled.size - trailingBytes)
    }

    fun requireComplete() {
        check(pendingOverlap == null) { "Final global PCM overlap was not consumed" }
    }

    internal companion object {
        fun raisedCosineCrossfade(previous: ByteArray, next: ByteArray): ByteArray {
            require(previous.size == next.size && previous.size % Short.SIZE_BYTES == 0)
            val samples = previous.size / Short.SIZE_BYTES
            require(samples >= 2) { "PCM overlap must contain at least two samples" }
            return ByteArray(previous.size).also { output ->
                for (index in 0 until samples) {
                    val offset = index * Short.SIZE_BYTES
                    val first = pcm16Sample(previous, offset)
                    val second = pcm16Sample(next, offset)
                    val phase = index.toDouble() / (samples - 1).toDouble()
                    val nextWeight = 0.5 - 0.5 * kotlin.math.cos(Math.PI * phase)
                    val blended = (first * (1.0 - nextWeight) + second * nextWeight)
                        .roundToInt()
                        .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    output[offset] = (blended and 0xff).toByte()
                    output[offset + 1] = (blended shr 8).toByte()
                }
            }
        }

        private fun pcm16Sample(pcm: ByteArray, offset: Int): Int =
            ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort().toInt()
    }
}

/** Kokoro v1.0 runtime with S24 Ultra QNN AOT acceleration and verified CPU fallback. */
internal class KokoroSynthesizer(
    private val context: Context,
    private val preference: BackendPreference = BackendPreference.AUTO,
) : AutoCloseable {
    companion object {
        const val SAMPLE_RATE = 24_000
        // Generate fewer acoustic frames for the same text. This is applied to
        // the Android-requested speech rate before chunk planning so both the
        // model and the latency planner use the same effective rate.
        internal const val ENGINE_RATE_MULTIPLIER = 1.3f
        private const val MIN_REQUEST_RATE_PERCENT = 80
        private const val MAX_REQUEST_RATE_PERCENT = 250
        private const val MIN_MODEL_SPEED = 0.8f
        private const val MAX_MODEL_SPEED = 2.5f
        private const val TAG = "KokoroRuntime"
        private const val MAX_TOKENS = 510
        private const val FIRST_CHUNK_BASE_TOKENS = 20
        private const val FOLLOWING_CHUNK_BASE_TOKENS = 23
        private const val MIN_FIRST_CHUNK_TOKENS = 16
        private const val MAX_FIRST_CHUNK_TOKENS = 50
        private const val MIN_FOLLOWING_CHUNK_TOKENS = 18
        private const val MAX_FOLLOWING_CHUNK_TOKENS = 50
        private const val MIN_TAIL_TOKENS = 10
        // This is an iterative operation budget, not call-stack recursion. B192-only refinement
        // can need more than eight word-safe splits for a long Android sentence request.
        private const val MAX_FRAME_SPLIT_DEPTH = 16
        private const val DURATION_UNIT_GENERATOR_FRAMES = 2
        private const val PCM_CACHE_BYTES = 12 * 1024 * 1024
        // Every v1 HTP context is a repaired masked neural-vocoder suffix.
        // The tiny source-spectrum branch and iSTFT stay on CPU because the
        // SM8650 QNN implementation miscomputes a voiced gate and Pow(x, 2).
        // The suffix replaces every square Pow with exact Mul(x, x), retaining
        // the heavy neural work on HTP without the former high-pitched output.
        private const val QNN_V1_B64_FRAMES = 64
        private const val QNN_V1_B96_FRAMES = 96
        private const val QNN_V1_B128_FRAMES = 128
        private const val QNN_V1_B192_FRAMES = 192
        private const val QNN_V1_B208_FRAMES = 208
        private const val QNN_V1_B224_FRAMES = 224
        private const val QNN_V1_B256_FRAMES = 256
        private const val QNN_V1_B320_FRAMES = 320
        private const val QNN_V1_B384_FRAMES = 384
        private const val QNN_V1_B512_FRAMES = 512
        private const val QNN_V1_B640_FRAMES = 640
        private const val GLOBAL_FRONT_OPENING_WINDOW_FRAMES = QNN_V1_B128_FRAMES
        // A 112-frame opening carries roughly 1.25 seconds of post-trim speech. That safely
        // covers a warm continuation inference on SM8650 without reverting to a slower B192
        // opening. Keep two future frames so the full 50 ms shared join remains available.
        private const val GLOBAL_FRONT_MIN_OPENING_RUNWAY_FRAMES = 112
        private const val GLOBAL_FRONT_MAX_OPENING_CORE_FRAMES = 126
        private const val GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES = QNN_V1_B192_FRAMES
        private const val GLOBAL_FRONT_FUTURE_CONTEXT_FRAMES = 40
        private const val GLOBAL_FRONT_MIN_CONTINUATION_CONTEXT_FRAMES = 32
        // Keep ordinary continuations inside the already-prewarmed B192 session with the full
        // 32-frame context allowance.  Loading B256/B320 after a short queued core took longer
        // than the buffered speech on SM8650 and caused a framework playback underrun. Larger
        // semantic cores are word-safely refined while retaining the one full-sentence front.
        private const val GLOBAL_FRONT_MAX_STREAMING_CORE_FRAMES =
            GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES - GLOBAL_FRONT_MIN_CONTINUATION_CONTEXT_FRAMES
        // A sub-runway bridge immediately before a heavy core is joined to its predecessor only
        // when the result retains the normal B192 32-frame context allowance.
        private const val GLOBAL_FRONT_MAX_COALESCED_BRIDGE_CORE_FRAMES =
            GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES - GLOBAL_FRONT_MIN_CONTINUATION_CONTEXT_FRAMES
        // Two frames on either side produce a 50 ms shared-timeline crossfade at 24 kHz.
        private const val GLOBAL_JOIN_HALF_OVERLAP_FRAMES = 2
        private const val QNN_PREWARM_SESSION_COUNT = 3
        private val QNN_READING_PREWARM_DEFAULTS =
            listOf(QNN_V1_B128_FRAMES, QNN_V1_B192_FRAMES, QNN_V1_B208_FRAMES)
        // Kept for the unreachable v0.19 compatibility helpers below.  No
        // v1 selection path returns these values and no matching assets ship.
        private const val QNN_B256_FRAMES = 256
        private const val QNN_B384_FRAMES = 384
        private const val FRAME_SAMPLES = 300
        private const val PCM_HEADROOM = 0.95f
        private const val PCM_LIMITER_KNEE = 0.90f
        private const val PCM_ROBUST_PEAK_PERCENTILE_NUMERATOR = 995L
        private const val PCM_ROBUST_PEAK_PERCENTILE_DENOMINATOR = 1_000L
        private const val MIN_PCM_DYNAMIC_RANGE = 1e-5f
        private const val EDGE_ACTIVE_RELATIVE_RMS = 0.005623413f // -45 dB
        private const val EDGE_ACTIVE_WINDOW_SAMPLES = SAMPLE_RATE / 100 // 10 ms
        private const val EDGE_ACTIVE_HOP_SAMPLES = SAMPLE_RATE / 200 // 5 ms
        private const val EDGE_ACTIVE_SUSTAINED_WINDOWS = 3 // 20 ms from first start to last end
        private const val MIN_EDGE_TRIM_SAMPLES = SAMPLE_RATE / 200 // 5 ms
        private const val MAX_EDGE_TRIM_SAMPLES = SAMPLE_RATE * 4 / 5 // 800 ms
        private const val CONTINUATION_LEADING_SAMPLES = SAMPLE_RATE * 30 / 1_000
        private const val CONTINUATION_TRAILING_SAMPLES = SAMPLE_RATE * 40 / 1_000
        // @Voice commonly submits one sentence per Android utterance and Samsung recreates a TTS
        // AudioTrack at that boundary. Its measured output latency can exceed the former 70 ms
        // leading guard, making a shortened 1.3x first phoneme sound clipped even though callback
        // PCM is intact. Keep the established 150 ms total boundary budget, but place 120 ms in
        // front of the next sentence and 30 ms after the previous one. Only model-generated
        // near-zero padding is retained; active speech is never removed or time-stretched.
        private const val REQUEST_BOUNDARY_LEADING_SAMPLES = SAMPLE_RATE * 120 / 1_000
        private const val REQUEST_BOUNDARY_TRAILING_SAMPLES = SAMPLE_RATE * 30 / 1_000
        private const val COMMA_TRAILING_SAMPLES = SAMPLE_RATE * 400 / 1_000
        private const val SEMICOLON_TRAILING_SAMPLES = SAMPLE_RATE * 420 / 1_000
        private const val COLON_TRAILING_SAMPLES = SAMPLE_RATE * 565 / 1_000
        private const val PERIOD_TRAILING_SAMPLES = SAMPLE_RATE * 530 / 1_000
        private const val QUESTION_TRAILING_SAMPLES = SAMPLE_RATE * 545 / 1_000
        private const val CONTINUATION_CROSSFADE_SAMPLES = SAMPLE_RATE / 100 // 10 ms
        private const val MAX_QNN_BUCKET_SESSIONS = 3
        private const val QNN_SOC_MODEL = "57"
        private const val QNN_HTP_ARCH = "75"
        private const val TARGET_ANDROID_SOC = "SM8650"
        private const val QNN_DISABLED_KEY = "qnn_disabled_v1_powmul_source_spectrum_v13"
        private const val NNAPI_DISABLED_KEY = "nnapi_disabled_masked_fp32_v2"
        private const val FRONT_CONDITIONING = "/decoder/Slice_output_0"
        private const val FRONT_PROSODY = "/decoder/decoder/Unsqueeze_output_0"
        private const val FRONT_DECODER = "/decoder/decoder/decode.3/Div_4_output_0"
        private const val FRONT_TOKEN_DURATIONS = "/encoder/Cast_output_0"
        private const val HARMONIC_SOURCE = "kokoro_harmonic_source"
        private const val SOURCE_SPECTRUM = "kokoro_source_spectrum"
        private const val VALID_MASK_10 = "valid_mask_10"
        private const val VALID_MASK_60 = "valid_mask_60"
        private const val QNN_CONTEXT_SCHEMA = "kokoro-v1-qnn248-powmul-source-spectrum-v13"
        private const val QNN_SESSION_SOURCE = "PACKAGED_SHARED_AOT_QNN248_MASKED_V1"
        private const val QNN_FALLBACK_SESSION_SOURCE = "PACKAGED_EMBEDDED_AOT_QNN248_MASKED_V1"
        private const val QNN_PERFORMANCE_POLICY =
            "embedded.provider=balanced;embedded.vtcm_mb=8_for_B192_B512_B640;" +
                "embedded.run.qnn.perf_mode=burst;external_shared_weights=disabled_on_android;" +
                "prewarm=B128_B192_B208_max3_lru3;q8_fallback=lazy_only;" +
                "global_front=duration_aligned_prewarmed_B128_B192_max160_bridge_coalesce_refine_depth16;" +
                "opening_runway=duration_min112_exact_B128_promotes_B192;" +
                "request_edge=leading120ms_trailing30ms_fixed150ms;" +
                "pcm=robust_p995_soft_limiter;loudness=speech_band_median_1db_corridor_6db_cap;" +
                "cpu_prefix=source_spectrum_only;cpu_suffix=istft_only"
        private const val QNN_PRECISION_POLICY =
            "graph_io=FP32;all_buckets=QAIRT_native_lowering;Pow_x2=Mul_xx;" +
                "unsafe_harmonic_source_gate=CPU"
        private const val RUNTIME_PREFERENCES = "kokoro_runtime"
        private const val DIAGNOSTIC_BACKEND_KEY = "last_backend"
        private const val DIAGNOSTIC_BUCKET_KEY = "last_bucket"
        private const val DIAGNOSTIC_RTF_KEY = "last_rtf"
        private const val DIAGNOSTIC_CONTEXT_SOURCE_KEY = "last_context_source"
        private const val DIAGNOSTIC_CONTEXT_HASH_KEY = "last_context_hash_prefix"
        private const val DIAGNOSTIC_FAILURE_KEY = "last_failure_reason"
        private const val DIAGNOSTIC_TIMESTAMP_KEY = "last_run_at_utc_ms"
        private const val QNN_RETRY_GENERATION_KEY =
            "qnn_retry_generation_v1_powmul_source_spectrum_v13"

        private val qnnPluginLock = Any()
        private val packagedAssetInstallLock = Any()
        @Volatile private var qnnPluginRegistered = false

        internal fun aotBucketForFrames(frames: Int): Int? {
            require(frames > 0) { "Generator frame count must be positive" }
            return when (frames) {
                in 1..QNN_V1_B64_FRAMES -> QNN_V1_B64_FRAMES
                in 65..QNN_V1_B96_FRAMES -> QNN_V1_B96_FRAMES
                in 97..QNN_V1_B128_FRAMES -> QNN_V1_B128_FRAMES
                in 129..QNN_V1_B192_FRAMES -> QNN_V1_B192_FRAMES
                in 193..QNN_V1_B208_FRAMES -> QNN_V1_B208_FRAMES
                in 209..QNN_V1_B224_FRAMES -> QNN_V1_B224_FRAMES
                in 225..QNN_V1_B256_FRAMES -> QNN_V1_B256_FRAMES
                in 257..QNN_V1_B320_FRAMES -> QNN_V1_B320_FRAMES
                in 321..QNN_V1_B384_FRAMES -> QNN_V1_B384_FRAMES
                in 385..QNN_V1_B512_FRAMES -> QNN_V1_B512_FRAMES
                in 513..QNN_V1_B640_FRAMES -> QNN_V1_B640_FRAMES
                else -> null
            }
        }

        /** Keeps the established B192 continuation route for ordinary cores. An oversized
         * semantic core may use the smallest packaged bucket that also preserves the normal
         * 32-frame context allowance. Near the largest packaged bucket, retaining the shared
         * overlap is the hard minimum; otherwise the caller must use its bounded fallback. */
        internal fun globalContinuationWindowFramesForTesting(coreFrames: Int): Int? {
            require(coreFrames > 0) { "Global continuation core must be positive" }
            // A continuation that consumes the entire B192 window has no shared temporal
            // context.  That exact-fit route is not merely a rough seam: some voices (notably
            // am_puck) can collapse the continuation to silence.  Reserve both two-frame join
            // halves before accepting B192; otherwise move to the smallest packaged bucket that
            // can preserve the overlap contract.
            val minimumSharedContextFrames = GLOBAL_JOIN_HALF_OVERLAP_FRAMES * 2
            if (coreFrames + minimumSharedContextFrames <= GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES) {
                return GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES
            }

            fun bucketWith(extraFrames: Int): Int? {
                val required = coreFrames.toLong() + extraFrames.toLong()
                return if (required <= Int.MAX_VALUE.toLong()) {
                    aotBucketForFrames(required.toInt())
                } else {
                    null
                }
            }

            // Preserve the already-qualified oversized-core policy: once the semantic core is
            // genuinely larger than B192, prefer the normal 32-frame context allowance.  The
            // smaller four-frame escape hatch is only for a near-exact B192 fit, where it avoids
            // an all-zero continuation without needlessly jumping to B256.
            return if (coreFrames <= GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES) {
                bucketWith(minimumSharedContextFrames)
            } else {
                bucketWith(GLOBAL_FRONT_MIN_CONTINUATION_CONTEXT_FRAMES)
                    ?: bucketWith(minimumSharedContextFrames)
            }
        }

        internal fun globalOpeningWindowFramesForTesting(coreFrames: Int): Int? {
            require(coreFrames > 0) { "Global opening core must be positive" }
            return if (coreFrames <= GLOBAL_FRONT_MAX_OPENING_CORE_FRAMES) {
                GLOBAL_FRONT_OPENING_WINDOW_FRAMES
            } else {
                globalContinuationWindowFramesForTesting(coreFrames)
            }
        }

        /**
         * Allocates a global-front generator window while reserving both sides of every internal
         * core for the shared-frame join. Future context remains preferred up to the established
         * 40-frame target; only capacity beyond the right-side preference is assigned to past
         * context. Near a sentence edge, otherwise-idle capacity is reused on the available side.
         */
        internal fun globalContextWindowForTesting(
            sentenceFrames: Int,
            coreStart: Int,
            coreEnd: Int,
            windowFrames: Int,
            opening: Boolean,
        ): Pair<Int, Int>? {
            require(sentenceFrames > 0) { "Global sentence must contain frames" }
            require(coreStart in 0 until coreEnd && coreEnd <= sentenceFrames) {
                "Invalid global core $coreStart..$coreEnd/$sentenceFrames"
            }
            val coreFrames = coreEnd - coreStart
            require(windowFrames >= coreFrames) {
                "Window T=$windowFrames cannot contain core T=$coreFrames"
            }
            if (sentenceFrames <= windowFrames) return 0 to sentenceFrames
            if (opening) {
                require(coreStart == 0) { "Opening core must start at the sentence boundary" }
                val contextEnd = min(sentenceFrames, windowFrames)
                val requiredRight = min(GLOBAL_JOIN_HALF_OVERLAP_FRAMES, sentenceFrames - coreEnd)
                return if (contextEnd - coreEnd >= requiredRight) 0 to contextEnd else null
            }

            val availableLeft = coreStart
            val availableRight = sentenceFrames - coreEnd
            val requiredLeft = min(GLOBAL_JOIN_HALF_OVERLAP_FRAMES, availableLeft)
            val requiredRight = min(GLOBAL_JOIN_HALF_OVERLAP_FRAMES, availableRight)
            val spare = windowFrames - coreFrames
            if (spare < requiredLeft + requiredRight) return null

            var future = min(
                min(availableRight, GLOBAL_FRONT_FUTURE_CONTEXT_FRAMES),
                spare - requiredLeft,
            )
            if (future < requiredRight) future = requiredRight
            var past = min(availableLeft, spare - future)
            var remaining = spare - past - future
            if (remaining > 0) {
                val extraFuture = min(availableRight - future, remaining)
                future += extraFuture
                remaining -= extraFuture
            }
            if (remaining > 0) {
                past += min(availableLeft - past, remaining)
            }
            return (coreStart - past) to (coreEnd + future)
        }

        internal fun hasContiguousAotCoverageThrough(frames: Int): Boolean {
            require(frames > 0) { "Coverage frame count must be positive" }
            return (1..frames).all { aotBucketForFrames(it) != null }
        }

        /** Keeps every bucket used by the normal global-sentence streaming route resident.
         *
         * Restoring an arbitrary last bucket displaced B208 from the three-session LRU.  A
         * near-exact B192 continuation then had to load B208 while B128 audio was playing, which
         * produced a reproducible playback underrun in @Voice.  The fixed set is bounded by the
         * existing LRU and deliberately ignores history from uncommon one-off routes.
         */
        internal fun qnnPrewarmBuckets(@Suppress("UNUSED_PARAMETER") lastBucket: Int?): List<Int> =
            QNN_READING_PREWARM_DEFAULTS.take(QNN_PREWARM_SESSION_COUNT)

        internal fun modelSpeedForRequest(requestedPercent: Int): Float =
            ((requestedPercent.coerceIn(MIN_REQUEST_RATE_PERCENT, MAX_REQUEST_RATE_PERCENT) / 100f) *
                ENGINE_RATE_MULTIPLIER).coerceIn(MIN_MODEL_SPEED, MAX_MODEL_SPEED)

        internal fun modelSpeedWithDeliveryMultiplier(baseSpeed: Float, multiplier: Float): Float =
            (baseSpeed * multiplier.coerceIn(
                ExpressionSettings.MIN_DELIVERY_SPEED,
                ExpressionSettings.MAX_DELIVERY_SPEED,
            )).coerceIn(MIN_MODEL_SPEED, MAX_MODEL_SPEED)

        internal fun isFullWaveformQnnBucket(@Suppress("UNUSED_PARAMETER") bucket: Int): Boolean = false

        /**
         * Coalesces only one adjacent pair from the original latency plan.  The caller supplies
         * CPU-front duration measurements; token counts remain a hard safety bound and an
         * existing static AOT bucket remains the sole eligibility test.  It never recursively
         * folds a newly joined span with a third one, so ordering and punctuation stay intact.
         */
        internal fun coalescePostDurationForTesting(
            spans: List<String>,
            tokenCount: (String) -> Int,
            frameCount: (String) -> Int,
        ): List<String> {
            val coalesced = mutableListOf<String>()
            var index = 0
            while (index < spans.size) {
                val current = spans[index]
                val frames = frameCount(current)
                val shortUnqualified = frames in 1 until QNN_V1_B192_FRAMES && aotBucketForFrames(frames) == null
                if (shortUnqualified && index < spans.lastIndex) {
                    val joined = "$current ${spans[index + 1]}"
                    if (tokenCount(joined) <= MAX_TOKENS - 2 && aotBucketForFrames(frameCount(joined)) != null) {
                        coalesced += joined
                        index += 2
                        continue
                    }
                }
                coalesced += current
                index++
            }
            return coalesced
        }

        internal fun chunkTokenLimitsForSpeed(speed: Float): Pair<Int, Int> {
            require(speed.isFinite() && speed > 0f) { "Speech speed must be positive and finite" }
            val first = (FIRST_CHUNK_BASE_TOKENS * speed).roundToInt()
                .coerceIn(MIN_FIRST_CHUNK_TOKENS, MAX_FIRST_CHUNK_TOKENS)
            val following = (FOLLOWING_CHUNK_BASE_TOKENS * speed).roundToInt()
                .coerceIn(MIN_FOLLOWING_CHUNK_TOKENS, MAX_FOLLOWING_CHUNK_TOKENS)
            return first to following
        }

        /**
         * Keeps each lookahead span inside the same speed-scaled latency envelope. The first span
         * is hard-bounded so a one-sentence Android request cannot silently become a whole-sentence
         * render before any PCM is available; bounding continuations prevents a cold large-bucket
         * render from exhausting the already-queued opening.
         * The CPU front-end frame guard remains authoritative: token counts deliberately do not
         * pretend to predict duration for every voice or speed.
         */
        internal fun splitForLatencyForTesting(
            input: String,
            firstTarget: Int,
            followingTarget: Int,
            countTokens: (String) -> Int,
        ): List<String> {
            require(firstTarget > 0 && followingTarget > 0) { "Chunk targets must be positive" }
            val chunks = mutableListOf<String>()
            var remaining = input.trim()
            if (remaining.isEmpty()) return chunks
            var first = true
            while (remaining.isNotEmpty()) {
                val target = if (first) firstTarget else followingTarget
                val slack = if (first) {
                    max(MIN_TAIL_TOKENS, target / 3)
                } else {
                    max(12, target / 2)
                }
                val hardLimit = min(MAX_TOKENS - 2, target + slack)
                val totalTokens = countTokens(remaining)
                // Keep genuinely short utterances whole. Once there is room for both a
                // target-sized opening and a useful tail, stream the opening even if the
                // complete sentence still fits inside the old semantic-whole window.
                val usefulProgressiveSplit = first && totalTokens >= target + MIN_TAIL_TOKENS
                if (totalTokens <= hardLimit && !usefulProgressiveSplit) {
                    chunks += remaining
                    break
                }

                val boundary = preferredTextBoundary(
                    input = remaining,
                    targetTokens = target,
                    hardTokens = hardLimit,
                    minimumTailTokens = MIN_TAIL_TOKENS,
                    countTokens = countTokens,
                ) ?: fallbackTokenBoundary(remaining, target, MIN_TAIL_TOKENS, countTokens)
                if (boundary == null) {
                    chunks += remaining
                    break
                }

                val head = remaining.substring(0, boundary).trim()
                val tail = remaining.substring(boundary).trim()
                if (head.isEmpty() || tail.isEmpty()) {
                    chunks += remaining
                    break
                }
                chunks += head
                remaining = tail
                first = false
            }
            return chunks
        }

        /**
         * Chooses source-text boundaries before G2P so syntax survives planning. The returned
         * pieces are subsequently mapped onto the one full-sentence phoneme stream; this never
         * conditions a second front or inserts a pause at the selected boundary.
         */
        internal fun splitSourceForLatencyForTesting(
            input: String,
            firstTarget: Int,
            followingTarget: Int,
            countTokens: (String) -> Int,
        ): List<String> {
            require(firstTarget > 0 && followingTarget > 0) { "Chunk targets must be positive" }
            val chunks = mutableListOf<String>()
            var remaining = input.trim()
            var first = true
            while (remaining.isNotEmpty()) {
                val target = if (first) firstTarget else followingTarget
                val slack = if (first) max(MIN_TAIL_TOKENS, target / 3) else max(12, target / 2)
                val hardLimit = min(MAX_TOKENS - 2, target + slack)
                val totalTokens = countTokens(remaining)
                val usefulProgressiveSplit = first && totalTokens >= target + MIN_TAIL_TOKENS
                if (totalTokens <= hardLimit && !usefulProgressiveSplit) {
                    chunks += remaining
                    break
                }
                val boundary = preferredSourceTextBoundary(
                    input = remaining,
                    targetTokens = target,
                    hardTokens = hardLimit,
                    minimumTailTokens = MIN_TAIL_TOKENS,
                    countTokens = countTokens,
                )
                if (boundary == null) {
                    chunks += remaining
                    break
                }
                val head = remaining.substring(0, boundary).trim()
                val tail = remaining.substring(boundary).trim()
                if (head.isEmpty() || tail.isEmpty()) {
                    chunks += remaining
                    break
                }
                chunks += head
                remaining = tail
                first = false
            }
            return chunks
        }

        /** Maps source-selected token counts to word-safe boundaries in the existing full G2P. */
        internal fun mapSourceBoundariesToPhonemesForTesting(
            phonemes: String,
            sourcePrefixTokenCounts: List<Int>,
            countTokens: (String) -> Int,
        ): List<String>? {
            var previous = 0
            val boundaries = mutableListOf<Int>()
            sourcePrefixTokenCounts.forEach { expectedTokens ->
                val candidates = textBoundaryPattern.findAll(phonemes)
                    .map { it.range.last + 1 }
                    .filter { it > previous && it < phonemes.length }
                    .toList()
                val boundary = candidates.firstOrNull { end ->
                    countTokens(phonemes.substring(0, end)) == expectedTokens
                } ?: candidates.minWithOrNull(compareBy<Int> { end ->
                    abs(countTokens(phonemes.substring(0, end)) - expectedTokens)
                }.thenBy { it }) ?: return null
                boundaries += boundary
                previous = boundary
            }
            val pieces = mutableListOf<String>()
            var start = 0
            (boundaries + phonemes.length).forEach { end ->
                val piece = phonemes.substring(start, end).trim()
                if (piece.isEmpty()) return null
                pieces += piece
                start = end
            }
            return pieces
        }

        /** Maps linguistic chunk starts onto the exact duration-expanded frame path exported by
         * the full-sentence front. The duration vector contains BOS, every model token, and EOS;
         * Kokoro expands every duration unit into two generator frames. Delimiter tokens between
         * trimmed chunks stay with the preceding core, so no full-sentence frame is lost. */
        internal fun durationAlignedBoundariesForTesting(
            input: String,
            chunks: List<String>,
            tokenDurations: LongArray,
            countTokens: (String) -> Int,
        ): List<Pair<Int, Int>> {
            require(input.isNotEmpty() && chunks.isNotEmpty()) { "Duration alignment requires text chunks" }
            val fullTokenCount = countTokens(input)
            require(tokenDurations.size == fullTokenCount + 2) {
                "Duration count ${tokenDurations.size} does not match $fullTokenCount tokens"
            }
            require(tokenDurations.all { it > 0L }) { "Token durations must be positive" }
            val starts = ArrayList<Int>(chunks.size)
            var cursor = 0
            chunks.forEach { chunk ->
                val start = input.indexOf(chunk, cursor)
                require(start >= cursor) { "Chunk is not an ordered substring: $chunk" }
                starts += start
                cursor = start + chunk.length
            }
            val totalFrames = Math.multiplyExact(
                tokenDurations.sum(),
                DURATION_UNIT_GENERATOR_FRAMES.toLong(),
            ).toInt()
            val boundaries = ArrayList<Pair<Int, Int>>(chunks.size + 1)
            boundaries += 0 to 0
            starts.drop(1).forEach { charIndex ->
                val prefixTokens = countTokens(input.substring(0, charIndex))
                val durationUnits = tokenDurations.copyOfRange(0, prefixTokens + 1).sum()
                val frame = Math.multiplyExact(
                    durationUnits,
                    DURATION_UNIT_GENERATOR_FRAMES.toLong(),
                ).toInt()
                require(frame > boundaries.last().second && frame < totalFrames) {
                    "Non-increasing duration boundary $frame/$totalFrames"
                }
                boundaries += charIndex to frame
            }
            boundaries += input.length to totalFrames
            return boundaries
        }

        /**
         * Produces a latency-safe global stream from the one full-sentence duration path. A tiny
         * interior bridge immediately before a heavy core may be absorbed by its predecessor
         * when the combined core remains in the normal B192 context envelope. Continuations
         * above the prewarmed B192/32-frame envelope are then word-safely split. Every mutation remaps all
         * boundaries against the original sentence and the same BOS/token/EOS duration vector;
         * no child part is independently phonemized or fronted.
         */
        internal fun refineGlobalDurationPartsForTesting(
            input: String,
            chunks: List<String>,
            tokenDurations: LongArray,
            countTokens: (String) -> Int,
            splitChunk: (String) -> Pair<String, String>?,
            maxRefinements: Int = MAX_FRAME_SPLIT_DEPTH,
        ): GlobalDurationRefinement? {
            require(maxRefinements >= 0) { "Maximum duration plan refinements must not be negative" }
            val refined = chunks.toMutableList()
            var boundaries = durationAlignedBoundariesForTesting(
                input = input,
                chunks = refined,
                tokenDurations = tokenDurations,
                countTokens = countTokens,
            )
            var refinements = 0
            var bridgeCoalescences = 0
            while (true) {
                val bridgeIndex = (2 until refined.lastIndex).firstOrNull { index ->
                    val previousFrames = boundaries[index].second - boundaries[index - 1].second
                    val bridgeFrames = boundaries[index + 1].second - boundaries[index].second
                    val nextFrames = boundaries[index + 2].second - boundaries[index + 1].second
                    bridgeFrames < GLOBAL_FRONT_MIN_OPENING_RUNWAY_FRAMES &&
                        nextFrames > GLOBAL_FRONT_MAX_STREAMING_CORE_FRAMES &&
                        previousFrames + bridgeFrames <= GLOBAL_FRONT_MAX_COALESCED_BRIDGE_CORE_FRAMES
                }
                if (bridgeIndex != null) {
                    if (refinements + bridgeCoalescences >= maxRefinements) return null
                    val merged = input.substring(
                        boundaries[bridgeIndex - 1].first,
                        boundaries[bridgeIndex + 1].first,
                    ).trim()
                    if (merged.isEmpty()) return null
                    refined[bridgeIndex - 1] = merged
                    refined.removeAt(bridgeIndex)
                    bridgeCoalescences++
                    boundaries = durationAlignedBoundariesForTesting(
                        input = input,
                        chunks = refined,
                        tokenDurations = tokenDurations,
                        countTokens = countTokens,
                    )
                    continue
                }

                var offendingIndex = -1
                for (index in 0 until boundaries.lastIndex) {
                    val coreFrames = boundaries[index + 1].second - boundaries[index].second
                    val windowFrames = if (index == 0) {
                        globalOpeningWindowFramesForTesting(coreFrames)
                    } else {
                        globalContinuationWindowFramesForTesting(coreFrames)
                    }
                    val exceedsStreamingEnvelope =
                        index > 0 && coreFrames > GLOBAL_FRONT_MAX_STREAMING_CORE_FRAMES
                    if (windowFrames == null || exceedsStreamingEnvelope) {
                        offendingIndex = index
                        break
                    }
                }
                if (offendingIndex < 0) {
                    return GlobalDurationRefinement(
                        refined.toList(),
                        boundaries,
                        refinements,
                        bridgeCoalescences,
                    )
                }
                if (refinements + bridgeCoalescences >= maxRefinements) return null

                val original = refined[offendingIndex]
                val split = splitChunk(original) ?: return null
                val head = split.first.trim()
                val tail = split.second.trim()
                if (head.isEmpty() || tail.isEmpty() || head == original || tail == original) return null
                refined[offendingIndex] = head
                refined.add(offendingIndex + 1, tail)
                refinements++
                boundaries = durationAlignedBoundariesForTesting(
                    input = input,
                    chunks = refined,
                    tokenDurations = tokenDurations,
                    countTokens = countTokens,
                )
            }
        }

        /** Moves an oversized opening only to an earlier word boundary whose exact expanded
         * duration fits the requested generator window. The caller supplies the same connector
         * safety predicate used by the semantic planner. */
        internal fun boundedDurationOpeningForTesting(
            input: String,
            originalBoundary: Pair<Int, Int>,
            tokenDurations: LongArray,
            maxFrames: Int,
            countTokens: (String) -> Int,
            isSafeTail: (String) -> Boolean = { true },
        ): Pair<Int, Int>? {
            require(originalBoundary.first in 1 until input.length)
            require(maxFrames > 0)
            for (charIndex in originalBoundary.first downTo 1) {
                if (!input[charIndex - 1].isWhitespace()) continue
                val head = input.substring(0, charIndex).trimEnd()
                if (head.isEmpty()) continue
                val tail = head.substringAfterLast(' ')
                if (!isSafeTail(tail)) continue
                val prefixTokens = countTokens(input.substring(0, charIndex))
                val durationUnits = tokenDurations.copyOfRange(0, prefixTokens + 1).sum()
                val frame = Math.multiplyExact(
                    durationUnits,
                    DURATION_UNIT_GENERATOR_FRAMES.toLong(),
                ).toInt()
                if (frame in 1..maxFrames) return charIndex to frame
            }
            return null
        }

        /**
         * Moves an undersized opening forward to the first speech-safe word boundary that gives
         * the continuation prefetch enough playback runway. Unlike token-count expansion, this
         * uses the already-computed voice-specific durations, so a slow voice is not mistaken for
         * a short opening (or vice versa).
         */
        internal fun expandedDurationOpeningForTesting(
            input: String,
            originalBoundary: Pair<Int, Int>,
            stopBeforeChar: Int,
            tokenDurations: LongArray,
            minFrames: Int,
            maxFrames: Int,
            countTokens: (String) -> Int,
            isSafeTail: (String) -> Boolean = { true },
        ): Pair<Int, Int>? {
            require(originalBoundary.first in 1 until input.length)
            require(stopBeforeChar in (originalBoundary.first + 1)..input.length)
            require(minFrames in 1..maxFrames)
            for (charIndex in originalBoundary.first + 1 until stopBeforeChar) {
                if (!input[charIndex - 1].isWhitespace()) continue
                val head = input.substring(0, charIndex).trimEnd()
                if (head.isEmpty()) continue
                val tail = head.substringAfterLast(' ')
                if (!isSafeTail(tail)) continue
                val prefixTokens = countTokens(input.substring(0, charIndex))
                val durationUnits = tokenDurations.copyOfRange(0, prefixTokens + 1).sum()
                val frame = Math.multiplyExact(
                    durationUnits,
                    DURATION_UNIT_GENERATOR_FRAMES.toLong(),
                ).toInt()
                if (frame > maxFrames) return null
                if (frame >= minFrames) return charIndex to frame
            }
            return null
        }

        /**
         * Produces the first value synchronously, then keeps at most one value in flight while
         * the caller consumes the current value. Consumer callbacks always stay on the caller.
         */
        internal fun <T, R> consumeOneChunkAhead(
            items: List<T>,
            produce: (T) -> R,
            consume: (R) -> Boolean,
            cancelProducer: () -> Unit = {},
        ) {
            if (items.isEmpty()) return
            if (items.size == 1) {
                consume(produce(items[0]))
                return
            }

            val executor = Executors.newSingleThreadExecutor { runnable ->
                Thread(
                    {
                        try {
                            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO)
                        } catch (_: Exception) {
                            // Best effort: priority changes can be denied on vendor builds.
                        }
                        runnable.run()
                    },
                    "kokoro-chunk-prefetch",
                ).apply { isDaemon = true }
            }
            var pending: Future<R>? = null
            try {
                var current = produce(items[0])
                for (index in items.indices) {
                    if (index < items.lastIndex) {
                        pending = executor.submit(Callable { produce(items[index + 1]) })
                    }
                    if (!consume(current)) return
                    if (index < items.lastIndex) {
                        current = awaitProduced(checkNotNull(pending))
                        pending = null
                    }
                }
            } finally {
                if (pending?.isDone == false) {
                    try {
                        cancelProducer()
                    } catch (_: Exception) {
                        // Preserve the synthesis/callback failure which led to cancellation.
                    }
                    pending?.cancel(true)
                }
                executor.shutdownNow()
                while (!executor.awaitTermination(100, TimeUnit.MILLISECONDS)) {
                    try {
                        cancelProducer()
                    } catch (_: Exception) {
                        // Keep waiting until no session can still be in use.
                    }
                }
            }
        }

        private fun <R> awaitProduced(future: Future<R>): R = try {
            future.get()
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
            throw CancellationException("Chunk prefetch interrupted").apply { initCause(interrupted) }
        } catch (failed: ExecutionException) {
            when (val cause = failed.cause ?: failed) {
                is Error -> throw cause
                is Exception -> throw cause
                else -> throw IllegalStateException("Chunk prefetch failed", cause)
            }
        }

        /** Removes only contiguous, near-zero model padding at a generated edge. */
        internal fun trimArtificialEdgeSilence(
            values: FloatArray,
            keepLeadingSamples: Int?,
            keepTrailingSamples: Int?,
        ): FloatArray {
            require(keepLeadingSamples == null || keepLeadingSamples >= 0)
            require(keepTrailingSamples == null || keepTrailingSamples >= 0)
            if (values.isEmpty()) return values

            val activeRange = sustainedActiveRange(values) ?: return values
            val leadingQuiet = if (keepLeadingSamples != null) activeRange.first else 0
            val trailingQuiet = if (keepTrailingSamples != null) values.size - activeRange.last else 0

            fun removable(quiet: Int, keep: Int?): Int {
                if (keep == null) return 0
                val excess = quiet - keep
                return if (excess >= MIN_EDGE_TRIM_SAMPLES) {
                    min(excess, MAX_EDGE_TRIM_SAMPLES)
                } else {
                    0
                }
            }

            val removeLeading = removable(leadingQuiet, keepLeadingSamples)
            val requestedTrailing = removable(trailingQuiet, keepTrailingSamples)
            val removeTrailing = min(requestedTrailing, (values.size - removeLeading - 1).coerceAtLeast(0))
            if (removeLeading == 0 && removeTrailing == 0) return values
            return values.copyOfRange(removeLeading, values.size - removeTrailing)
        }

        internal fun joinWaveformsAtSeam(
            first: FloatArray,
            second: FloatArray,
            seam: WaveformSeam,
        ): FloatArray {
            val left = trimArtificialEdgeSilence(first, null, trailingSamplesFor(seam))
            val right = trimArtificialEdgeSilence(second, leadingSamplesFor(seam), null)
            val overlap = if (seam == WaveformSeam.CONTINUATION) {
                min(CONTINUATION_CROSSFADE_SAMPLES, min(left.size, right.size))
                    .takeIf { candidate ->
                        candidate > 0 &&
                            isRelativelyQuiet(left, left.size - candidate, left.size) &&
                            isRelativelyQuiet(right, 0, candidate)
                    } ?: 0
            } else {
                0
            }
            check(left.size <= Int.MAX_VALUE - right.size + overlap) {
                "Combined generator audio is too large"
            }
            return FloatArray(left.size + right.size - overlap).also { combined ->
                left.copyInto(combined, endIndex = left.size - overlap)
                for (index in 0 until overlap) {
                    val ratio = (index + 1f) / (overlap + 1f)
                    val leftGain = kotlin.math.sqrt(1f - ratio)
                    val rightGain = kotlin.math.sqrt(ratio)
                    combined[left.size - overlap + index] =
                        left[left.size - overlap + index] * leftGain + right[index] * rightGain
                }
                right.copyInto(combined, left.size, startIndex = overlap)
            }
        }

        private data class ActiveSampleRange(val first: Int, val last: Int)

        private fun sustainedActiveRange(values: FloatArray): ActiveSampleRange? {
            if (values.size < EDGE_ACTIVE_WINDOW_SAMPLES) return null
            val peak = values.maxOf { abs(it) }
            if (!peak.isFinite() || peak <= 0f) return null
            val threshold = peak * EDGE_ACTIVE_RELATIVE_RMS
            val thresholdSquared = (threshold * threshold).toDouble()
            val activeWindows = ArrayList<Boolean>(
                1 + (values.size - EDGE_ACTIVE_WINDOW_SAMPLES) / EDGE_ACTIVE_HOP_SAMPLES,
            )
            var energy = 0.0
            for (index in 0 until EDGE_ACTIVE_WINDOW_SAMPLES) {
                val sample = values[index].toDouble()
                energy += sample * sample
            }
            var start = 0
            while (true) {
                activeWindows += energy / EDGE_ACTIVE_WINDOW_SAMPLES >= thresholdSquared
                val next = start + EDGE_ACTIVE_HOP_SAMPLES
                if (next + EDGE_ACTIVE_WINDOW_SAMPLES > values.size) break
                for (index in start until next) {
                    val sample = values[index].toDouble()
                    energy -= sample * sample
                }
                for (index in start + EDGE_ACTIVE_WINDOW_SAMPLES until next + EDGE_ACTIVE_WINDOW_SAMPLES) {
                    val sample = values[index].toDouble()
                    energy += sample * sample
                }
                start = next
            }

            var firstSustained = -1
            var lastSustained = -1
            var runStart = 0
            var runLength = 0
            activeWindows.forEachIndexed { index, active ->
                if (active) {
                    if (runLength == 0) runStart = index
                    runLength++
                    if (runLength >= EDGE_ACTIVE_SUSTAINED_WINDOWS) {
                        if (firstSustained < 0) firstSustained = runStart
                        lastSustained = index
                    }
                } else {
                    runLength = 0
                }
            }
            if (firstSustained < 0) return null
            return ActiveSampleRange(
                first = firstSustained * EDGE_ACTIVE_HOP_SAMPLES,
                last = min(
                    values.size,
                    lastSustained * EDGE_ACTIVE_HOP_SAMPLES + EDGE_ACTIVE_WINDOW_SAMPLES,
                ),
            )
        }

        private fun isRelativelyQuiet(values: FloatArray, start: Int, end: Int): Boolean {
            if (start !in 0..end || end > values.size || start == end) return false
            val peak = values.maxOf { abs(it) }
            if (!peak.isFinite() || peak <= 0f) return true
            var energy = 0.0
            for (index in start until end) {
                val sample = values[index].toDouble()
                energy += sample * sample
            }
            val rms = kotlin.math.sqrt(energy / (end - start)).toFloat()
            return rms <= peak * EDGE_ACTIVE_RELATIVE_RMS
        }

        private data class BoundaryCandidate(
            val end: Int,
            val tokens: Int,
            val tier: Int,
            val speechSafe: Boolean,
        )

        private data class SourceBoundaryCandidate(
            val end: Int,
            val tokens: Int,
            val tier: Int,
            val clauseStart: Boolean,
        )

        // A continuation render should not begin because the opening was cut immediately after
        // a high-confidence English connector. The planner operates on the already-generated IPA
        // so the full-sentence G2P result remains intact; this small set covers only unambiguous
        // dangling forms and falls back to any word boundary if no safer choice exists.
        private val unsafeOpeningTailPhonemes = setOf(
            "ɐ", "ɐn", "ðɐ", "ðə", "ʌv", "tə", "tʊ", "ɪn", "ɐt", "æz", "baɪ",
            "fɔɹ", "fɚ", "frʌm", "wɪð", "ænd", "ɔɹ", "bʌt", "ɪz", "ɑɹ",
        )

        // These are intentionally conservative syntax signals, not a pretend parser. They
        // prevent the common visibly broken stream cuts ("the | boat", "were | racing",
        // "because | it rained", "example | of") while letting a finished verb phrase lead a
        // following relative clause ("were racing | which caused...").
        private val sourceUnsafeHeadWords = setOf(
            "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
            "its", "our", "their", "some", "any", "each", "every", "either", "neither", "another",
            "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
            "have", "has", "had", "can", "could", "will", "would", "shall", "should", "may",
            "might", "must", "and", "or", "but", "nor", "yet", "so", "because", "although",
            "if", "when", "while", "unless", "since", "than", "as", "that", "which", "who",
            "whom", "whose", "of", "to", "for", "from", "with", "in", "on", "at", "by",
            "about", "into", "onto", "over", "under", "between", "through", "without", "within",
        )
        private val sourcePhraseContinuationStarts = setOf(
            "of", "to", "for", "from", "with", "in", "on", "at", "by", "about", "into", "onto",
            "over", "under", "between", "through", "without", "within", "as",
        )
        private val sourceClauseStarts = setOf("which", "who", "whom", "whose", "that", "because", "although", "while", "when", "if")
        private val sourceWordPattern = Regex("[A-Za-z]+(?:['\u2019][A-Za-z]+)?")

        private val textBoundaryPattern =
            Regex(
                "[.!?\\u2026]+(?:[\\\"')}\\u2019\\u201d\\u00bb]|\\])*(?:\\s+|\\z)|" +
                    "[,;:]+(?:[\\\"')}\\u2019\\u201d\\u00bb]|\\])*(?:\\s+|\\z)|" +
                    "[\\u2013\\u2014]+\\s*|\\s+",
            )

        private fun sourceWordBefore(input: String, end: Int): String? =
            sourceWordPattern.findAll(input.substring(0, end)).lastOrNull()?.value?.lowercase(Locale.US)

        private fun sourceWordAfter(input: String, start: Int): String? =
            sourceWordPattern.find(input, start)?.value?.lowercase(Locale.US)

        private fun preferredSourceTextBoundary(
            input: String,
            targetTokens: Int,
            hardTokens: Int,
            minimumTailTokens: Int,
            countTokens: (String) -> Int,
        ): Int? {
            val candidates = textBoundaryPattern.findAll(input).mapNotNull { match ->
                val end = match.range.last + 1
                if (end >= input.length) return@mapNotNull null
                val headTokens = countTokens(input.substring(0, end))
                val tailTokens = countTokens(input.substring(end))
                if (headTokens !in 1..hardTokens || tailTokens < minimumTailTokens) return@mapNotNull null
                val headWord = sourceWordBefore(input, end) ?: return@mapNotNull null
                val tailWord = sourceWordAfter(input, end)
                if (headWord in sourceUnsafeHeadWords || tailWord in sourcePhraseContinuationStarts) {
                    return@mapNotNull null
                }
                val tier = when {
                    match.value.any { it in ".!?" || it == '\u2026' } -> 3
                    match.value.any { it in ",;:" || it == '\u2013' || it == '\u2014' } -> 2
                    tailWord in sourceClauseStarts -> 1
                    else -> 0
                }
                SourceBoundaryCandidate(end, headTokens, tier, tailWord in sourceClauseStarts)
            }.toList()
            if (candidates.isEmpty()) return null
            val floor = max(1, targetTokens * 2 / 3)
            val comparator = compareByDescending<SourceBoundaryCandidate> { it.tier }
                .thenByDescending { it.clauseStart }
                .thenBy { abs(it.tokens - targetTokens) }
                .thenByDescending { it.tokens }
            return candidates.filter { it.tokens >= floor }.minWithOrNull(comparator)?.end
                ?: candidates.minWithOrNull(comparator)?.end
        }

        private fun preferredTextBoundary(
            input: String,
            targetTokens: Int,
            hardTokens: Int,
            minimumTailTokens: Int,
            countTokens: (String) -> Int,
        ): Int? {
            val candidates = textBoundaryPattern.findAll(input).mapNotNull { match ->
                val end = match.range.last + 1
                if (end >= input.length) return@mapNotNull null
                val headTokens = countTokens(input.substring(0, end))
                val tailTokens = countTokens(input.substring(end))
                if (headTokens !in 1..hardTokens || tailTokens < minimumTailTokens) {
                    return@mapNotNull null
                }
                val tier = when {
                    match.value.any { it in ".!?" || it == '\u2026' } -> 2
                    match.value.any { it in ",;:" || it == '\u2013' || it == '\u2014' } -> 1
                    else -> 0
                }
                val tail = input.substring(0, end).trimEnd().substringAfterLast(' ')
                BoundaryCandidate(
                    end = end,
                    tokens = headTokens,
                    tier = tier,
                    speechSafe = tier > 0 || tail !in unsafeOpeningTailPhonemes,
                )
            }.toList()
            if (candidates.isEmpty()) return null

            val punctuationFloor = max(1, targetTokens * 2 / 3)
            val comparator = compareBy<BoundaryCandidate> { abs(it.tokens - targetTokens) }
                .thenByDescending { it.tokens }
            for (tier in 2 downTo 1) {
                candidates.filter { it.tier == tier && it.tokens >= punctuationFloor }
                    .minWithOrNull(comparator)
                    ?.let { return it.end }
            }
            val wordBoundaries = candidates.filter { it.tier == 0 }
            val safeWordBoundaries = wordBoundaries.filter { it.speechSafe }
            // Prefer a substantial safe opening at or before the target. This keeps the starter
            // bucket bounded instead of crossing the target merely because the next word ending
            // is a few tokens closer.
            safeWordBoundaries.filter { it.tokens in punctuationFloor..targetTokens }
                .maxByOrNull { it.tokens }
                ?.let { return it.end }
            return safeWordBoundaries.minWithOrNull(comparator)?.end
                ?: wordBoundaries.minWithOrNull(comparator)?.end
                ?: candidates.minWithOrNull(comparator)?.end
        }

        private fun fallbackTokenBoundary(
            input: String,
            targetTokens: Int,
            minimumTailTokens: Int,
            countTokens: (String) -> Int,
        ): Int? {
            var low = 1
            var high = input.length - 1
            var best = -1
            while (low <= high) {
                val middle = (low + high) ushr 1
                val tokens = countTokens(input.substring(0, middle))
                if (tokens <= targetTokens) {
                    if (tokens > 0) best = middle
                    low = middle + 1
                } else {
                    high = middle - 1
                }
            }
            if (best <= 0) return null
            while (best > 1 && Character.isLowSurrogate(input[best])) best--
            return best.takeIf {
                countTokens(input.substring(0, it)) > 0 &&
                    countTokens(input.substring(it)) >= minimumTailTokens
            }
        }

        private fun leadingSamplesFor(seam: WaveformSeam): Int? = when (seam) {
            WaveformSeam.NONE -> null
            WaveformSeam.REQUEST_BOUNDARY -> REQUEST_BOUNDARY_LEADING_SAMPLES
            else -> CONTINUATION_LEADING_SAMPLES
        }

        private fun trailingSamplesFor(seam: WaveformSeam): Int? = when (seam) {
            WaveformSeam.NONE -> null
            WaveformSeam.REQUEST_BOUNDARY -> REQUEST_BOUNDARY_TRAILING_SAMPLES
            WaveformSeam.CONTINUATION -> CONTINUATION_TRAILING_SAMPLES
            WaveformSeam.COMMA -> COMMA_TRAILING_SAMPLES
            WaveformSeam.SEMICOLON -> SEMICOLON_TRAILING_SAMPLES
            WaveformSeam.COLON -> COLON_TRAILING_SAMPLES
            WaveformSeam.PERIOD -> PERIOD_TRAILING_SAMPLES
            WaveformSeam.QUESTION -> QUESTION_TRAILING_SAMPLES
        }

        internal fun requestBoundaryEdgeSamplesForTesting(): Pair<Int, Int> =
            REQUEST_BOUNDARY_LEADING_SAMPLES to REQUEST_BOUNDARY_TRAILING_SAMPLES

        internal fun waveformSeamAfter(input: String): WaveformSeam {
            val boundary = input.trimEnd().dropLastWhile {
                it in "\"')]}" || it == '\u2019' || it == '\u201d' || it == '\u00bb'
            }.lastOrNull()
            return when (boundary) {
                ',' -> WaveformSeam.COMMA
                ';', '\u2013', '\u2014' -> WaveformSeam.SEMICOLON
                ':' -> WaveformSeam.COLON
                '.', '!', '\u2026' -> WaveformSeam.PERIOD
                '?' -> WaveformSeam.QUESTION
                else -> WaveformSeam.CONTINUATION
            }
        }

        internal fun qnnSessionSourceForTesting(): String = QNN_FALLBACK_SESSION_SOURCE

        internal fun qnnPerformancePolicyForTesting(): String = QNN_PERFORMANCE_POLICY

        internal fun qnnPrecisionPolicyForTesting(): String = QNN_PRECISION_POLICY

        internal fun qnnAssignmentPolicyForTesting(): String = "session.disable_cpu_ep_fallback=1"

        internal fun qnnPcmCacheIdentity(contextFingerprint: String, retryGeneration: Long): String =
            "QNN_HTP:$contextFingerprint:retry=$retryGeneration"

        internal fun soleCacheBackend(backends: Set<String>): String? =
            backends.singleOrNull()

        internal fun runtimeDiagnostics(context: Context): RuntimeDiagnosticsSnapshot {
            val preferences = context.getSharedPreferences(RUNTIME_PREFERENCES, Context.MODE_PRIVATE)
            return RuntimeDiagnosticsSnapshot(
                backend = preferences.getString(DIAGNOSTIC_BACKEND_KEY, "") ?: "",
                bucket = preferences.getInt(DIAGNOSTIC_BUCKET_KEY, -1).takeIf { it > 0 },
                generatorRtf = preferences.getString(DIAGNOSTIC_RTF_KEY, "") ?: "",
                contextSource = preferences.getString(DIAGNOSTIC_CONTEXT_SOURCE_KEY, "") ?: "",
                contextHashPrefix = preferences.getString(DIAGNOSTIC_CONTEXT_HASH_KEY, "") ?: "",
                failureReason = preferences.getString(DIAGNOSTIC_FAILURE_KEY, "") ?: "",
                timestampUtcMillis = preferences.getLong(DIAGNOSTIC_TIMESTAMP_KEY, 0L),
                qnnDisabled = preferences.getBoolean(QNN_DISABLED_KEY, false),
                nnapiDisabled = preferences.getBoolean(NNAPI_DISABLED_KEY, false),
                qnnAotIncluded = BuildConfig.KOKORO_QNN_AOT_INCLUDED,
                b192ContextHashPrefix = BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256.take(12),
                b256ContextHashPrefix = BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256.take(12),
                b320ContextHashPrefix = BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256.take(12),
                b384ContextHashPrefix = BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256.take(12),
                qnnRetryGeneration = preferences.getLong(QNN_RETRY_GENERATION_KEY, 0L),
            )
        }

        /** Clears only this AOT generation's QNN fallback and failure note. */
        internal fun requestQnnRetry(context: Context): Boolean {
            val preferences = context.getSharedPreferences(RUNTIME_PREFERENCES, Context.MODE_PRIVATE)
            val changed = preferences.getBoolean(QNN_DISABLED_KEY, false) ||
                !preferences.getString(DIAGNOSTIC_FAILURE_KEY, "").isNullOrBlank()
            val currentGeneration = preferences.getLong(QNN_RETRY_GENERATION_KEY, 0L)
            val nextGeneration = if (currentGeneration == Long.MAX_VALUE) 0L else currentGeneration + 1L
            preferences.edit()
                .remove(QNN_DISABLED_KEY)
                .remove(DIAGNOSTIC_FAILURE_KEY)
                .putLong(QNN_RETRY_GENERATION_KEY, nextGeneration)
                .apply()
            return changed
        }

        internal fun qnnRetryKeysForTesting(): Set<String> =
            setOf(QNN_DISABLED_KEY, DIAGNOSTIC_FAILURE_KEY, QNN_RETRY_GENERATION_KEY)

        internal fun isTargetQnnDevice(
            sdk: Int,
            socModel: String?,
            deviceModel: String?,
            supportedAbis: Collection<String>,
        ): Boolean =
            sdk >= 31 &&
                socModel.equals(TARGET_ANDROID_SOC, ignoreCase = true) &&
                deviceModel.orEmpty().uppercase(Locale.US).startsWith("SM-S928") &&
                supportedAbis.contains("arm64-v8a")

        internal fun pcmHeadroomGain(values: FloatArray): Float {
            if (values.isEmpty()) throw InvalidGeneratorAudioException("Generator produced empty audio")
            if (!values.all { it.isFinite() }) {
                throw InvalidGeneratorAudioException("Generator produced non-finite audio")
            }
            var minimum = Float.POSITIVE_INFINITY
            var maximum = Float.NEGATIVE_INFINITY
            var maxAbs = 0f
            values.forEach { value ->
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                maxAbs = max(maxAbs, abs(value))
            }
            if (maximum - minimum < MIN_PCM_DYNAMIC_RANGE || maxAbs < MIN_PCM_DYNAMIC_RANGE) {
                throw InvalidGeneratorAudioException("Generator produced near-static audio")
            }
            return min(1f, PCM_HEADROOM / maxAbs)
        }

        /**
         * Encodes generator audio without letting a single iSTFT boundary impulse turn down an
         * entire continuation. The old max-peak gain made this exact failure mode possible: one
         * click at the start of a window could be many times larger than its speech, so all later
         * samples were attenuated by the same tiny factor.
         *
         * Sustained over-range audio still receives ordinary global headroom. Only the highest
         * 0.5 percent of absolute samples are treated as possible transients, and those residual
         * peaks pass through a continuous soft limiter instead of a hard discontinuity.
         */
        internal fun encodePcm16WithDiagnostics(values: FloatArray): PcmEncodingResult {
            pcmHeadroomGain(values) // Preserve the existing finite/dynamic-range validation.
            val magnitudes = FloatArray(values.size) { index -> abs(values[index]) }
            var absolutePeak = 0f
            magnitudes.forEach { absolutePeak = max(absolutePeak, it) }
            val robustIndex = ((magnitudes.lastIndex.toLong() * PCM_ROBUST_PEAK_PERCENTILE_NUMERATOR) /
                PCM_ROBUST_PEAK_PERCENTILE_DENOMINATOR).toInt()
            val robustPeak = selectKth(magnitudes, robustIndex)
            val gain = min(1f, PCM_HEADROOM / max(robustPeak, MIN_PCM_DYNAMIC_RANGE))
            val limiterNeeded = absolutePeak * gain > PCM_HEADROOM
            val out = ByteArray(values.size * 2)
            var index = 0
            var limitedSamples = 0
            for (value in values) {
                val scaled = value * gain
                val magnitude = abs(scaled)
                val limited = if (!limiterNeeded || magnitude <= PCM_LIMITER_KNEE) {
                    scaled
                } else {
                    limitedSamples++
                    val compressed = PCM_LIMITER_KNEE +
                        (PCM_HEADROOM - PCM_LIMITER_KNEE) *
                        tanh(((magnitude - PCM_LIMITER_KNEE) /
                            (PCM_HEADROOM - PCM_LIMITER_KNEE)).toDouble()).toFloat()
                    if (scaled < 0f) -compressed else compressed
                }
                val sample = (limited.coerceIn(-PCM_HEADROOM, PCM_HEADROOM) * 32767f).toInt().toShort()
                out[index++] = (sample.toInt() and 0xff).toByte()
                out[index++] = (sample.toInt() shr 8).toByte()
            }
            return PcmEncodingResult(out, gain, robustPeak, absolutePeak, limitedSamples)
        }

        internal fun encodePcm16(values: FloatArray): ByteArray = encodePcm16WithDiagnostics(values).pcm

        /** In-place expected-linear-time selection; avoids sorting every generated waveform. */
        private fun selectKth(values: FloatArray, requestedIndex: Int): Float {
            require(values.isNotEmpty())
            val target = requestedIndex.coerceIn(0, values.lastIndex)
            var left = 0
            var right = values.lastIndex
            while (left < right) {
                val pivot = values[left + (right - left) / 2]
                var low = left
                var high = right
                while (low <= high) {
                    while (values[low] < pivot) low++
                    while (values[high] > pivot) high--
                    if (low <= high) {
                        val swap = values[low]
                        values[low] = values[high]
                        values[high] = swap
                        low++
                        high--
                    }
                }
                when {
                    target <= high -> right = high
                    target >= low -> left = low
                    else -> return values[target]
                }
            }
            return values[target]
        }
    }

    private data class CacheKey(
        val phonemes: String,
        val voice: String,
        val speedBits: Int,
        val backendIdentity: String,
        val leadingSeam: WaveformSeam,
        val trailingSeam: WaveformSeam,
    )

    private data class PlannedChunk(
        val phonemes: String,
        val leadingSeam: WaveformSeam,
        val trailingSeam: WaveformSeam,
        // Post-duration planning has already run the CPU front. Keeping that exact result avoids
        // a second duration pass before the selected static QNN context runs.
        val front: GeneratorInputs? = null,
        // A globally conditioned frame slice includes bounded left/right generator context.
        // Only this core range is emitted, so every full-sentence frame appears exactly once.
        val coreStartFrame: Int = 0,
        val coreFrames: Int? = null,
        // These are shared global-timeline context, not added pause or duplicated cadence.
        val leadingOverlapFrames: Int = 0,
        val trailingOverlapFrames: Int = 0,
        val globallyConditioned: Boolean = false,
        val cacheIdentity: String = phonemes,
    )

    private data class FrontedSpan(
        val phonemes: String,
        val front: GeneratorInputs?,
    )

    private data class PhonemeSplit(
        val head: String,
        val tail: String,
        val seam: WaveformSeam,
    )

    private data class GeneratorInputs(
        val conditioning: FloatArray,
        val prosody: FloatArray,
        val decoder: FloatArray,
        val frames: Int,
        val tokenDurations: LongArray? = null,
    ) {
        init {
            require(conditioning.size == 128) { "Unexpected generator conditioning size ${conditioning.size}" }
            require(prosody.size == frames) { "Unexpected generator prosody shape ${prosody.size} for T=$frames" }
            require(decoder.size == 512 * frames) {
                "Unexpected generator decoder shape ${decoder.size} for T=$frames"
            }
            tokenDurations?.let { durations ->
                require(durations.isNotEmpty() && durations.all { it > 0L }) {
                    "Invalid global token-duration alignment"
                }
                require(durations.sum() * DURATION_UNIT_GENERATOR_FRAMES == frames.toLong()) {
                    "Token durations do not cover T=$frames"
                }
            }
        }
    }

    private data class PreparedQnnPrefix(
        val front: GeneratorInputs,
        val bucket: Int,
        val sourceSpectrum: FloatArray,
        val sourceWasResident: Boolean,
        val sourceSessionMs: Double,
        val sourceRunMs: Double,
    )

    /** ORT requires session options to outlive every session created from them. */
    private class SessionHolder(
        val session: OrtSession,
        private val options: OrtSession.SessionOptions,
        val qnnSource: String? = null,
        val qnnContextSha256: String? = null,
        val qnnBucket: Int? = null,
    ) : AutoCloseable {
        private var closed = false

        @Synchronized
        override fun close() {
            if (closed) return
            closed = true
            try {
                session.close()
            } finally {
                options.close()
            }
        }
    }

    private class InvalidGeneratorAudioException(message: String) : IllegalStateException(message)

    private data class SharedQnnArtifact(
        val asset: String,
        val sha256: String,
        val bytes: Long,
    )

    private data class SharedQnnEntry(
        val bucket: Int,
        val wrapper: SharedQnnArtifact,
    )

    private data class SharedQnnGroup(
        val binary: SharedQnnArtifact,
        val entries: List<SharedQnnEntry>,
    )

    private data class SourceSpectrumArtifact(
        val bucket: Int,
        val asset: String,
        val sha256: String,
        val bytes: Long,
    )

    private val environment = OrtEnvironment.getEnvironment()
    private val instanceId = UUID.randomUUID().toString()
    private val styles by lazy { VoiceStyleStore(context) }
    private val phonemizer by lazy { EnglishPhonemizer(context) }
    // The opening HTP run may overlap a continuation's CPU-only front/source
    // preparation. Track every active ORT run so cancellation still reaches
    // both sessions; HTP execution itself remains serialized by sessionUseLock.
    private val activeRuns = ConcurrentHashMap.newKeySet<OrtSession.RunOptions>()
    private val nnapiSlowRuns = AtomicInteger()
    private val sessionLock = Any()
    private val v1ModelCreationLock = Any()
    private val frontCreationLock = Any()
    private val cpuGeneratorCreationLock = Any()
    private val v1IstftCreationLock = Any()
    private val nnapiGeneratorCreationLock = Any()
    private val sessionUseLock = Any()
    private val qnnCreationLocks = ConcurrentHashMap<Int, Any>()
    private val v1QnnCreationLocks = ConcurrentHashMap<Int, Any>()
    private val sourceSpectrumCreationLocks = ConcurrentHashMap<Int, Any>()
    private val sharedQnnCreationLock = Any()
    private val modelMappingLocks = ConcurrentHashMap<String, Any>()
    private val performanceLock = Any()
    private val diagnosticsLock = Any()
    private val cacheLock = Any()
    private val mappedModels = mutableMapOf<String, ByteBuffer>()
    private val qnnContextCacheId by lazy { buildQnnContextCacheId() }
    private val qnnGeneratorSessions = LinkedHashMap<Int, SessionHolder>(4, 0.75f, true)
    private val v1QnnAcousticSessions = LinkedHashMap<Int, SessionHolder>(4, 0.75f, true)
    private val sourceSpectrumSessions = LinkedHashMap<Int, SessionHolder>(16, 0.75f, false)
    private val sharedV1QnnSessions = LinkedHashMap<Int, SessionHolder>(16, 0.75f, false)
    private val pcmCache = LinkedHashMap<CacheKey, ByteArray>(16, 0.75f, true)
    private var frontSession: SessionHolder? = null
    private var v1ModelSession: SessionHolder? = null
    private var cpuGeneratorSession: SessionHolder? = null
    private var v1IstftSession: SessionHolder? = null
    private var nnapiGeneratorSession: SessionHolder? = null
    private var qnnEpDevices: List<OrtEpDevice>? = null
    @Volatile private var sharedQnnDisabledForProcess = false
    private val sharedQnnGroups by lazy {
        if (!BuildConfig.KOKORO_QNN_SHARED_INCLUDED) {
            emptyList()
        } else {
            listOf(
                BuildConfig.KOKORO_QNN_SHARED_ACOUSTIC_MANIFEST,
                BuildConfig.KOKORO_QNN_SHARED_MID_MANIFEST,
                BuildConfig.KOKORO_QNN_SHARED_LARGE_MANIFEST,
            ).map(::parseSharedQnnManifest)
        }
    }
    private val sourceSpectrumArtifacts by lazy {
        parseSourceSpectrumManifest(BuildConfig.KOKORO_QNN_SOURCE_SPECTRUM_MANIFEST)
    }
    private var pcmCacheSize = 0

    @Volatile private var lastGeneratorBackend = InferenceBackend.CPU
    @Volatile private var maxGeneratorRtfSinceReset = Double.NaN
    private val generatorBackendsSinceReset = linkedSetOf<InferenceBackend>()
    private val currentInferenceBackends = linkedSetOf<InferenceBackend>()
    private val qnnContextSourcesSinceReset = linkedMapOf<Int, String>()
    private val qnnContextHashesSinceReset = linkedMapOf<Int, String>()
    private var diagnosticBackend = ""
    private var diagnosticBucket: Int? = null
    private var diagnosticRtf = ""
    private var diagnosticContextSource = ""
    private var diagnosticContextHashPrefix = ""
    private var diagnosticFailureReason = ""
    private var diagnosticsDirty = false
    @Volatile private var cancelRequested = false
    @Volatile private var closed = false

    /** Initializes the current native Misaki frontend and v1.0 ONNX graph. */
    fun prepare() {
        check(!closed) { "Kokoro synthesizer is closed" }
        phonemizer.phonemize("Ready.", Locale.US)
        KokoroTokenizer.tokenize(context, "ready")
        styles.style(VoiceCatalog.default.id, 5)
        if (shouldUseQnn()) {
            // Do not eagerly open the 92 MiB fallback model on an HTP-capable
            // device. The progressive QNN buckets need only the front graph,
            // exact CPU source-spectrum prefix, and CPU iSTFT suffix. The
            // monolithic q8 session is created lazily if a duration has no
            // qualified static acoustic bucket or HTP later fails.
            // Register the plugin and enumerate HTP before creating any ORT
            // session. This matches Qualcomm's Android integration order and
            // avoids freezing the environment's provider-device inventory
            // after the CPU front-end session has already been constructed.
            qnnDevices()
            cpuFrontSession()
            // This runs on the service's background prepare thread. Prefer the
            // external shared-weight groups only when a supported build ships them.
            // Any shared-context load failure is isolated to this process and
            // immediately falls back to the proven embedded per-bucket path.
            if (BuildConfig.KOKORO_QNN_SHARED_INCLUDED) {
                try {
                    sharedQnnGroups.forEach { group -> sharedV1QnnSession(group.entries.first().bucket) }
                } catch (problem: Exception) {
                    if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
                    disableSharedQnnForProcess("prewarm failure", problem)
                }
            }
            if (sharedQnnDisabledForProcess || !BuildConfig.KOKORO_QNN_SHARED_INCLUDED) {
                // Keep the complete normal streaming set resident.  In particular, B208 must
                // not be displaced: it is the safe continuation for a near-exact B192 core and
                // loading it after B128 playback begins creates an audible media underrun.
                val lastBucket = runtimePreferences().getInt(DIAGNOSTIC_BUCKET_KEY, -1).takeIf { it > 0 }
                val prewarmBuckets = qnnPrewarmBuckets(lastBucket)
                Log.i(TAG, "Prewarming QNN buckets ${prewarmBuckets.joinToString()}")
                prewarmBuckets.forEach { bucket ->
                    cpuSourceSpectrumSession(bucket)
                    legacyV1QnnAcousticSession(bucket)
                }
            }
        } else {
            v1ModelSession()
        }
    }

    private fun v1ModelSession(): SessionHolder = synchronized(v1ModelCreationLock) {
        val existing = synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            v1ModelSession
        }
        if (existing != null) return@synchronized existing
        val created = createCpuSession(BuildConfig.KOKORO_MODEL_ASSET)
        synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during v1.0 model creation")
            }
            v1ModelSession?.let { installed ->
                created.close()
                return@synchronized installed
            }
            v1ModelSession = created
            Log.i(TAG, "Using Kokoro v1.0 q8 single-graph CPU runtime")
            created
        }
    }

    private fun cpuFrontSession(): SessionHolder = synchronized(frontCreationLock) {
        val existing = synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            frontSession
        }
        if (existing != null) return@synchronized existing

        // CPU graph optimization is too expensive to hold the lifecycle lock.
        val created = createCpuSession(BuildConfig.KOKORO_FRONT_MODEL_ASSET)
        synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during CPU front-end creation")
            }
            frontSession?.let { existing ->
                created.close()
                return@synchronized existing
            }
            frontSession = created
            Log.i(TAG, "Using optimized CPU front-end")
            created
        }
    }

    private fun cpuGeneratorSession(): SessionHolder = synchronized(cpuGeneratorCreationLock) {
        val existing = synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            cpuGeneratorSession
        }
        if (existing != null) return@synchronized existing

        val created = createCpuSession(BuildConfig.KOKORO_GENERATOR_MODEL_ASSET)
        synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during CPU generator creation")
            }
            cpuGeneratorSession?.let { installed ->
                created.close()
                return@synchronized installed
            }
            cpuGeneratorSession = created
            Log.i(TAG, "Using optimized CPU waveform generator")
            created
        }
    }

    private fun v1IstftSession(): SessionHolder = synchronized(v1IstftCreationLock) {
        val existing = synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            v1IstftSession
        }
        if (existing != null) return@synchronized existing
        val created = createCpuSession(BuildConfig.KOKORO_ISTFT_MODEL_ASSET)
        synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during v1 iSTFT creation")
            }
            v1IstftSession?.let { installed ->
                created.close()
                return@synchronized installed
            }
            v1IstftSession = created
            Log.i(TAG, "Using CPU v1 iSTFT suffix after HTP acoustic inference")
            created
        }
    }

    private fun nnapiGeneratorSession(): SessionHolder = synchronized(nnapiGeneratorCreationLock) {
        val existing = synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            nnapiGeneratorSession
        }
        if (existing != null) return@synchronized existing

        val created = createNnapiGeneratorSession()
        synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during NNAPI generator creation")
            }
            nnapiGeneratorSession?.let { installed ->
                created.close()
                return@synchronized installed
            }
            nnapiGeneratorSession = created
            Log.i(TAG, "Using Samsung NNAPI waveform generator")
            created
        }
    }

    private fun createNnapiGeneratorSession(): SessionHolder {
        val options = baseSessionOptions()
        try {
            options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
        } catch (failure: Throwable) {
            closeAfterFailure(options, failure)
            throw failure
        }
        return createOwnedSession(options) {
            environment.createSession(modelBuffer(BuildConfig.KOKORO_GENERATOR_MODEL_ASSET), options)
        }
    }

    private fun createCpuSession(assetName: String): SessionHolder {
        val options = baseSessionOptions()
        return createOwnedSession(options) {
            environment.createSession(modelBuffer(assetName), options)
        }
    }

    private inline fun createOwnedSession(
        options: OrtSession.SessionOptions,
        qnnSource: String? = null,
        qnnContextSha256: String? = null,
        qnnBucket: Int? = null,
        factory: () -> OrtSession,
    ): SessionHolder = try {
        SessionHolder(factory(), options, qnnSource, qnnContextSha256, qnnBucket)
    } catch (failure: Throwable) {
        closeAfterFailure(options, failure)
        throw failure
    }

    private fun closeAfterFailure(closeable: AutoCloseable, failure: Throwable) {
        try {
            closeable.close()
        } catch (closeFailure: Throwable) {
            failure.addSuppressed(closeFailure)
        }
    }

    private fun baseSessionOptions(): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        return try {
            options.apply {
                val threads = min(4, max(2, Runtime.getRuntime().availableProcessors() / 2))
                setIntraOpNumThreads(threads)
                setInterOpNumThreads(1)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setMemoryPatternOptimization(true)
                setCPUArenaAllocator(true)
            }
        } catch (failure: Throwable) {
            closeAfterFailure(options, failure)
            throw failure
        }
    }

    private fun shouldUseQnn(): Boolean {
        if (!BuildConfig.QNN_EP_INCLUDED || !BuildConfig.KOKORO_QNN_AOT_INCLUDED ||
            !Build.SUPPORTED_ABIS.contains("arm64-v8a")
        ) return false
        return when (preference) {
            BackendPreference.QNN_HTP -> true
            BackendPreference.AUTO -> isTargetQnnDevice() && !runtimePreferences().getBoolean(QNN_DISABLED_KEY, false)
            BackendPreference.CPU, BackendPreference.NNAPI -> false
        }
    }

    private fun isTargetQnnDevice(): Boolean = isTargetQnnDevice(
        sdk = Build.VERSION.SDK_INT,
        socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else null,
        deviceModel = Build.MODEL,
        supportedAbis = Build.SUPPORTED_ABIS.asList(),
    )

    private fun shouldUseNnapi(): Boolean {
        if (Build.VERSION.SDK_INT < 29 || runtimePreferences().getBoolean(NNAPI_DISABLED_KEY, false)) return false
        return preference == BackendPreference.NNAPI
    }

    private fun qnnGeneratorSession(bucket: Int): SessionHolder {
        require(bucket == QNN_B256_FRAMES || bucket == QNN_B384_FRAMES) {
            "No packaged QNN AOT context for T=$bucket"
        }
        val creationLock = qnnCreationLocks.computeIfAbsent(bucket) { Any() }
        return synchronized(creationLock) { qnnGeneratorSessionSingleFlight(bucket) }
    }

    /** Chooses shared-weight AOT first and the prior embedded AOT on any shared
     * setup failure. Neither route can compile a source graph on the phone. */
    private fun v1QnnAcousticSession(bucket: Int): SessionHolder {
        if (BuildConfig.KOKORO_QNN_SHARED_INCLUDED && !sharedQnnDisabledForProcess) {
            try {
                return sharedV1QnnSession(bucket)
            } catch (problem: Exception) {
                if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
                disableSharedQnnForProcess("session load failure for T=$bucket", problem)
            }
        }
        return legacyV1QnnAcousticSession(bucket)
    }

    private fun sharedV1QnnSession(bucket: Int): SessionHolder {
        check(BuildConfig.KOKORO_QNN_SHARED_INCLUDED && !sharedQnnDisabledForProcess) {
            "Shared QNN contexts are unavailable"
        }
        val group = sharedQnnGroups.singleOrNull { candidate -> candidate.entries.any { it.bucket == bucket } }
            ?: error("No shared QNN group for T=$bucket")
        return synchronized(sharedQnnCreationLock) sharedSessionCreation@{
            val existing = synchronized(sessionLock) {
                check(!closed) { "Kokoro synthesizer is closed" }
                check(!sharedQnnDisabledForProcess) { "Shared QNN contexts were disabled for this process" }
                sharedV1QnnSessions[bucket]
            }
            if (existing != null) return@sharedSessionCreation existing

            val directory = stageSharedQnnGroup(group)
            val created = linkedMapOf<Int, SessionHolder>()
            try {
                group.entries.forEach { entry ->
                    val options = baseSessionOptions()
                    try {
                        options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                        options.addConfigEntry("ep.share_ep_contexts", "1")
                        val htpBackend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                        check(htpBackend.isFile) { "Packaged QNN HTP backend is missing: $htpBackend" }
                        options.addExecutionProvider(
                            qnnDevices(),
                            qnnProviderOptions(
                                htpBackend.absolutePath,
                                vtcmMb = 8,
                                sharedPowerTuning = true,
                            ),
                        )
                        val wrapperFile = File(directory, entry.wrapper.asset)
                        val holder = createOwnedSession(
                            options = options,
                            qnnSource = QNN_SESSION_SOURCE,
                            qnnContextSha256 = group.binary.sha256,
                            qnnBucket = entry.bucket,
                        ) { environment.createSession(wrapperFile.absolutePath, options) }
                        created[entry.bucket] = validateSharedQnnContextContract(holder, entry.bucket)
                    } catch (failure: Throwable) {
                        if (!created.values.any { it.qnnBucket == entry.bucket }) {
                            try {
                                options.close()
                            } catch (_: Throwable) {
                                // createOwnedSession already owns normal failure cleanup.
                            }
                        }
                        throw failure
                    }
                }
                synchronized(sessionLock) {
                    check(!closed) { "Kokoro synthesizer closed during shared QNN session creation" }
                    group.entries.forEach { entry ->
                        sharedV1QnnSessions[entry.bucket]?.let { existing ->
                            created.remove(entry.bucket)?.close()
                            created[entry.bucket] = existing
                        }
                    }
                    created.forEach { (createdBucket, holder) ->
                        if (!sharedV1QnnSessions.containsKey(createdBucket)) {
                            sharedV1QnnSessions[createdBucket] = holder
                        }
                    }
                    Log.i(TAG, "Using shared QNN weight group ${group.binary.sha256.take(12)} " +
                        "for buckets ${group.entries.map { it.bucket }}")
                    sharedV1QnnSessions.getValue(bucket)
                }
            } catch (failure: Throwable) {
                created.values.distinct().forEach { holder ->
                    try {
                        holder.close()
                    } catch (closeFailure: Throwable) {
                        failure.addSuppressed(closeFailure)
                    }
                }
                throw failure
            }
        }
    }

    /** Loads the proven embedded per-bucket EPContext fallback. */
    private fun legacyV1QnnAcousticSession(bucket: Int): SessionHolder {
        require(bucket == QNN_V1_B64_FRAMES || bucket == QNN_V1_B96_FRAMES ||
            bucket == QNN_V1_B128_FRAMES || bucket == QNN_V1_B192_FRAMES ||
            bucket == QNN_V1_B208_FRAMES || bucket == QNN_V1_B224_FRAMES ||
            bucket == QNN_V1_B256_FRAMES ||
            bucket == QNN_V1_B320_FRAMES || bucket == QNN_V1_B384_FRAMES ||
            bucket == QNN_V1_B512_FRAMES || bucket == QNN_V1_B640_FRAMES) {
            "No packaged v1 QNN acoustic bucket for T=$bucket"
        }
        val creationLock = v1QnnCreationLocks.computeIfAbsent(bucket) { Any() }
        return synchronized(creationLock) qnnSessionCreation@{
            val existing = synchronized(sessionLock) {
                check(!closed) { "Kokoro synthesizer is closed" }
                v1QnnAcousticSessions[bucket]
            }
            if (existing != null) return@qnnSessionCreation existing
            val assetAndHash = when (bucket) {
                QNN_V1_B64_FRAMES -> BuildConfig.KOKORO_QNN_B64_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B64_CONTEXT_SHA256
                QNN_V1_B96_FRAMES -> BuildConfig.KOKORO_QNN_B96_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B96_CONTEXT_SHA256
                QNN_V1_B128_FRAMES -> BuildConfig.KOKORO_QNN_B128_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B128_CONTEXT_SHA256
                QNN_V1_B192_FRAMES -> BuildConfig.KOKORO_QNN_B192_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256
                QNN_V1_B208_FRAMES -> BuildConfig.KOKORO_QNN_B208_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B208_CONTEXT_SHA256
                QNN_V1_B224_FRAMES -> BuildConfig.KOKORO_QNN_B224_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B224_CONTEXT_SHA256
                QNN_V1_B256_FRAMES -> BuildConfig.KOKORO_QNN_B256_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256
                QNN_V1_B320_FRAMES -> BuildConfig.KOKORO_QNN_B320_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256
                QNN_V1_B384_FRAMES -> BuildConfig.KOKORO_QNN_B384_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256
                QNN_V1_B512_FRAMES -> BuildConfig.KOKORO_QNN_B512_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B512_CONTEXT_SHA256
                else -> BuildConfig.KOKORO_QNN_B640_CONTEXT_ASSET to BuildConfig.KOKORO_QNN_B640_CONTEXT_SHA256
            }
            check(assetAndHash.first.isNotBlank() && assetAndHash.second.matches(Regex("[0-9a-f]{64}"))) {
                "Invalid packaged v1 QNN context identity for T=$bucket"
            }
            val options = baseSessionOptions()
            try {
                options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                val htpBackend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                check(htpBackend.isFile) { "Packaged QNN HTP backend is missing: $htpBackend" }
                options.addExecutionProvider(
                    qnnDevices(),
                    qnnProviderOptions(
                        htpBackend.absolutePath,
                        vtcmMb = if (bucket == QNN_V1_B192_FRAMES || bucket == QNN_V1_B512_FRAMES || bucket == QNN_V1_B640_FRAMES) 8 else 0,
                    ),
                )
                val created = validateRepairedV1QnnContextContract(createOwnedSession(
                    options = options,
                    qnnSource = QNN_FALLBACK_SESSION_SOURCE,
                    qnnContextSha256 = assetAndHash.second,
                    qnnBucket = bucket,
                ) { environment.createSession(modelBuffer(assetAndHash.first), options) }, bucket)
                synchronized(sessionLock) {
                    if (closed) {
                        created.close()
                        error("Kokoro synthesizer closed during v1 QNN session creation")
                    }
                    v1QnnAcousticSessions[bucket]?.let { existing ->
                        created.close()
                        return@synchronized existing
                    }
                    v1QnnAcousticSessions[bucket] = created
                    while (v1QnnAcousticSessions.size > MAX_QNN_BUCKET_SESSIONS) {
                        val eldest = v1QnnAcousticSessions.entries.iterator().next()
                        v1QnnAcousticSessions.remove(eldest.key)
                        eldest.value.close()
                    }
                    Log.i(TAG, "Using packaged v1 QNN acoustic context bucket T=$bucket")
                    created
                }
            } catch (failure: Throwable) {
                closeAfterFailure(options, failure)
                throw failure
            }
        }
    }

    private fun validateRepairedV1QnnContextContract(holder: SessionHolder, bucket: Int): SessionHolder {
        try {
            val expectedInputs = linkedMapOf(
                FRONT_CONDITIONING to longArrayOf(1, 128),
                FRONT_DECODER to longArrayOf(1, 512, bucket.toLong()),
                VALID_MASK_10 to longArrayOf(1, 1, 10L * bucket),
                VALID_MASK_60 to longArrayOf(1, 1, 60L * bucket + 1L),
                "valid_length_10" to longArrayOf(1),
                "valid_length_60" to longArrayOf(1),
                SOURCE_SPECTRUM to longArrayOf(1, 22, 60L * bucket + 1L),
            )
            val inputInfo = holder.session.inputInfo
            check(inputInfo.keys == expectedInputs.keys) {
                "Repaired QNN B$bucket inputs changed: ${inputInfo.keys}"
            }
            expectedInputs.forEach { (name, expectedShape) ->
                val info = inputInfo.getValue(name).info as? TensorInfo
                    ?: error("Repaired QNN input $name is not a tensor")
                check(info.type == OnnxJavaType.FLOAT && info.shape.contentEquals(expectedShape)) {
                    "Repaired QNN input $name changed: type=${info.type} shape=${info.shape.contentToString()}"
                }
            }
            val outputInfo = holder.session.outputInfo
            check(outputInfo.keys == setOf("acoustic")) {
                "Repaired QNN B$bucket outputs changed: ${outputInfo.keys}"
            }
            val output = outputInfo.getValue("acoustic").info as? TensorInfo
                ?: error("Repaired QNN acoustic output is not a tensor")
            val expectedOutput = longArrayOf(1, 22, 60L * bucket + 1L)
            check(output.type == OnnxJavaType.FLOAT && output.shape.contentEquals(expectedOutput)) {
                "Repaired QNN output changed: type=${output.type} shape=${output.shape.contentToString()}"
            }
            return holder
        } catch (failure: Throwable) {
            closeAfterFailure(holder, failure)
            throw failure
        }
    }

    private fun parseSourceSpectrumManifest(manifest: String): Map<Int, SourceSpectrumArtifact> {
        check(manifest.isNotBlank()) { "Missing CPU source-spectrum manifest" }
        val encodedArtifacts = manifest.split('|')
        val artifacts = encodedArtifacts.associate { encoded ->
            val fields = encoded.split(',')
            check(fields.size == 4) { "Invalid CPU source-spectrum manifest entry" }
            val bucket = fields[0].toIntOrNull() ?: error("Invalid CPU source-spectrum bucket")
            val asset = fields[1]
            val hash = fields[2]
            val bytes = fields[3].toLongOrNull() ?: error("Invalid CPU source-spectrum size")
            check(asset.isNotBlank() && File(asset).name == asset) { "Invalid CPU source-spectrum asset" }
            check(hash.matches(Regex("[0-9a-f]{64}"))) { "Invalid CPU source-spectrum hash" }
            check(bytes > 0) { "Invalid CPU source-spectrum size" }
            bucket to SourceSpectrumArtifact(bucket, asset, hash, bytes)
        }
        check(artifacts.keys == setOf(64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640)) {
            "CPU source-spectrum bucket set changed: ${artifacts.keys.sorted()}"
        }
        check(artifacts.size == encodedArtifacts.size) { "Duplicate CPU source-spectrum bucket" }
        return artifacts
    }

    private fun cpuSourceSpectrumSession(bucket: Int): SessionHolder {
        val artifact = sourceSpectrumArtifacts[bucket]
            ?: error("No packaged CPU source-spectrum model for B$bucket")
        val creationLock = sourceSpectrumCreationLocks.computeIfAbsent(bucket) { Any() }
        return synchronized(creationLock) sourceSessionCreation@{
            val existing = synchronized(sessionLock) {
                check(!closed) { "Kokoro synthesizer is closed" }
                sourceSpectrumSessions[bucket]
            }
            if (existing != null) return@sourceSessionCreation existing
            val options = baseSessionOptions()
            val created = createOwnedSession(options = options) {
                environment.createSession(modelBuffer(artifact.asset), options)
            }
            try {
                val input = created.session.inputInfo[FRONT_PROSODY]?.info as? TensorInfo
                    ?: error("CPU source-spectrum B$bucket prosody input is missing")
                check(input.type == OnnxJavaType.FLOAT &&
                    input.shape.contentEquals(longArrayOf(1, 1, bucket.toLong()))) {
                    "CPU source-spectrum B$bucket prosody contract changed"
                }
                val output = created.session.outputInfo.values.singleOrNull()?.info as? TensorInfo
                    ?: error("CPU source-spectrum B$bucket output is missing")
                check(output.type == OnnxJavaType.FLOAT &&
                    output.shape.contentEquals(longArrayOf(1, 22, 60L * bucket + 1L))) {
                    "CPU source-spectrum B$bucket output contract changed"
                }
                synchronized(sessionLock) {
                    if (closed) {
                        created.close()
                        error("Kokoro synthesizer closed during CPU source-spectrum session creation")
                    }
                    sourceSpectrumSessions[bucket]?.let { existing ->
                        created.close()
                        return@synchronized existing
                    }
                    sourceSpectrumSessions[bucket] = created
                    created
                }
            } catch (failure: Throwable) {
                closeAfterFailure(created, failure)
                throw failure
            }
        }
    }

    private fun parseSharedQnnManifest(manifest: String): SharedQnnGroup {
        val fields = manifest.split('|')
        check(fields.size >= 4) { "Invalid shared QNN manifest" }
        fun artifact(asset: String, sha256: String, bytes: String): SharedQnnArtifact {
            check(asset.isNotBlank() && File(asset).name == asset) { "Invalid shared QNN asset name" }
            check(sha256.matches(Regex("[0-9a-f]{64}"))) { "Invalid shared QNN artifact hash" }
            val byteCount = bytes.toLongOrNull() ?: error("Invalid shared QNN artifact size")
            check(byteCount > 0) { "Invalid shared QNN artifact size" }
            return SharedQnnArtifact(asset, sha256, byteCount)
        }
        val binary = artifact(fields[0], fields[1], fields[2])
        val entries = fields.drop(3).map { encoded ->
            val values = encoded.split(',')
            check(values.size == 4) { "Invalid shared QNN wrapper entry" }
            val bucket = values[0].toIntOrNull() ?: error("Invalid shared QNN bucket")
            SharedQnnEntry(bucket, artifact(values[1], values[2], values[3]))
        }
        check(entries.isNotEmpty() && entries.map { it.bucket }.distinct().size == entries.size) {
            "Shared QNN manifest has no entries or duplicate buckets"
        }
        return SharedQnnGroup(binary, entries)
    }

    private fun stageSharedQnnGroup(group: SharedQnnGroup): File = synchronized(packagedAssetInstallLock) {
        val stableDirectory = File(context.noBackupFilesDir, "qnn-shared-${group.binary.sha256.take(12)}")
        val priorDirectory = context.noBackupFilesDir.listFiles()?.firstOrNull { candidate ->
            candidate.isDirectory && candidate.name.startsWith("qnn-shared-v") &&
                candidate.name.endsWith(group.binary.sha256.take(12))
        }
        val directory = when {
            stableDirectory.isDirectory -> stableDirectory
            priorDirectory != null -> priorDirectory
            else -> stableDirectory
        }
        check(directory.isDirectory || directory.mkdirs()) { "Cannot create shared QNN directory $directory" }
        val marker = File(directory, ".verified-${group.binary.sha256}")
        val binaryAlreadyVerified = marker.isFile &&
            marker.length() == group.binary.sha256.length.toLong() &&
            marker.readText(Charsets.US_ASCII) == group.binary.sha256
        (listOf(group.binary) + group.entries.map { it.wrapper }).forEach { artifact ->
            val destination = File(directory, artifact.asset)
            val isLargeBinary = artifact.asset == group.binary.asset
            val validExisting = destination.isFile && destination.length() == artifact.bytes &&
                (isLargeBinary && binaryAlreadyVerified || sha256File(destination) == artifact.sha256)
            if (!validExisting) {
                val temporary = File(directory, ".${artifact.asset}.$instanceId.part")
                try {
                    context.assets.open(artifact.asset).use { input ->
                        FileOutputStream(temporary).use { output -> input.copyTo(output, 1024 * 1024) }
                    }
                    check(temporary.length() == artifact.bytes && sha256File(temporary) == artifact.sha256) {
                        "Shared QNN asset failed integrity validation: ${artifact.asset}"
                    }
                    if (!temporary.renameTo(destination)) temporary.copyTo(destination, overwrite = true)
                    check(destination.length() == artifact.bytes) {
                        "Shared QNN asset install was incomplete: ${artifact.asset}"
                    }
                } finally {
                    if (temporary.exists() && !temporary.delete()) {
                        Log.w(TAG, "Unable to delete shared QNN partial $temporary")
                    }
                }
            }
        }
        if (!binaryAlreadyVerified) marker.writeText(group.binary.sha256, Charsets.US_ASCII)
        directory
    }

    private fun sha256File(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(Locale.US, it) }
    }

    private fun validateSharedQnnContextContract(holder: SessionHolder, bucket: Int): SessionHolder {
        try {
            val expectedInputs = linkedMapOf(
                FRONT_CONDITIONING to longArrayOf(1, 128),
                FRONT_PROSODY to longArrayOf(1, 1, bucket.toLong()),
                FRONT_DECODER to longArrayOf(1, 512, bucket.toLong()),
                VALID_MASK_10 to longArrayOf(1, 1, 10L * bucket),
                VALID_MASK_60 to longArrayOf(1, 1, 60L * bucket + 1L),
                "valid_length_10" to longArrayOf(1),
                "valid_length_60" to longArrayOf(1),
            )
            val inputInfo = holder.session.inputInfo
            check(inputInfo.keys == expectedInputs.keys) { "Shared QNN T=$bucket inputs changed: ${inputInfo.keys}" }
            expectedInputs.forEach { (name, expectedShape) ->
                val tensorInfo = inputInfo.getValue(name).info as? TensorInfo
                    ?: error("Shared QNN input $name is not a tensor")
                check(tensorInfo.type == OnnxJavaType.FLOAT && tensorInfo.shape.contentEquals(expectedShape)) {
                    "Shared QNN input $name changed: type=${tensorInfo.type} shape=${tensorInfo.shape.contentToString()}"
                }
            }
            val expectedOutput = if (isFullWaveformQnnBucket(bucket)) {
                "waveform" to longArrayOf(1, FRAME_SAMPLES.toLong() * bucket)
            } else {
                "acoustic" to longArrayOf(1, 22, 60L * bucket + 1L)
            }
            val outputInfo = holder.session.outputInfo
            check(outputInfo.keys == setOf(expectedOutput.first)) {
                "Shared QNN T=$bucket outputs changed: ${outputInfo.keys}"
            }
            val tensorInfo = outputInfo.getValue(expectedOutput.first).info as? TensorInfo
                ?: error("Shared QNN output is not a tensor")
            check(tensorInfo.type == OnnxJavaType.FLOAT && tensorInfo.shape.contentEquals(expectedOutput.second)) {
                "Shared QNN output changed: type=${tensorInfo.type} shape=${tensorInfo.shape.contentToString()}"
            }
            return holder
        } catch (failure: Throwable) {
            closeAfterFailure(holder, failure)
            throw failure
        }
    }

    private fun disableSharedQnnForProcess(reason: String, problem: Throwable? = null) {
        synchronized(sharedQnnCreationLock) {
            synchronized(sessionLock) {
                if (sharedQnnDisabledForProcess) return
                sharedQnnDisabledForProcess = true
                sharedV1QnnSessions.values.distinct().forEach { it.close() }
                sharedV1QnnSessions.clear()
            }
        }
        if (problem == null) {
            Log.w(TAG, "Shared QNN disabled for this process; using embedded contexts: $reason")
        } else {
            Log.w(TAG, "Shared QNN disabled for this process; using embedded contexts: $reason", problem)
        }
    }

    private fun runV1QnnAcoustic(
        holder: SessionHolder,
        inputs: GeneratorInputs,
        bucket: Int,
        sourceFeatures: FloatArray? = null,
    ): FloatArray {
        check(inputs.frames in 1..bucket) { "v1 QNN bucket $bucket cannot hold T=${inputs.frames}" }
        val leftPadFrames = bucket - inputs.frames
        val inputNames = holder.session.inputInfo.keys
        val tensorInputs = linkedMapOf<String, OnnxTensor>()
        if (FRONT_CONDITIONING in inputNames) {
            tensorInputs[FRONT_CONDITIONING] = floatTensor(inputs.conditioning, longArrayOf(1, 128))
        }
        if (FRONT_PROSODY in inputNames) {
            tensorInputs[FRONT_PROSODY] = floatTensor(
                leftPadLastDimension(inputs.prosody, 1, inputs.frames, bucket),
                longArrayOf(1, 1, bucket.toLong()),
            )
        }
        if (FRONT_DECODER in inputNames) {
            tensorInputs[FRONT_DECODER] = floatTensor(
                leftPadLastDimension(inputs.decoder, 512, inputs.frames, bucket),
                longArrayOf(1, 512, bucket.toLong()),
            )
        }
        if (HARMONIC_SOURCE in inputNames) {
            val source = requireNotNull(sourceFeatures) {
                "QNN suffix T=$bucket requires a CPU harmonic-source tensor"
            }
            check(source.size == bucket * FRAME_SAMPLES) {
                "CPU harmonic source returned ${source.size} values for B$bucket"
            }
            tensorInputs[HARMONIC_SOURCE] = floatTensor(
                source,
                longArrayOf(1, 1, (bucket * FRAME_SAMPLES).toLong()),
            )
        }
        if (SOURCE_SPECTRUM in inputNames) {
            val spectrum = requireNotNull(sourceFeatures) {
                "QNN suffix T=$bucket requires CPU source-spectrum features"
            }
            val spectrumFrames = bucket * 60 + 1
            check(spectrum.size == 22 * spectrumFrames) {
                "CPU source spectrum returned ${spectrum.size} values for B$bucket"
            }
            tensorInputs[SOURCE_SPECTRUM] = floatTensor(
                spectrum,
                longArrayOf(1, 22, spectrumFrames.toLong()),
            )
        }
        if (VALID_MASK_10 in inputNames && VALID_MASK_60 in inputNames) {
            val mask10Values = FloatArray(bucket * 10).apply {
                fill(1f, leftPadFrames * 10, bucket * 10)
            }
            val mask60Values = FloatArray(bucket * 60 + 1).apply {
                fill(1f, leftPadFrames * 60, bucket * 60 + 1)
            }
            tensorInputs[VALID_MASK_10] = floatTensor(
                mask10Values,
                longArrayOf(1, 1, (bucket * 10).toLong()),
            )
            tensorInputs[VALID_MASK_60] = floatTensor(
                mask60Values,
                longArrayOf(1, 1, (bucket * 60 + 1).toLong()),
            )
        }
        if ("valid_length_10" in inputNames && "valid_length_60" in inputNames) {
            tensorInputs["valid_length_10"] = floatTensor(
                floatArrayOf((inputs.frames * 10).toFloat()),
                longArrayOf(1),
            )
            tensorInputs["valid_length_60"] = floatTensor(
                floatArrayOf((inputs.frames * 60 + 1).toFloat()),
                longArrayOf(1),
            )
        }
        check(tensorInputs.keys == inputNames) {
            "Unsupported QNN acoustic inputs: expected=$inputNames supplied=${tensorInputs.keys}"
        }
        val runOptions = OrtSession.RunOptions()
        return try {
            configureQnnRunOptions(runOptions, holder)
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            holder.session.run(tensorInputs, runOptions).use { result ->
                check(result.size() == 1) { "v1 QNN acoustic returned ${result.size()} outputs" }
                val full = (result[0] as OnnxTensor).copyFloats()
                cropLastDimension(full, 22, 60 * bucket + 1, leftPadFrames * 60, 60 * inputs.frames + 1)
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            tensorInputs.values.forEach { it.close() }
        }
    }

    /** Runs the small, exact source-spectrum prefix on CPU. The returned
     * tensor includes bucket padding and is consumed directly by the repaired
     * HTP neural-vocoder suffix. */
    private fun runCpuSourceSpectrum(
        holder: SessionHolder,
        inputs: GeneratorInputs,
        bucket: Int,
    ): FloatArray {
        val prosody = floatTensor(
            leftPadLastDimension(inputs.prosody, 1, inputs.frames, bucket),
            longArrayOf(1, 1, bucket.toLong()),
        )
        val runOptions = OrtSession.RunOptions()
        return try {
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            holder.session.run(mapOf(FRONT_PROSODY to prosody), runOptions).use { result ->
                check(result.size() == 1) { "CPU source spectrum returned ${result.size()} outputs" }
                (result[0] as OnnxTensor).copyFloats().also { source ->
                    val validSize = source.size == bucket * FRAME_SAMPLES ||
                        source.size == 22 * (bucket * 60 + 1)
                    check(validSize && source.all(Float::isFinite)) {
                        "CPU source features are invalid for B$bucket: size=${source.size}"
                    }
                }
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            prosody.close()
        }
    }

    /** Runs a static masked v1 context whose output already includes iSTFT.
     *
     * Inputs are left-padded so temporal AdaIN sees only the real extent via
     * the valid masks. The resulting full-bucket waveform is cropped at the
     * corresponding audio offset; keeping this offset is essential because
     * iSTFT overlap-add is temporal, not frame-independent. */
    private fun runV1QnnFullWaveform(holder: SessionHolder, inputs: GeneratorInputs, bucket: Int): FloatArray {
        check(inputs.frames in 1..bucket) { "v1 QNN bucket $bucket cannot hold T=${inputs.frames}" }
        val leftPadFrames = bucket - inputs.frames
        val conditioning = floatTensor(inputs.conditioning, longArrayOf(1, 128))
        val prosody = floatTensor(
            leftPadLastDimension(inputs.prosody, 1, inputs.frames, bucket),
            longArrayOf(1, 1, bucket.toLong()),
        )
        val decoder = floatTensor(
            leftPadLastDimension(inputs.decoder, 512, inputs.frames, bucket),
            longArrayOf(1, 512, bucket.toLong()),
        )
        val mask10Values = FloatArray(bucket * 10).apply { fill(1f, leftPadFrames * 10, bucket * 10) }
        val mask60Values = FloatArray(bucket * 60 + 1).apply { fill(1f, leftPadFrames * 60, bucket * 60 + 1) }
        val mask10 = floatTensor(mask10Values, longArrayOf(1, 1, (bucket * 10).toLong()))
        val mask60 = floatTensor(mask60Values, longArrayOf(1, 1, (bucket * 60 + 1).toLong()))
        val length10 = floatTensor(floatArrayOf((inputs.frames * 10).toFloat()), longArrayOf(1))
        val length60 = floatTensor(floatArrayOf((inputs.frames * 60 + 1).toFloat()), longArrayOf(1))
        val tensorInputs = linkedMapOf(
            FRONT_CONDITIONING to conditioning,
            FRONT_PROSODY to prosody,
            FRONT_DECODER to decoder,
            VALID_MASK_10 to mask10,
            VALID_MASK_60 to mask60,
        )
        val inputNames = holder.session.inputInfo.keys
        if ("valid_length_10" in inputNames && "valid_length_60" in inputNames) {
            tensorInputs["valid_length_10"] = length10
            tensorInputs["valid_length_60"] = length60
        } else {
            length10.close()
            length60.close()
        }
        val runOptions = OrtSession.RunOptions()
        return try {
            configureQnnRunOptions(runOptions, holder)
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            holder.session.run(tensorInputs, runOptions).use { result ->
                check(result.size() == 1) { "v1 QNN full generator returned ${result.size()} outputs" }
                val full = (result[0] as OnnxTensor).copyFloats()
                val start = leftPadFrames * FRAME_SAMPLES
                val end = start + inputs.frames * FRAME_SAMPLES
                check(full.size >= end) {
                    "v1 QNN full generator returned ${full.size} samples; requires $end for T=${inputs.frames}"
                }
                full.copyOfRange(start, end)
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            tensorInputs.values.forEach { it.close() }
        }
    }

    /** Executes an arbitrary prefix probe with whichever standard v1 inputs
     * survived backward pruning. Diagnostic builds use this to locate the
     * first HTP tensor that becomes non-finite without involving iSTFT. */
    private fun runExternalQnnRaw(
        holder: SessionHolder,
        inputs: GeneratorInputs,
        bucket: Int,
        harmonicSource: FloatArray? = null,
    ): FloatArray {
        check(inputs.frames == bucket) {
            "Unmasked QNN prefix probes require exact T=$bucket; got T=${inputs.frames}"
        }
        val inputNames = holder.session.inputInfo.keys
        val tensorInputs = linkedMapOf<String, OnnxTensor>()
        if (FRONT_CONDITIONING in inputNames) {
            tensorInputs[FRONT_CONDITIONING] = floatTensor(inputs.conditioning, longArrayOf(1, 128))
        }
        if (FRONT_PROSODY in inputNames) {
            tensorInputs[FRONT_PROSODY] = floatTensor(
                inputs.prosody,
                longArrayOf(1, 1, bucket.toLong()),
            )
        }
        if (FRONT_DECODER in inputNames) {
            tensorInputs[FRONT_DECODER] = floatTensor(
                inputs.decoder,
                longArrayOf(1, 512, bucket.toLong()),
            )
        }
        if (HARMONIC_SOURCE in inputNames) {
            val source = requireNotNull(harmonicSource) {
                "QNN suffix probe requires a CPU harmonic-source tensor"
            }
            check(source.size == bucket * FRAME_SAMPLES) {
                "CPU harmonic source returned ${source.size} values for B$bucket"
            }
            tensorInputs[HARMONIC_SOURCE] = floatTensor(
                source,
                longArrayOf(1, 1, (bucket * FRAME_SAMPLES).toLong()),
            )
        }
        if (SOURCE_SPECTRUM in inputNames) {
            val spectrum = requireNotNull(harmonicSource) {
                "QNN suffix probe requires CPU source-spectrum features"
            }
            val spectrumFrames = bucket * 60 + 1
            check(spectrum.size == 22 * spectrumFrames) {
                "CPU source spectrum returned ${spectrum.size} values for B$bucket"
            }
            tensorInputs[SOURCE_SPECTRUM] = floatTensor(
                spectrum,
                longArrayOf(1, 22, spectrumFrames.toLong()),
            )
        }
        check(tensorInputs.keys == inputNames) {
            "Unsupported QNN prefix-probe inputs: expected=$inputNames supplied=${tensorInputs.keys}"
        }
        val runOptions = OrtSession.RunOptions()
        return try {
            configureQnnRunOptions(runOptions, holder)
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            holder.session.run(tensorInputs, runOptions).use { result ->
                check(result.size() == 1) { "QNN prefix probe returned ${result.size()} outputs" }
                (result[0] as OnnxTensor).copyFloats()
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            tensorInputs.values.forEach { it.close() }
        }
    }

    private fun runV1QnnWithSharedFallback(
        inputs: GeneratorInputs,
        bucket: Int,
        preparedPrefix: PreparedQnnPrefix? = null,
        onBeforeQnnRun: (() -> Unit)? = null,
    ): Pair<FloatArray, SessionHolder> {
        preparedPrefix?.let { prepared ->
            check(prepared.front === inputs && prepared.bucket == bucket) {
                "Prepared QNN prefix does not match T=${inputs.frames}/B$bucket"
            }
        }
        val sourceWasResident: Boolean
        val sourceSessionMs: Double
        val sourceRunMs: Double
        val sourceSpectrum: FloatArray
        if (preparedPrefix != null) {
            sourceWasResident = preparedPrefix.sourceWasResident
            sourceSessionMs = preparedPrefix.sourceSessionMs
            sourceRunMs = preparedPrefix.sourceRunMs
            sourceSpectrum = preparedPrefix.sourceSpectrum
        } else {
            sourceWasResident = synchronized(sessionLock) { bucket in sourceSpectrumSessions }
            val sourceSessionStarted = SystemClock.elapsedRealtimeNanos()
            val sourceSession = cpuSourceSpectrumSession(bucket)
            sourceSessionMs = elapsedMillis(sourceSessionStarted)
            val sourceRunStarted = SystemClock.elapsedRealtimeNanos()
            sourceSpectrum = runCpuSourceSpectrum(sourceSession, inputs, bucket)
            sourceRunMs = elapsedMillis(sourceRunStarted)
        }
        var beforeQnnRun = onBeforeQnnRun

        fun execute(
            holder: SessionHolder,
            qnnWasResident: Boolean,
            qnnSessionMs: Double,
        ): FloatArray {
            beforeQnnRun?.let { callback ->
                beforeQnnRun = null
                callback()
            }
            val qnnRunStarted = SystemClock.elapsedRealtimeNanos()
            val acoustic = runV1QnnAcoustic(holder, inputs, bucket, sourceSpectrum)
            val qnnRunMs = elapsedMillis(qnnRunStarted)
            val istftStarted = SystemClock.elapsedRealtimeNanos()
            val waveform = runV1Istft(acoustic, inputs.frames)
            val istftMs = elapsedMillis(istftStarted)
            Log.i(
                TAG,
                "stage=qnn_path T=${inputs.frames} B=$bucket " +
                    "source_context=${if (sourceWasResident) "hit" else "miss"} " +
                    "source_session_ms=${formatMillis(sourceSessionMs)} " +
                    "source_run_ms=${formatMillis(sourceRunMs)} " +
                    "qnn_context=${if (qnnWasResident) "hit" else "miss"} " +
                    "qnn_session_ms=${formatMillis(qnnSessionMs)} " +
                    "qnn_run_ms=${formatMillis(qnnRunMs)} istft_ms=${formatMillis(istftMs)}",
            )
            return waveform
        }

        val preferredWasResident = synchronized(sessionLock) {
            bucket in sharedV1QnnSessions || bucket in v1QnnAcousticSessions
        }
        val preferredSessionStarted = SystemClock.elapsedRealtimeNanos()
        val preferred = v1QnnAcousticSession(bucket)
        val preferredSessionMs = elapsedMillis(preferredSessionStarted)
        return try {
            execute(preferred, preferredWasResident, preferredSessionMs) to preferred
        } catch (problem: Exception) {
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            if (preferred.qnnSource != QNN_SESSION_SOURCE) throw problem
            disableSharedQnnForProcess("runtime failure for T=$bucket", problem)
            val fallbackWasResident = synchronized(sessionLock) { bucket in v1QnnAcousticSessions }
            val fallbackSessionStarted = SystemClock.elapsedRealtimeNanos()
            val fallback = legacyV1QnnAcousticSession(bucket)
            val fallbackSessionMs = elapsedMillis(fallbackSessionStarted)
            execute(fallback, fallbackWasResident, fallbackSessionMs) to fallback
        }
    }

    /** Debug-only physical-device harness for qualifying a newly compiled
     * embedded EPContext before it is admitted to the packaged bucket table.
     * The candidate must first be copied into this app's private filesDir.
     * Nothing here changes runtime preferences, caches, or fallback state. */
    internal fun captureExternalQnnCandidateForTesting(
        phonemes: String,
        voice: String,
        speed: Float,
        bucket: Int,
        contextFile: File,
        harmonicSourceFile: File? = null,
        fullWaveform: Boolean,
        rawOutput: Boolean = false,
        warmupRuns: Int = 0,
    ): QnnCandidateCapture = synchronized(sessionUseLock) {
        check(BuildConfig.DEBUG) { "External QNN capture is forbidden in release builds" }
        require(warmupRuns in 0..4) { "External QNN warmup count must be within 0..4" }
        check(!closed) { "Kokoro synthesizer is closed" }
        val privateRoot = context.filesDir.canonicalFile
        val candidate = contextFile.canonicalFile
        check(candidate.isFile && candidate.path.startsWith(privateRoot.path + File.separator)) {
            "QNN candidate must be a regular file inside $privateRoot"
        }
        val sourceCandidate = harmonicSourceFile?.canonicalFile
        check(sourceCandidate == null || sourceCandidate.isFile &&
            sourceCandidate.path.startsWith(privateRoot.path + File.separator)) {
            "CPU harmonic-source candidate must be a regular file inside $privateRoot"
        }
        val front = checkNotNull(frontSpan(phonemes, voice, speed).front)
        check(front.frames in 1..bucket) { "QNN candidate B$bucket cannot hold T=${front.frames}" }

        val options = baseSessionOptions()
        var holder: SessionHolder? = null
        var sourceHolder: SessionHolder? = null
        try {
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            val htpBackend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(htpBackend.isFile) { "Packaged QNN HTP backend is missing: $htpBackend" }
            options.addExecutionProvider(
                qnnDevices(),
                qnnProviderOptions(
                    htpBackend.absolutePath,
                    vtcmMb = if (bucket == QNN_V1_B192_FRAMES ||
                        bucket == QNN_V1_B512_FRAMES || bucket == QNN_V1_B640_FRAMES
                    ) 8 else 0,
                ),
            )
            holder = createOwnedSession(
                options = options,
                qnnSource = QNN_FALLBACK_SESSION_SOURCE,
                qnnContextSha256 = sha256File(candidate),
                qnnBucket = bucket,
            ) { environment.createSession(candidate.absolutePath, options) }
            sourceHolder = sourceCandidate?.let { file ->
                val sourceOptions = baseSessionOptions()
                createOwnedSession(options = sourceOptions) {
                    environment.createSession(file.absolutePath, sourceOptions)
                }
            }
            fun execute(): FloatArray {
                val sourceFeatures = sourceHolder?.let { runCpuSourceSpectrum(it, front, bucket) }
                return when {
                    rawOutput -> runExternalQnnRaw(holder, front, bucket, sourceFeatures)
                    fullWaveform -> runV1QnnFullWaveform(holder, front, bucket)
                    else -> runV1Istft(
                        runV1QnnAcoustic(holder, front, bucket, sourceFeatures),
                        front.frames,
                    )
                }
            }
            repeat(warmupRuns) { execute() }
            val started = SystemClock.elapsedRealtimeNanos()
            val samples = execute()
            QnnCandidateCapture(
                frames = front.frames,
                samples = samples,
                elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000L,
            )
        } catch (failure: Throwable) {
            if (holder == null) closeAfterFailure(options, failure)
            throw failure
        } finally {
            sourceHolder?.close()
            holder?.close()
        }
    }

    /** Debug-only sentence-level cadence reference. It runs one full logical front and one
     * uncut dynamic CPU generator, then applies only the same two external request-edge trims as
     * streaming. No independent text fragment is re-fronted. */
    internal fun captureUnsplitGlobalFrontForTesting(
        text: String,
        voice: KokoroVoice,
        speed: Float,
    ): QnnCandidateCapture = synchronized(sessionUseLock) {
        check(BuildConfig.DEBUG) { "Global-front capture is forbidden in release builds" }
        check(!closed) { "Kokoro synthesizer is closed" }
        val started = SystemClock.elapsedRealtimeNanos()
        val phonemes = phonemizer.phonemize(text, voice.locale)
        val front = checkNotNull(frontSpan(phonemes, voice.id, speed).front)
        val generated = runGenerator(front)
        val trimmed = trimArtificialEdgeSilence(
            generated,
            REQUEST_BOUNDARY_LEADING_SAMPLES,
            REQUEST_BOUNDARY_TRAILING_SAMPLES,
        )
        QnnCandidateCapture(
            frames = front.frames,
            samples = trimmed,
            elapsedMs = elapsedMillis(started).toLong(),
        )
    }

    /** Captures the production first global window immediately before and after its sole leading
     * request-edge trim. No Android audio sink is opened and no continuation is generated. */
    @Synchronized
    internal fun captureOpeningEdgeForTesting(
        text: String,
        voice: KokoroVoice,
        speed: Float,
    ): OpeningEdgeCapture = synchronized(sessionUseLock) {
        check(BuildConfig.DEBUG) { "Opening-edge capture is forbidden in release builds" }
        check(!closed) { "Kokoro synthesizer is closed" }
        val phonemes = phonemizer.phonemize(text, voice.locale)
        val parts = sourcePlannedPhonemeParts(text, phonemes, voice.locale, speed)
        val plan = checkNotNull(globallyConditionedPlans(phonemes, parts, voice.id, speed)?.firstOrNull()) {
            "Opening-edge capture requires the global sentence route"
        }
        val generated = infer(plan.phonemes, voice.id, speed, plan.front)
        val coreFrames = checkNotNull(plan.coreFrames)
        val extendedStartFrame = plan.coreStartFrame - plan.leadingOverlapFrames
        val extendedFrames = coreFrames + plan.leadingOverlapFrames + plan.trailingOverlapFrames
        val startSample = Math.multiplyExact(extendedStartFrame, FRAME_SAMPLES)
        val sampleCount = Math.multiplyExact(extendedFrames, FRAME_SAMPLES)
        val raw = generated.copyOfRange(startSample, startSample + sampleCount)
        val detectedActiveStart = sustainedActiveRange(raw)?.first
        val trimmed = trimArtificialEdgeSilence(raw, REQUEST_BOUNDARY_LEADING_SAMPLES, null)
        OpeningEdgeCapture(
            frames = checkNotNull(plan.front).frames,
            rawSamples = raw,
            trimmedSamples = trimmed,
            detectedActiveStartSample = detectedActiveStart,
            removedLeadingSamples = raw.size - trimmed.size,
        )
    }

    /** Inspects production request planning without running a generator or producing PCM. */
    @Synchronized
    internal fun inspectGlobalPlanForTesting(
        text: String,
        voice: KokoroVoice,
        speed: Float,
    ): GlobalPlanInspection {
        check(BuildConfig.DEBUG) { "Global plan inspection is forbidden in release builds" }
        check(!closed) { "Kokoro synthesizer is closed" }
        val phonemes = phonemizer.phonemize(text, voice.locale)
        val parts = sourcePlannedPhonemeParts(text, phonemes, voice.locale, speed)
        val plans = globallyConditionedPlans(phonemes, parts, voice.id, speed)
        val windowFrames = plans.orEmpty().map { plan -> checkNotNull(plan.front).frames }
        return GlobalPlanInspection(
            phonemeChars = phonemes.length,
            initialChunks = parts.size,
            finalChunks = plans?.size ?: parts.size,
            globallyConditioned = plans != null,
            coreFrames = plans.orEmpty().map { plan -> checkNotNull(plan.coreFrames) },
            windowFrames = windowFrames,
            qnnBuckets = windowFrames.map(::aotBucketForFrames),
            leadingOverlapFrames = plans.orEmpty().map { plan -> plan.leadingOverlapFrames },
            trailingOverlapFrames = plans.orEmpty().map { plan -> plan.trailingOverlapFrames },
        )
    }

    private fun runV1Istft(acoustic: FloatArray, frames: Int): FloatArray {
        val input = floatTensor(acoustic, longArrayOf(1, 22, (60 * frames + 1).toLong()))
        val runOptions = OrtSession.RunOptions()
        return try {
            registerActiveRun(runOptions)
            v1IstftSession().session.run(
                mapOf("/decoder/decoder/generator/conv_post/Conv_output_0" to input),
                runOptions,
            ).use { result ->
                check(result.size() == 1) { "v1 iSTFT returned ${result.size()} outputs" }
                (result[0] as OnnxTensor).copyFloats()
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            input.close()
        }
    }

    private fun qnnGeneratorSessionSingleFlight(bucket: Int): SessionHolder {
        synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            qnnGeneratorSessions[bucket]?.let { return it }
        }

        // Mmap/load the embedded EPContext outside the lifecycle lock. There is
        // deliberately no source-model/JIT path in the shipping runtime.
        val created = createQnnGeneratorSession(bucket)
        return synchronized(sessionLock) {
            if (closed) {
                created.close()
                error("Kokoro synthesizer closed during QNN session creation")
            }
            qnnGeneratorSessions[bucket]?.let { existing ->
                created.close()
                return@synchronized existing
            }
            qnnGeneratorSessions[bucket] = created
            while (qnnGeneratorSessions.size > MAX_QNN_BUCKET_SESSIONS) {
                val eldest = qnnGeneratorSessions.entries.iterator().next()
                qnnGeneratorSessions.remove(eldest.key)
                eldest.value.close()
                Log.i(TAG, "Evicted QNN generator bucket T=${eldest.key}")
            }
            Log.i(TAG, "Using packaged QAIRT 2.48 QNN HTP FP32 context bucket T=$bucket")
            created
        }
    }

    private fun qnnSessionOptions(bucket: Int): OrtSession.SessionOptions {
        val options = baseSessionOptions()
        return try {
            options.apply {
                setSymbolicDimensionValue("unk__443", 1)
                setSymbolicDimensionValue("unk__445", bucket.toLong())
                setSymbolicDimensionValue("unk__648", bucket.toLong())
                setSymbolicDimensionValue("kokoro_mask_10_length", 10L * bucket)
                setSymbolicDimensionValue("kokoro_mask_60_length", 60L * bucket + 1L)
                setSymbolicDimensionValue("audio_length", FRAME_SAMPLES.toLong() * bucket)
                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                val htpBackend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                check(htpBackend.isFile) { "Packaged QNN HTP backend is missing: $htpBackend" }
                addExecutionProvider(qnnDevices(), qnnProviderOptions(htpBackend.absolutePath))
            }
        } catch (failure: Throwable) {
            closeAfterFailure(options, failure)
            throw failure
        }
    }

    private fun qnnProviderOptions(
        backendPath: String,
        vtcmMb: Int = 0,
        sharedPowerTuning: Boolean = false,
    ): LinkedHashMap<String, String> = linkedMapOf(
        "backend_path" to backendPath,
        "htp_performance_mode" to "balanced",
        "qnn_context_priority" to "high",
        "htp_graph_finalization_optimization_mode" to "3",
        "soc_model" to QNN_SOC_MODEL,
        "htp_arch" to QNN_HTP_ARCH,
        // The compiled contexts already contain QAIRT's device-native lowering.
        // Do not request an additional blind FP16 rewrite while loading them.
        "enable_htp_fp16_precision" to "0",
        "offload_graph_io_quantization" to "0",
    ).also { options ->
        if (vtcmMb > 0) options["vtcm_mb"] = vtcmMb.toString()
        if (sharedPowerTuning) options["htp_share_resource_optimization"] = "1"
    }

    private fun configureQnnRunOptions(runOptions: OrtSession.RunOptions, holder: SessionHolder) {
        runOptions.addRunConfigEntry("qnn.perf_mode", "burst")
        if (holder.qnnSource == QNN_SESSION_SOURCE) {
            runOptions.addRunConfigEntry("qnn.rpc_control_latency", "100")
        }
    }

    private fun buildQnnContextCacheId(): String {
        val identity = buildString {
            append("schema=").append(QNN_CONTEXT_SCHEMA).append('\n')
            append("front_sha256=").append(BuildConfig.KOKORO_FRONT_MODEL_SHA256).append('\n')
            append("model_sha256=").append(BuildConfig.KOKORO_GENERATOR_MODEL_SHA256).append('\n')
            append("ort=").append(BuildConfig.ORT_RUNTIME_COORDINATE).append('\n')
            append("plugin=").append(BuildConfig.QNN_PROVIDER_COORDINATE).append('\n')
            append("runtime=").append(BuildConfig.QNN_RUNTIME_COORDINATE).append('\n')
            append("context_producer=").append(BuildConfig.KOKORO_QNN_CONTEXT_PRODUCER).append('\n')
            append("context_b256_sha256=").append(BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256).append('\n')
            append("context_b192_sha256=").append(BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256).append('\n')
            append("context_b384_sha256=").append(BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256).append('\n')
            append("context_b64_sha256=").append(BuildConfig.KOKORO_QNN_B64_CONTEXT_SHA256).append('\n')
            append("context_b96_sha256=").append(BuildConfig.KOKORO_QNN_B96_CONTEXT_SHA256).append('\n')
            append("context_b128_sha256=").append(BuildConfig.KOKORO_QNN_B128_CONTEXT_SHA256).append('\n')
            append("context_b208_sha256=").append(BuildConfig.KOKORO_QNN_B208_CONTEXT_SHA256).append('\n')
            append("context_b224_sha256=").append(BuildConfig.KOKORO_QNN_B224_CONTEXT_SHA256).append('\n')
            append("context_b320_sha256=").append(BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256).append('\n')
            append("context_b512_sha256=").append(BuildConfig.KOKORO_QNN_B512_CONTEXT_SHA256).append('\n')
            append("context_b640_sha256=").append(BuildConfig.KOKORO_QNN_B640_CONTEXT_SHA256).append('\n')
            append("source_spectrum_manifest=")
                .append(BuildConfig.KOKORO_QNN_SOURCE_SPECTRUM_MANIFEST).append('\n')
            append("shared_included=").append(BuildConfig.KOKORO_QNN_SHARED_INCLUDED).append('\n')
            append("shared_acoustic=").append(BuildConfig.KOKORO_QNN_SHARED_ACOUSTIC_MANIFEST).append('\n')
            append("shared_mid=").append(BuildConfig.KOKORO_QNN_SHARED_MID_MANIFEST).append('\n')
            append("shared_large=").append(BuildConfig.KOKORO_QNN_SHARED_LARGE_MANIFEST).append('\n')
            append("session.disable_cpu_ep_fallback=1\n")
            append("ep.share_ep_contexts=").append(if (BuildConfig.KOKORO_QNN_SHARED_INCLUDED) "1" else "0").append('\n')
            append("session.execution_mode=SEQUENTIAL\n")
            append("session.graph_optimization=ALL\n")
            append("session.memory_pattern=1\n")
            append("session.cpu_arena=1\n")
            append("symbol_contract=conditioning[1,128];decoder[1,512,B];")
            append("source_spectrum[1,22,60B+1];mask10[1,1,10B];mask60[1,1,60B+1];")
            append("acoustic[1,22,60B+1];all_buckets_pow_x2_mul=1\n")
            append("run.qnn.perf_mode=burst\n")
            qnnProviderOptions("libQnnHtp.so").forEach { (key, value) ->
                append("fallback.").append(key).append('=').append(value).append('\n')
            }
            if (BuildConfig.KOKORO_QNN_SHARED_INCLUDED) {
                qnnProviderOptions("libQnnHtp.so", vtcmMb = 8, sharedPowerTuning = true).forEach { (key, value) ->
                    append("shared.").append(key).append('=').append(value).append('\n')
                }
            }
        }
        return MessageDigest.getInstance("SHA-256")
            .digest(identity.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(Locale.US, it) }
            .take(24)
    }

    private fun qnnDevices(): List<OrtEpDevice> {
        qnnEpDevices?.let { return it }
        synchronized(qnnPluginLock) {
            if (!qnnPluginRegistered) {
                val nativeLibraryDirectory = File(context.applicationInfo.nativeLibraryDir)
                check(nativeLibraryDirectory.isDirectory) {
                    "Android nativeLibraryDir is unavailable: $nativeLibraryDirectory"
                }
                configureDspLibraryPath(nativeLibraryDirectory)
                val plugin = File(nativeLibraryDirectory, getLibraryPath())
                check(plugin.isFile) { "Packaged QNN plugin is missing: $plugin" }
                environment.registerExecutionProviderLibrary(getEpName(), plugin.absolutePath)
                qnnPluginRegistered = true
                Log.i(TAG, "Registered ${getEpName()} from ${plugin.absolutePath}")
            }
        }
        val publishedDevices = environment.epDevices
        Log.i(TAG, "ORT EP devices after QNN registration: ${publishedDevices.joinToString()}")
        return publishedDevices.filter { it.epName == getEpName() }.also { devices ->
            check(devices.isNotEmpty()) {
                "${getEpName()} did not publish an EP device; ORT devices=$publishedDevices"
            }
            qnnEpDevices = devices
        }
    }

    private fun configureDspLibraryPath(nativeLibraryDirectory: File) {
        val paths = linkedSetOf(nativeLibraryDirectory.absolutePath)
        System.getenv("ADSP_LIBRARY_PATH")
            ?.split(';')
            ?.filterTo(paths) { it.isNotBlank() }
        paths += listOf(
            "/vendor/dsp",
            "/system/vendor/lib/rfsa/adsp",
            "/system/lib/rfsa/adsp",
            "/dsp",
        )
        Os.setenv("ADSP_LIBRARY_PATH", paths.joinToString(";"), true)
        Log.i(TAG, "Configured ADSP_LIBRARY_PATH with Android nativeLibraryDir")
    }

    private fun createQnnGeneratorSession(bucket: Int): SessionHolder {
        check(BuildConfig.KOKORO_QNN_AOT_INCLUDED) { "Packaged QNN AOT contexts are unavailable" }
        val (assetName, sha256) = when (bucket) {
            QNN_B256_FRAMES -> BuildConfig.KOKORO_QNN_B256_CONTEXT_ASSET to
                BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256
            QNN_B384_FRAMES -> BuildConfig.KOKORO_QNN_B384_CONTEXT_ASSET to
                BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256
            else -> error("No packaged QNN AOT context for T=$bucket")
        }
        check(assetName.isNotBlank() && sha256.matches(Regex("[0-9a-f]{64}"))) {
            "Invalid packaged QNN AOT identity for T=$bucket"
        }
        val options = qnnSessionOptions(bucket)
        val holder = createOwnedSession(
            options = options,
            qnnSource = QNN_SESSION_SOURCE,
            qnnContextSha256 = sha256,
            qnnBucket = bucket,
        ) {
            environment.createSession(modelBuffer(assetName), options)
        }
        return validateQnnContextContract(holder, bucket)
    }

    private fun validateQnnContextContract(holder: SessionHolder, bucket: Int): SessionHolder {
        try {
            val expectedInputs = linkedMapOf(
                FRONT_CONDITIONING to longArrayOf(1, 128),
                FRONT_PROSODY to longArrayOf(1, 1, bucket.toLong()),
                FRONT_DECODER to longArrayOf(1, 512, bucket.toLong()),
                VALID_MASK_10 to longArrayOf(1, 1, 10L * bucket),
                VALID_MASK_60 to longArrayOf(1, 1, 60L * bucket + 1L),
            )
            val inputInfo = holder.session.inputInfo
            check(inputInfo.keys == expectedInputs.keys) {
                "Packaged QNN T=$bucket inputs changed: ${inputInfo.keys}"
            }
            expectedInputs.forEach { (name, expectedShape) ->
                val tensorInfo = inputInfo.getValue(name).info as? TensorInfo
                    ?: error("Packaged QNN input $name is not a tensor")
                check(tensorInfo.type == OnnxJavaType.FLOAT && tensorInfo.shape.contentEquals(expectedShape)) {
                    "Packaged QNN input $name changed: type=${tensorInfo.type} " +
                        "shape=${tensorInfo.shape.contentToString()}"
                }
            }

            val outputInfo = holder.session.outputInfo
            check(outputInfo.keys == setOf("audio")) {
                "Packaged QNN T=$bucket outputs changed: ${outputInfo.keys}"
            }
            val audioInfo = outputInfo.getValue("audio").info as? TensorInfo
                ?: error("Packaged QNN audio output is not a tensor")
            val expectedAudioShape = longArrayOf(FRAME_SAMPLES.toLong() * bucket)
            check(audioInfo.type == OnnxJavaType.FLOAT && audioInfo.shape.contentEquals(expectedAudioShape)) {
                "Packaged QNN audio changed: type=${audioInfo.type} shape=${audioInfo.shape.contentToString()}"
            }
            return holder
        } catch (failure: Throwable) {
            closeAfterFailure(holder, failure)
            throw failure
        }
    }

    private fun modelBuffer(assetName: String): ByteBuffer {
        val mappingLock = modelMappingLocks.computeIfAbsent(assetName) { Any() }
        return synchronized(mappingLock) {
            val existing = synchronized(sessionLock) {
                check(!closed) { "Kokoro synthesizer is closed" }
                mappedModels[assetName]?.duplicate()?.apply { position(0) }
            }
            if (existing != null) return@synchronized existing

            // AOT contexts are hundreds of megabytes. Map/copy them without
            // holding the lifecycle lock so onDestroy can cancel immediately.
            val mapped = mapPackagedModel(assetName)
            synchronized(sessionLock) {
                check(!closed) { "Kokoro synthesizer closed while mapping $assetName" }
                val installed = mappedModels[assetName] ?: mapped.also { mappedModels[assetName] = it }
                installed.duplicate().apply { position(0) }
            }
        }
    }

    private fun mapPackagedModel(assetName: String): ByteBuffer = try {
        context.assets.openFd(assetName).use { asset ->
            val duplicate = ParcelFileDescriptor.dup(asset.fileDescriptor)
            ParcelFileDescriptor.AutoCloseInputStream(duplicate).use { stream ->
                stream.channel.map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
            }
        }
    } catch (notMappable: Exception) {
        Log.w(TAG, "$assetName was compressed; using a private installed copy", notMappable)
        val file = installAsset(assetName, "v${BuildConfig.VERSION_CODE}-$assetName")
        FileInputStream(file).use { input ->
            input.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        }
    }

    /** Emits a sentence-aware opening and computes one bounded chunk ahead during playback. */
    @Synchronized
    fun synthesizeChunks(
        text: String,
        voice: KokoroVoice,
        speed: Float,
        diagnostics: ((ChunkOutputDiagnostics) -> Unit)? = null,
        consumer: (ByteArray) -> Boolean,
    ) {
        val utteranceStarted = SystemClock.elapsedRealtimeNanos()
        synchronized(sessionLock) {
            check(!closed) { "Kokoro synthesizer is closed" }
            cancelRequested = false
        }
        beginUtteranceDiagnostics()
        try {
            val phonemizeStarted = SystemClock.elapsedRealtimeNanos()
            val phonemes = phonemizer.phonemize(text, voice.locale)
            val phonemizeMs = elapsedMillis(phonemizeStarted)
            val planningStarted = SystemClock.elapsedRealtimeNanos()
            val parts = sourcePlannedPhonemeParts(text, phonemes, voice.locale, speed)
            val globallyConditioned = globallyConditionedPlans(phonemes, parts, voice.id, speed)
            val plans = globallyConditioned ?: run {
                val postDurationSpans = coalescePostDurationSpans(parts, voice.id, speed)
                postDurationSpans.mapIndexed { index, span ->
                    PlannedChunk(
                        phonemes = span.phonemes,
                        leadingSeam = if (index == 0) {
                            WaveformSeam.REQUEST_BOUNDARY
                        } else {
                            waveformSeamAfter(postDurationSpans[index - 1].phonemes)
                        },
                        trailingSeam = if (index == postDurationSpans.lastIndex) {
                            WaveformSeam.REQUEST_BOUNDARY
                        } else {
                            waveformSeamAfter(span.phonemes)
                        },
                        front = span.front,
                    )
                }
            }
            val planningMs = elapsedMillis(planningStarted)
            Log.i(
                TAG,
                    "stage=request_plan chars=${text.length} phoneme_chars=${phonemes.length} " +
                    "phonemize_ms=${formatMillis(phonemizeMs)} planning_ms=${formatMillis(planningMs)} " +
                    "initial_chunks=${parts.size} final_chunks=${plans.size} " +
                    "global_conditioned=${globallyConditioned != null}",
            )
            if (globallyConditioned != null && shouldUseQnn()) {
                ensureGlobalPlanQnnSessions(plans)
            }
            var emittedChunks = 0
            // Both controllers sit after chunk synthesis/cache lookup. Loudness correction is
            // utterance-local; the overlap joiner collapses duplicate renders of shared global
            // frames without changing the sentence timeline.
            val loudnessStabilizer = ChunkLoudnessStabilizer()
            val overlapJoiner = GlobalPcmOverlapJoiner()
            var previousRetainedTrailingQuietSamples = 0
            val producedChunks = AtomicInteger(0)
            val prefixStart = CountDownLatch(1)
            val prefixExecutor = if (plans.size > 1 && shouldUseQnn()) {
                Executors.newSingleThreadExecutor { runnable ->
                    Thread(
                        {
                            try {
                                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO)
                            } catch (_: Exception) {
                                // Best effort: priority changes can be denied on vendor builds.
                            }
                            runnable.run()
                        },
                        "kokoro-prefix-prefetch",
                    ).apply { isDaemon = true }
                }
            } else {
                null
            }
            val prefixFuture = prefixExecutor?.submit(Callable {
                try {
                    prefixStart.await()
                } catch (interrupted: InterruptedException) {
                    Thread.currentThread().interrupt()
                    throw CancellationException("Continuation prefix prefetch interrupted").apply {
                        initCause(interrupted)
                    }
                }
                Log.i(TAG, "stage=chunk_prefix_start index=1")
                prepareQnnPrefix(plans[1], voice.id, speed)
            })
            try {
                consumeOneChunkAhead(
                    items = plans,
                    produce = { plan ->
                        val index = producedChunks.getAndIncrement()
                        val started = SystemClock.elapsedRealtimeNanos()
                        Log.i(TAG, "stage=chunk_produce_start index=$index")
                        val preparedPrefix = if (index == 1) {
                            prefixFuture?.let(::awaitProduced)
                        } else {
                            null
                        }
                        try {
                            synthesizePlannedChunk(
                                plan,
                                voice.id,
                                speed,
                                preparedQnnPrefix = preparedPrefix,
                                onBeforeQnnRun = if (index == 0 && prefixFuture != null) {
                                    {
                                        Log.i(TAG, "stage=opening_qnn_starting prefix_overlap=true")
                                        prefixStart.countDown()
                                    }
                                } else {
                                    null
                                },
                            ).also { pcm ->
                                val pcmMs = pcm.size * 1_000.0 / (SAMPLE_RATE * Short.SIZE_BYTES)
                                Log.i(
                                    TAG,
                                    "stage=chunk_produce_ready index=$index bytes=${pcm.size} " +
                                        "pcm_ms=${formatMillis(pcmMs)} " +
                                        "elapsed_ms=${formatMillis(elapsedMillis(started))}",
                                )
                            }
                        } finally {
                            if (index == 0) prefixStart.countDown()
                        }
                    },
                    consume = { pcm ->
                        val index = emittedChunks++
                        val plan = plans[index]
                        val stabilized = loudnessStabilizer.stabilize(pcm)
                        val leadingHalfOverlapSamples = plan.leadingOverlapFrames * FRAME_SAMPLES
                        val trailingHalfOverlapSamples = plan.trailingOverlapFrames * FRAME_SAMPLES
                        val delivered = overlapJoiner.stitch(
                            stabilized.pcm,
                            leadingHalfOverlapSamples,
                            trailingHalfOverlapSamples,
                        )
                        diagnostics?.invoke(
                            ChunkOutputDiagnostics(
                                index = index,
                                preStabilizerPcm = pcm.copyOf(),
                                postStabilizerPcm = stabilized.pcm.copyOf(),
                                deliveredPcm = delivered.copyOf(),
                                coreSamples = stabilized.pcm.size / Short.SIZE_BYTES -
                                    leadingHalfOverlapSamples - trailingHalfOverlapSamples,
                                leadingOverlapSamples = leadingHalfOverlapSamples,
                                trailingOverlapSamples = trailingHalfOverlapSamples,
                                activeRms = stabilized.activeRms,
                                referenceRms = stabilized.referenceRms,
                                requestedGain = stabilized.requestedGain,
                                startGain = stabilized.startGain,
                                appliedGain = stabilized.appliedGain,
                                rampSamples = stabilized.rampSamples,
                                peak = stabilized.peak,
                            ),
                        )
                        val leadingQuietSamples = pcmEdgeQuietSamples(delivered, leading = true)
                        val trailingQuietSamples = pcmEdgeQuietSamples(delivered, leading = false)
                        if (index > 0 && plans[index - 1].globallyConditioned && plans[index].globallyConditioned) {
                            val retainedMs = (previousRetainedTrailingQuietSamples + leadingQuietSamples) *
                                1_000.0 / SAMPLE_RATE
                            Log.i(
                                TAG,
                                "stage=global_boundary_pause index=${index - 1}:$index " +
                                    "retained_near_silence_ms=${formatMillis(retainedMs)} " +
                                    "designed_pause_ms=0 computation_gap_not_padded=true",
                            )
                        }
                        previousRetainedTrailingQuietSamples = trailingQuietSamples
                        if (leadingHalfOverlapSamples > 0 || trailingHalfOverlapSamples > 0) {
                            Log.i(
                                TAG,
                                "stage=global_overlap index=$index half_samples=" +
                                    "$leadingHalfOverlapSamples:$trailingHalfOverlapSamples " +
                                    "window_bytes=${stabilized.pcm.size} delivered_bytes=${delivered.size}",
                            )
                        }
                        Log.i(
                            TAG,
                            "stage=chunk_loudness index=$index active_rms=${stabilized.activeRms ?: -1.0} " +
                                "reference_rms=${stabilized.referenceRms ?: -1.0} " +
                                "requested_gain=${stabilized.requestedGain} applied_gain=${stabilized.appliedGain} " +
                                "peak=${stabilized.peak}",
                        )
                        if (index == 0) {
                            Log.i(
                                TAG,
                                "stage=first_pcm_ready bytes=${delivered.size} " +
                                    "elapsed_ms=${formatMillis(elapsedMillis(utteranceStarted))}",
                            )
                        }
                        val consumeStarted = SystemClock.elapsedRealtimeNanos()
                        Log.i(TAG, "stage=chunk_consume_start index=$index bytes=${delivered.size}")
                        consumer(delivered).also { accepted ->
                            Log.i(
                                TAG,
                                "stage=chunk_consume_end index=$index accepted=$accepted " +
                                    "elapsed_ms=${formatMillis(elapsedMillis(consumeStarted))}",
                            )
                        }
                    },
                    cancelProducer = { cancel() },
                )
                if (emittedChunks == plans.size) overlapJoiner.requireComplete()
            } finally {
                prefixStart.countDown()
                prefixFuture?.cancel(true)
                prefixExecutor?.shutdownNow()
                while (prefixExecutor?.awaitTermination(100, TimeUnit.MILLISECONDS) == false) {
                    if (closed || cancelRequested) cancel()
                }
            }
            Log.i(
                TAG,
                "stage=request_complete chunks=$emittedChunks " +
                    "elapsed_ms=${formatMillis(elapsedMillis(utteranceStarted))}",
            )
        } finally {
            persistRuntimeDiagnostics()
        }
    }

    /**
     * Resolves every QNN bucket needed by a globally conditioned utterance before its opening
     * PCM is exposed.  This is normally a zero-cost cache check after startup prewarming.  For a
     * rarer adaptive window such as B256, paying the one-time context load here is preferable to
     * exhausting already-queued B128 audio and restarting the speaker halfway through a phrase.
     */
    private fun ensureGlobalPlanQnnSessions(plans: List<PlannedChunk>) {
        val buckets = plans.mapNotNull { plan ->
            plan.front?.frames?.let(::aotBucketForFrames)
        }.distinct()
        if (buckets.isEmpty()) return
        if (buckets.size > MAX_QNN_BUCKET_SESSIONS) {
            Log.w(
                TAG,
                "stage=qnn_plan_preflight skipped=true buckets=${buckets.joinToString()} " +
                    "reason=lru_capacity_$MAX_QNN_BUCKET_SESSIONS",
            )
            return
        }
        val started = SystemClock.elapsedRealtimeNanos()
        try {
            buckets.forEach { bucket ->
                cpuSourceSpectrumSession(bucket)
                legacyV1QnnAcousticSession(bucket)
            }
        } catch (problem: Exception) {
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            // Preserve the established per-chunk QNN -> CPU fallback/disable path.  Preflight is
            // a continuity optimization and must not turn an accelerator problem into no audio.
            Log.w(TAG, "QNN plan preflight did not complete; retaining normal fallback", problem)
            return
        }
        Log.i(
            TAG,
            "stage=qnn_plan_preflight buckets=${buckets.joinToString()} " +
                "elapsed_ms=${formatMillis(elapsedMillis(started))}",
        )
    }

    /**
     * A single left-to-right post-front pass over the original text plan. A short unqualified
     * clause may consume its immediate neighbor exactly once when the joined clause fits the
     * model and its measured duration selects a packaged static context. No span is reordered or
     * re-split, and final waveform seams are computed only after this pass.
     */
    private fun coalescePostDurationSpans(parts: List<String>, voice: String, speed: Float): List<FrontedSpan> {
        // CPU/NNAPI keeps its established lazy full-model path. The extra front pass exists only
        // to recover QNN-eligible joins and must not make fallback-only synthesis do new work.
        if (!shouldUseQnn()) return parts.map { part -> FrontedSpan(part, null) }
        // Every short duration currently has a packaged B64..B192 AOT route, so the historical
        // "short unqualified" merge cannot fire. Keep the exact fallback pass below for any
        // future incomplete context set, while deferring current remainder fronts until their
        // one-chunk-ahead producer overlaps Android playback of the opening.
        if (hasContiguousAotCoverageThrough(QNN_V1_B192_FRAMES - 1)) {
            return parts.map { part -> FrontedSpan(part, null) }
        }
        val output = mutableListOf<FrontedSpan>()
        var index = 0
        while (index < parts.size) {
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            val current = frontSpan(parts[index], voice, speed)
            val currentFront = checkNotNull(current.front)
            val shortUnqualified = currentFront.frames in 1 until QNN_V1_B192_FRAMES &&
                aotBucketForFrames(currentFront.frames) == null
            if (shortUnqualified && index < parts.lastIndex) {
                val joinedText = "${current.phonemes} ${parts[index + 1]}"
                if (KokoroTokenizer.tokenize(context, joinedText).size <= MAX_TOKENS - 2) {
                    val joined = frontSpan(joinedText, voice, speed)
                    if (aotBucketForFrames(checkNotNull(joined.front).frames) != null) {
                        output += joined
                        index += 2
                        continue
                    }
                }
            }
            output += current
            index++
        }
        return output
    }

    /** Runs sentence-level text/duration/intonation conditioning once, then creates bounded QNN
     * generator windows from that one frame path. A speech-safe opening normally fills B128; an
     * original B192 semantic boundary is retained when forcing B128 would leave the next core
     * without useful context. Ordinary continuations retain B192; an oversized semantic core
     * selects the smallest packaged bucket that preserves context and the shared boundary.
     * Only disjoint semantic cores are emitted. */
    private fun globallyConditionedPlans(
        phonemes: String,
        parts: List<String>,
        voice: String,
        speed: Float,
    ): List<PlannedChunk>? {
        if (!shouldUseQnn()) return null
        val tokenCount = KokoroTokenizer.tokenize(context, phonemes).size
        if (tokenCount !in 1..(MAX_TOKENS - 2)) return null
        val globalFrontStarted = SystemClock.elapsedRealtimeNanos()
        val globalFront = checkNotNull(frontSpan(phonemes, voice, speed).front)
        val durations = checkNotNull(globalFront.tokenDurations) {
            "CPU front did not expose exact token durations"
        }
        val refined = refineGlobalDurationPartsForTesting(
            input = phonemes,
            chunks = parts,
            tokenDurations = durations,
            countTokens = { value -> KokoroTokenizer.tokenize(context, value).size },
            splitChunk = { part ->
                splitOversizedPhonemes(part)?.let { split -> split.head to split.tail }
            },
        ) ?: run {
            Log.w(
                TAG,
                "Global front has a duration core beyond packaged coverage and no bounded " +
                    "word-safe refinement; retaining bounded independent fallback",
            )
            return null
        }
        val boundaries = refined.boundaries.toMutableList()
        if (refined.refinements > 0 || refined.bridgeCoalescences > 0) {
            Log.i(
                TAG,
                "stage=global_front_duration_refine initial_parts=${parts.size} " +
                    "final_parts=${refined.chunks.size} splits=${refined.refinements} " +
                    "bridge_coalescences=${refined.bridgeCoalescences} " +
                    "cores=${boundaries.zipWithNext().joinToString { (start, end) ->
                        "${start.second}:${end.second}"
                    }}",
            )
        }
        if (boundaries.size > 2 && boundaries[1].second > GLOBAL_FRONT_OPENING_WINDOW_FRAMES) {
            val original = boundaries[1]
            val bounded = boundedDurationOpeningForTesting(
                input = phonemes,
                originalBoundary = original,
                tokenDurations = durations,
                maxFrames = GLOBAL_FRONT_OPENING_WINDOW_FRAMES,
                countTokens = { value -> KokoroTokenizer.tokenize(context, value).size },
                isSafeTail = { tail -> tail !in unsafeOpeningTailPhonemes },
            ) ?: run {
                Log.w(
                    TAG,
                    "Global front opening ${original.second} has no speech-safe B128 boundary; " +
                        "retaining bounded independent fallback",
                )
                return null
            }
            val expandedNextCore = boundaries.getOrNull(2)?.let { next ->
                next.second - bounded.second
            } ?: 0
            val boundedKeepsContext = expandedNextCore <=
                GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES - GLOBAL_FRONT_MIN_CONTINUATION_CONTEXT_FRAMES
            if (original.second > GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES || boundedKeepsContext) {
                boundaries[1] = bounded
                Log.i(
                    TAG,
                    "stage=global_front_replan opening=${original.first}:${original.second}->" +
                        "${bounded.first}:${bounded.second} next_core=$expandedNextCore",
                )
            } else {
                Log.i(
                    TAG,
                    "stage=global_front_replan opening=${original.second} retained_B192 " +
                        "because_B128_next_core=$expandedNextCore",
                )
            }
        }
        if (boundaries.size > 2 && boundaries[1].second < GLOBAL_FRONT_MIN_OPENING_RUNWAY_FRAMES) {
            val original = boundaries[1]
            val expanded = expandedDurationOpeningForTesting(
                input = phonemes,
                originalBoundary = original,
                stopBeforeChar = boundaries[2].first,
                tokenDurations = durations,
                minFrames = GLOBAL_FRONT_MIN_OPENING_RUNWAY_FRAMES,
                maxFrames = GLOBAL_FRONT_MAX_OPENING_CORE_FRAMES,
                countTokens = { value -> KokoroTokenizer.tokenize(context, value).size },
                isSafeTail = { tail -> tail !in unsafeOpeningTailPhonemes },
            )
            if (expanded != null) {
                boundaries[1] = expanded
                Log.i(
                    TAG,
                    "stage=global_front_runway opening=${original.first}:${original.second}->" +
                        "${expanded.first}:${expanded.second}",
                )
            } else {
                Log.i(
                    TAG,
                    "stage=global_front_runway opening=${original.second} retained " +
                        "no_safe_${GLOBAL_FRONT_MIN_OPENING_RUNWAY_FRAMES}_frame_boundary",
                )
            }
        }
        check(boundaries.last().second == globalFront.frames) {
            "Duration alignment ended at ${boundaries.last().second}/${globalFront.frames} frames"
        }
        val basePlans = boundaries.zipWithNext().mapIndexed { index, (start, end) ->
            val coreFrames = end.second - start.second
            check(coreFrames > 0) { "Empty global-front core at index=$index" }
            val windowFrames = if (index == 0) {
                globalOpeningWindowFramesForTesting(coreFrames)
            } else {
                globalContinuationWindowFramesForTesting(coreFrames)
            } ?: run {
                Log.w(
                    TAG,
                    "Global front core ${start.second}..${end.second} requires T=$coreFrames " +
                        "plus shared context beyond packaged coverage; " +
                        "retaining bounded independent fallback",
                )
                return null
            }
            if (windowFrames > GLOBAL_FRONT_CONTINUATION_WINDOW_FRAMES) {
                Log.i(
                    TAG,
                    "stage=global_front_window_expand core=${start.second}:${end.second} " +
                        "core_frames=$coreFrames window_frames=$windowFrames",
                )
            }
            val (contextStart, contextEnd) = globalContextWindowForTesting(
                sentenceFrames = globalFront.frames,
                coreStart = start.second,
                coreEnd = end.second,
                windowFrames = windowFrames,
                opening = index == 0,
            ) ?: run {
                Log.w(
                    TAG,
                    "Global front window B$windowFrames cannot reserve shared context for " +
                        "core ${start.second}..${end.second}; retaining bounded independent fallback",
                )
                return null
            }
            val contextFrames = contextEnd - contextStart
            check(contextFrames in coreFrames..windowFrames) {
                "Invalid global-front window $contextStart..$contextEnd for core ${start.second}..${end.second}"
            }
            val chunkPhonemes = phonemes.substring(start.first, end.first).trim().ifEmpty { phonemes }
            PlannedChunk(
                phonemes = chunkPhonemes,
                leadingSeam = if (index == 0) WaveformSeam.REQUEST_BOUNDARY else WaveformSeam.CONTINUATION,
                trailingSeam = if (index == boundaries.size - 2) {
                    WaveformSeam.REQUEST_BOUNDARY
                } else {
                    WaveformSeam.CONTINUATION
                },
                front = sliceGeneratorInputs(globalFront, contextStart, contextEnd),
                coreStartFrame = start.second - contextStart,
                coreFrames = coreFrames,
                globallyConditioned = true,
                cacheIdentity = "$phonemes\u0000global=$contextStart:$contextEnd:${start.second}:${end.second}",
            )
        }
        val plans = basePlans.toMutableList()
        for (index in 0 until plans.lastIndex) {
            val previous = plans[index]
            val next = plans[index + 1]
            val previousCoreFrames = checkNotNull(previous.coreFrames)
            val nextCoreFrames = checkNotNull(next.coreFrames)
            val previousRightContext = checkNotNull(previous.front).frames -
                previous.coreStartFrame - previousCoreFrames
            val nextLeftContext = next.coreStartFrame
            val overlapFrames = min(
                GLOBAL_JOIN_HALF_OVERLAP_FRAMES,
                min(previousRightContext, nextLeftContext),
            )
            if (overlapFrames > 0) {
                plans[index] = previous.copy(
                    trailingOverlapFrames = overlapFrames,
                    cacheIdentity = "${previous.cacheIdentity}\u0000overlap=0:$overlapFrames",
                )
                plans[index + 1] = next.copy(
                    leadingOverlapFrames = overlapFrames,
                    cacheIdentity = "${next.cacheIdentity}\u0000overlap=$overlapFrames:0",
                )
            } else {
                Log.w(TAG, "stage=global_join overlap_unavailable index=$index:${index + 1}")
            }
        }
        Log.i(
            TAG,
                "stage=global_front tokens=$tokenCount frames=${globalFront.frames} " +
                "duration_entries=${durations.size} elapsed_ms=${formatMillis(elapsedMillis(globalFrontStarted))} " +
                "cores=${plans.joinToString { plan ->
                    val core = checkNotNull(plan.coreFrames)
                    "${plan.front?.frames}:${plan.coreStartFrame}:$core:" +
                        "${plan.leadingOverlapFrames}:${plan.trailingOverlapFrames}"
                }}",
        )
        return plans
    }

    private fun sliceGeneratorInputs(source: GeneratorInputs, startFrame: Int, endFrame: Int): GeneratorInputs {
        require(startFrame in 0 until endFrame && endFrame <= source.frames)
        val frames = endFrame - startFrame
        return GeneratorInputs(
            conditioning = source.conditioning,
            prosody = cropLastDimension(source.prosody, 1, source.frames, startFrame, frames),
            decoder = cropLastDimension(source.decoder, 512, source.frames, startFrame, frames),
            frames = frames,
        )
    }

    private fun frontSpan(phonemes: String, voice: String, speed: Float): FrontedSpan {
        val tokenIds = KokoroTokenizer.tokenize(context, phonemes)
        require(tokenIds.size <= MAX_TOKENS - 2) { "Kokoro input exceeds $MAX_TOKENS tokens" }
        val padded = LongArray(tokenIds.size + 2)
        tokenIds.copyInto(padded, 1)
        val tokens = OnnxTensor.createTensor(environment, arrayOf(padded))
        val style = OnnxTensor.createTensor(environment, styles.style(voice, tokenIds.size))
        val pace = OnnxTensor.createTensor(environment, floatArrayOf(speed))
        return try {
            FrontedSpan(phonemes, runFront(mapOf("input_ids" to tokens, "style" to style, "speed" to pace)))
        } finally {
            tokens.close()
            style.close()
            pace.close()
        }
    }

    /**
     * Prepares only CPU-owned continuation inputs. The caller starts this after the opening's
     * CPU front/source stages and before its HTP run, so distinct CPU sessions can overlap HTP
     * without allowing two DSP sessions to execute concurrently.
     */
    private fun prepareQnnPrefix(plan: PlannedChunk, voice: String, speed: Float): PreparedQnnPrefix? {
        val started = SystemClock.elapsedRealtimeNanos()
        if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
        val front = plan.front ?: checkNotNull(frontSpan(plan.phonemes, voice, speed).front)
        val bucket = aotBucketForFrames(front.frames) ?: return null
        val sourceWasResident = synchronized(sessionLock) { bucket in sourceSpectrumSessions }
        val sourceSessionStarted = SystemClock.elapsedRealtimeNanos()
        val sourceSession = cpuSourceSpectrumSession(bucket)
        val sourceSessionMs = elapsedMillis(sourceSessionStarted)
        val sourceRunStarted = SystemClock.elapsedRealtimeNanos()
        val sourceSpectrum = runCpuSourceSpectrum(sourceSession, front, bucket)
        val sourceRunMs = elapsedMillis(sourceRunStarted)
        Log.i(
            TAG,
            "stage=chunk_prefix_ready index=1 T=${front.frames} B=$bucket " +
                "source_session_ms=${formatMillis(sourceSessionMs)} " +
                "source_run_ms=${formatMillis(sourceRunMs)} " +
                "elapsed_ms=${formatMillis(elapsedMillis(started))}",
        )
        return PreparedQnnPrefix(
            front = front,
            bucket = bucket,
            sourceSpectrum = sourceSpectrum,
            sourceWasResident = sourceWasResident,
            sourceSessionMs = sourceSessionMs,
            sourceRunMs = sourceRunMs,
        )
    }

    private fun synthesizePlannedChunk(
        plan: PlannedChunk,
        voice: String,
        speed: Float,
        preparedQnnPrefix: PreparedQnnPrefix? = null,
        onBeforeQnnRun: (() -> Unit)? = null,
    ): ByteArray = synchronized(sessionUseLock) {
        val chunkStarted = SystemClock.elapsedRealtimeNanos()
        if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
        val intendedIdentity = intendedBackendCacheIdentity()
        val key = CacheKey(
            plan.cacheIdentity,
            voice,
            java.lang.Float.floatToIntBits(speed),
            intendedIdentity,
            plan.leadingSeam,
            plan.trailingSeam,
        )
        val cachedPcm = cached(key)
        if (cachedPcm != null) {
            Log.i(
                TAG,
                "stage=chunk cache=hit bytes=${cachedPcm.size} " +
                    "elapsed_ms=${formatMillis(elapsedMillis(chunkStarted))}",
            )
            cachedPcm
        } else run {
            beginInferenceBackendTracking()
            val inferenceStarted = SystemClock.elapsedRealtimeNanos()
            val generated = infer(
                plan.phonemes,
                voice,
                speed,
                plan.front,
                preparedQnnPrefix = preparedQnnPrefix,
                onBeforeQnnRun = onBeforeQnnRun,
            )
            val inferenceMs = elapsedMillis(inferenceStarted)
            finalizeInferenceDiagnostics()
            val postStarted = SystemClock.elapsedRealtimeNanos()
            val core = plan.coreFrames?.let { coreFrames ->
                val extendedStartFrame = plan.coreStartFrame - plan.leadingOverlapFrames
                val extendedFrames = coreFrames + plan.leadingOverlapFrames + plan.trailingOverlapFrames
                val startSample = Math.multiplyExact(extendedStartFrame, FRAME_SAMPLES)
                val extendedSamples = Math.multiplyExact(extendedFrames, FRAME_SAMPLES)
                check(startSample >= 0 && startSample + extendedSamples <= generated.size) {
                    "Global core ${plan.coreStartFrame}:$coreFrames with overlap " +
                        "${plan.leadingOverlapFrames}:${plan.trailingOverlapFrames} exceeds " +
                        "${generated.size / FRAME_SAMPLES} frames"
                }
                generated.copyOfRange(startSample, startSample + extendedSamples)
            } ?: generated
            // Generator context is not a second utterance. Internal global-frame edges must stay
            // sample-contiguous and receive neither the old independent-render trims nor a
            // crossfade; only the two external Android-request edges retain bounded quiet padding.
            val leadingSamples = if (plan.globallyConditioned) {
                REQUEST_BOUNDARY_LEADING_SAMPLES.takeIf {
                    plan.leadingSeam == WaveformSeam.REQUEST_BOUNDARY
                }
            } else {
                leadingSamplesFor(plan.leadingSeam)
            }
            val trailingSamples = if (plan.globallyConditioned) {
                REQUEST_BOUNDARY_TRAILING_SAMPLES.takeIf {
                    plan.trailingSeam == WaveformSeam.REQUEST_BOUNDARY
                }
            } else {
                trailingSamplesFor(plan.trailingSeam)
            }
            val seamAdjusted = trimArtificialEdgeSilence(core, leadingSamples, trailingSamples)
            val encoding = encodePcm16WithDiagnostics(seamAdjusted)
            val encoded = encoding.pcm
            val postMs = elapsedMillis(postStarted)
            observedBackendCacheIdentity()?.let { actualIdentity ->
                cache(key.copy(backendIdentity = actualIdentity), encoded)
            }
            Log.i(
                TAG,
                "stage=chunk cache=miss phoneme_chars=${plan.phonemes.length} bytes=${encoded.size} " +
                    "global=${plan.globallyConditioned} core=${plan.coreStartFrame}:${plan.coreFrames ?: -1} " +
                    "overlap=${plan.leadingOverlapFrames}:${plan.trailingOverlapFrames} " +
                    "encoding_gain=${encoding.gain} robust_peak=${encoding.robustPeak} " +
                    "absolute_peak=${encoding.absolutePeak} limited_samples=${encoding.limitedSamples} " +
                    "inference_ms=${formatMillis(inferenceMs)} post_ms=${formatMillis(postMs)} " +
                    "elapsed_ms=${formatMillis(elapsedMillis(chunkStarted))}",
            )
            encoded
        }
    }

    private fun infer(
        phonemes: String,
        voice: String,
        speed: Float,
        preparedFront: GeneratorInputs? = null,
        frameSplitDepth: Int = 0,
        preparedQnnPrefix: PreparedQnnPrefix? = null,
        onBeforeQnnRun: (() -> Unit)? = null,
    ): FloatArray {
        val tokenIds = KokoroTokenizer.tokenize(context, phonemes)
        require(tokenIds.size <= MAX_TOKENS - 2) { "Kokoro input exceeds $MAX_TOKENS tokens" }
        val padded = LongArray(tokenIds.size + 2)
        tokenIds.copyInto(padded, 1)
        val tokens = OnnxTensor.createTensor(environment, arrayOf(padded))
        val style = OnnxTensor.createTensor(environment, styles.style(voice, tokenIds.size))
        val pace = OnnxTensor.createTensor(environment, floatArrayOf(speed))
        val runOptions = OrtSession.RunOptions()
        val started = SystemClock.elapsedRealtimeNanos()
        return try {
            // On a qualifying S24, run the exact CPU source-spectrum prefix,
            // the repaired fixed-shape neural vocoder on HTP, and the tiny
            // iSTFT suffix on CPU. All 1..640-frame durations map to a packaged
            // AOT bucket; longer inputs retain the verified q8 CPU fallback.
            if (shouldUseQnn()) {
                try {
                    val front = preparedQnnPrefix?.front ?: preparedFront ?: runFront(
                        mapOf("input_ids" to tokens, "style" to style, "speed" to pace),
                    )
                    val bucket = aotBucketForFrames(front.frames)
                    if (bucket != null) {
                        val (generated, qnn) = runV1QnnWithSharedFallback(
                            front,
                            bucket,
                            preparedPrefix = preparedQnnPrefix,
                            onBeforeQnnRun = onBeforeQnnRun,
                        )
                        check(generated.size >= front.frames * FRAME_SAMPLES) {
                            "v1 QNN iSTFT returned ${generated.size} samples for T=${front.frames}"
                        }
                        pcmHeadroomGain(generated)
                        recordQnnContextEvidence(qnn)
                        lastGeneratorBackend = InferenceBackend.QNN_HTP
                        val rtf = recordPerformance(InferenceBackend.QNN_HTP, started, generated.size)
                        recordSuccessfulGenerator(InferenceBackend.QNN_HTP, bucket, rtf, qnn)
                        return generated.copyOf(front.frames * FRAME_SAMPLES)
                    }
                } catch (problem: Exception) {
                    if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
                    Log.w(TAG, "v1 QNN acoustic path failed; retaining verified CPU v1 path", problem)
                    disableGeneratorBackend(InferenceBackend.QNN_HTP, "v1 acoustic runtime failure")
                }
            }
            // A prepared front is already the authoritative sentence-level duration/intonation
            // path. If HTP is unavailable or fails, preserve those exact tensors in the dynamic
            // split CPU generator instead of silently re-fronting an independent text fragment.
            if (preparedFront != null) {
                return runGenerator(preparedFront)
            }
            val holder = v1ModelSession()
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            val generated = holder.session.run(
                mapOf("input_ids" to tokens, "style" to style, "speed" to pace),
                runOptions,
            ).use { result ->
                check(result.size() >= 1) { "Kokoro v1.0 returned no audio output" }
                (result[0] as OnnxTensor).copyFloats()
            }
            pcmHeadroomGain(generated)
            lastGeneratorBackend = InferenceBackend.CPU
            val rtf = recordPerformance(InferenceBackend.CPU, started, generated.size)
            recordSuccessfulGenerator(InferenceBackend.CPU, tokenIds.size, rtf, holder)
            generated
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            tokens.close()
            style.close()
            pace.close()
        }
    }

    private fun runFront(inputs: Map<String, OnnxTensor>): GeneratorInputs {
        val holder = cpuFrontSession()
        val runOptions = OrtSession.RunOptions()
        registerActiveRun(runOptions)
        return try {
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            holder.session.run(inputs, runOptions).use { result ->
                check(result.size() == 4) { "CPU front-end returned ${result.size()} outputs" }
                check(result.iterator().asSequence().map { it.key }.toSet() ==
                    setOf(FRONT_CONDITIONING, FRONT_PROSODY, FRONT_DECODER, FRONT_TOKEN_DURATIONS)) {
                    "CPU front-end output contract changed"
                }
                val conditioning = result[0] as OnnxTensor
                val prosody = result[1] as OnnxTensor
                val decoder = result[2] as OnnxTensor
                val tokenDurations = result[3] as OnnxTensor
                val frames = (prosody.info as TensorInfo).shape.last().toInt()
                val decoderFrames = (decoder.info as TensorInfo).shape.last().toInt()
                check(frames > 0 && decoderFrames == frames) {
                    "CPU front-end frame mismatch: prosody=$frames decoder=$decoderFrames"
                }
                GeneratorInputs(
                    conditioning = conditioning.copyFloats(),
                    prosody = prosody.copyFloats(),
                    decoder = decoder.copyFloats(),
                    frames = frames,
                    tokenDurations = tokenDurations.copyLongs(),
                )
            }
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
        }
    }

    private fun runGenerator(inputs: GeneratorInputs): FloatArray {
        val useQnn = shouldUseQnn()
        val qnnBucket = if (useQnn) aotBucketForFrames(inputs.frames) else null
        if (qnnBucket != null) {
            try {
                val session = qnnGeneratorSession(qnnBucket)
                val started = SystemClock.elapsedRealtimeNanos()
                val audio = runGeneratorSession(session, inputs, qnnBucket)
                recordQnnContextEvidence(session)
                lastGeneratorBackend = InferenceBackend.QNN_HTP
                val rtf = recordPerformance(InferenceBackend.QNN_HTP, started, audio.size)
                recordSuccessfulGenerator(InferenceBackend.QNN_HTP, qnnBucket, rtf, session)
                return audio
            } catch (problem: Exception) {
                if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
                Log.w(TAG, "QNN HTP generator failed; retrying this utterance on CPU", problem)
                val reason = if (problem is InvalidGeneratorAudioException) "invalid generator audio" else "runtime failure"
                disableGeneratorBackend(InferenceBackend.QNN_HTP, reason)
            }
        } else if (useQnn) {
            Log.w(
                TAG,
                "Skipping QNN for T=${inputs.frames}; no qualified v1 acoustic bucket accepts it",
            )
        }

        if (shouldUseNnapi()) {
            try {
                val started = SystemClock.elapsedRealtimeNanos()
                val session = nnapiGeneratorSession()
                val audio = runGeneratorSession(session, inputs, inputs.frames)
                lastGeneratorBackend = InferenceBackend.NNAPI
                val rtf = recordPerformance(InferenceBackend.NNAPI, started, audio.size)
                recordSuccessfulGenerator(InferenceBackend.NNAPI, inputs.frames, rtf, session)
                return audio
            } catch (problem: Exception) {
                if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
                Log.w(TAG, "NNAPI generator failed; retrying this utterance on CPU", problem)
                val reason = if (problem is InvalidGeneratorAudioException) "invalid generator audio" else "runtime failure"
                disableGeneratorBackend(InferenceBackend.NNAPI, reason)
            }
        }

        lastGeneratorBackend = InferenceBackend.CPU
        val session = cpuGeneratorSession()
        val started = SystemClock.elapsedRealtimeNanos()
        val audio = runGeneratorSession(session, inputs, inputs.frames)
        val rtf = recordPerformance(InferenceBackend.CPU, started, audio.size)
        recordSuccessfulGenerator(InferenceBackend.CPU, inputs.frames, rtf, session)
        return audio
    }

    private fun runGeneratorSession(
        holder: SessionHolder,
        inputs: GeneratorInputs,
        paddedFrames: Int,
    ): FloatArray {
        check(paddedFrames >= inputs.frames) { "Generator bucket $paddedFrames is smaller than T=${inputs.frames}" }
        val conditioning = floatTensor(inputs.conditioning, longArrayOf(1, 128))
        val prosodyValues = if (paddedFrames == inputs.frames) inputs.prosody else inputs.prosody.copyOf(paddedFrames)
        val decoderValues = if (paddedFrames == inputs.frames) {
            inputs.decoder
        } else {
            padLastDimension(inputs.decoder, 512, inputs.frames, paddedFrames)
        }
        val prosody = floatTensor(prosodyValues, longArrayOf(1, 1, paddedFrames.toLong()))
        val decoder = floatTensor(decoderValues, longArrayOf(1, 512, paddedFrames.toLong()))
        val valid10 = 10 * inputs.frames
        val padded10 = 10 * paddedFrames
        val valid60 = 60 * inputs.frames + 1
        val padded60 = 60 * paddedFrames + 1
        val mask10 = floatTensor(validityMask(valid10, padded10), longArrayOf(1, 1, padded10.toLong()))
        val mask60 = floatTensor(validityMask(valid60, padded60), longArrayOf(1, 1, padded60.toLong()))
        val tensorInputs = mapOf(
            FRONT_CONDITIONING to conditioning,
            FRONT_PROSODY to prosody,
            FRONT_DECODER to decoder,
            VALID_MASK_10 to mask10,
            VALID_MASK_60 to mask60,
        )
        val runOptions = OrtSession.RunOptions()
        try {
            if (holder.qnnSource != null) {
                runOptions.addRunConfigEntry("qnn.perf_mode", "burst")
            }
            registerActiveRun(runOptions)
            if (closed || cancelRequested) throw CancellationException("Synthesis cancelled")
            val generated = holder.session.run(tensorInputs, runOptions).use { result ->
                (result[0] as OnnxTensor).copyFloats()
            }
            if (!generated.all { it.isFinite() }) {
                throw InvalidGeneratorAudioException("Generator produced non-finite audio")
            }
            val expectedSamples = inputs.frames * FRAME_SAMPLES
            check(generated.size >= expectedSamples) {
                "Generator returned ${generated.size} samples for T=${inputs.frames}; expected $expectedSamples"
            }
            val cropped = if (generated.size == expectedSamples) generated else generated.copyOf(expectedSamples)
            pcmHeadroomGain(cropped)
            return cropped
        } finally {
            unregisterActiveRun(runOptions)
            runOptions.close()
            tensorInputs.values.forEach { it.close() }
        }
    }

    private fun floatTensor(values: FloatArray, shape: LongArray): OnnxTensor {
        // Plugin EPs hand these buffers to a device backend. A heap-backed
        // FloatBuffer makes ORT stage an implicit JNI copy whose lifetime is
        // not part of the QNN execution contract. Keep the backing storage
        // direct and native-order until the tensor is closed so HTP always
        // sees the exact bytes supplied by Kotlin.
        val buffer = ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(values)
        buffer.flip()
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun OnnxTensor.copyFloats(): FloatArray {
        val buffer = floatBuffer
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun OnnxTensor.copyLongs(): LongArray {
        val buffer = longBuffer
        return LongArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun padLastDimension(
        source: FloatArray,
        channels: Int,
        frames: Int,
        paddedFrames: Int,
    ): FloatArray {
        require(source.size == channels * frames)
        return FloatArray(channels * paddedFrames).also { padded ->
            repeat(channels) { channel ->
                source.copyInto(
                    destination = padded,
                    destinationOffset = channel * paddedFrames,
                    startIndex = channel * frames,
                    endIndex = (channel + 1) * frames,
                )
            }
        }
    }

    private fun leftPadLastDimension(
        source: FloatArray,
        channels: Int,
        frames: Int,
        paddedFrames: Int,
    ): FloatArray {
        require(source.size == channels * frames)
        val offset = paddedFrames - frames
        return FloatArray(channels * paddedFrames).also { padded ->
            repeat(channels) { channel ->
                source.copyInto(
                    destination = padded,
                    destinationOffset = channel * paddedFrames + offset,
                    startIndex = channel * frames,
                    endIndex = (channel + 1) * frames,
                )
            }
        }
    }

    private fun cropLastDimension(
        source: FloatArray,
        channels: Int,
        sourceFrames: Int,
        startFrame: Int,
        frames: Int,
    ): FloatArray {
        require(source.size == channels * sourceFrames)
        require(startFrame >= 0 && frames > 0 && startFrame + frames <= sourceFrames)
        return FloatArray(channels * frames).also { cropped ->
            repeat(channels) { channel ->
                source.copyInto(
                    destination = cropped,
                    destinationOffset = channel * frames,
                    startIndex = channel * sourceFrames + startFrame,
                    endIndex = channel * sourceFrames + startFrame + frames,
                )
            }
        }
    }

    private fun validityMask(valid: Int, padded: Int): FloatArray {
        require(valid in 1..padded) { "Invalid generator mask extent $valid/$padded" }
        return FloatArray(padded).apply { fill(1f, 0, valid) }
    }

    private fun recordPerformance(backend: InferenceBackend, started: Long, samples: Int): Double {
        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
        val audioMs = samples * 1_000.0 / SAMPLE_RATE
        val rtf = elapsedMs / audioMs.coerceAtLeast(1.0)
        synchronized(performanceLock) {
            if (maxGeneratorRtfSinceReset.isNaN() || rtf > maxGeneratorRtfSinceReset) {
                maxGeneratorRtfSinceReset = rtf
            }
        }
        Log.i(
            TAG,
            "generator=$backend inference=${elapsedMs.toInt()}ms audio=${audioMs.toInt()}ms " +
                "rtf=${"%.2f".format(Locale.US, rtf)}",
        )
        if (backend == InferenceBackend.QNN_HTP) {
            if (rtf > 0.32) Log.w(TAG, "QNN HTP generator RTF is ${"%.2f".format(Locale.US, rtf)}; retaining HTP")
            return rtf
        }
        if (backend == InferenceBackend.CPU) {
            return rtf
        }
        if (rtf <= 0.32) {
            nnapiSlowRuns.set(0)
        } else if (nnapiSlowRuns.incrementAndGet() >= 2) {
            Log.w(TAG, "$backend generator is consistently slow; selecting CPU for later requests")
            disableGeneratorBackend(backend, "two consecutive RTF values above 0.32")
        }
        return rtf
    }

    private fun elapsedMillis(startedNanos: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - startedNanos) / 1_000_000.0

    private fun formatMillis(value: Double): String = String.format(Locale.US, "%.1f", value)

    private fun recordQnnContextEvidence(holder: SessionHolder) {
        val bucket = checkNotNull(holder.qnnBucket) { "QNN session did not identify its packaged bucket" }
        val source = checkNotNull(holder.qnnSource) { "QNN session did not identify its source" }
        val sha256 = checkNotNull(holder.qnnContextSha256) { "QNN session did not identify its context hash" }
        check(source == QNN_SESSION_SOURCE || source == QNN_FALLBACK_SESSION_SOURCE) {
            "Unexpected QNN session source $source"
        }
        check(sha256.matches(Regex("[0-9a-f]{64}"))) { "Invalid QNN context hash for T=$bucket" }
        val expectedSha256 = if (source == QNN_SESSION_SOURCE) {
            sharedQnnGroups.single { group -> group.entries.any { it.bucket == bucket } }.binary.sha256
        } else {
            when (bucket) {
                QNN_V1_B64_FRAMES -> BuildConfig.KOKORO_QNN_B64_CONTEXT_SHA256
                QNN_V1_B96_FRAMES -> BuildConfig.KOKORO_QNN_B96_CONTEXT_SHA256
                QNN_V1_B128_FRAMES -> BuildConfig.KOKORO_QNN_B128_CONTEXT_SHA256
                QNN_V1_B192_FRAMES -> BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256
                QNN_V1_B208_FRAMES -> BuildConfig.KOKORO_QNN_B208_CONTEXT_SHA256
                QNN_V1_B224_FRAMES -> BuildConfig.KOKORO_QNN_B224_CONTEXT_SHA256
                QNN_V1_B256_FRAMES -> BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256
                QNN_V1_B320_FRAMES -> BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256
                QNN_V1_B384_FRAMES -> BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256
                QNN_V1_B512_FRAMES -> BuildConfig.KOKORO_QNN_B512_CONTEXT_SHA256
                QNN_V1_B640_FRAMES -> BuildConfig.KOKORO_QNN_B640_CONTEXT_SHA256
                else -> error("Unexpected packaged QNN context bucket T=$bucket")
            }
        }
        check(sha256 == expectedSha256) { "Packaged QNN context identity changed for T=$bucket" }
        synchronized(performanceLock) {
            qnnContextSourcesSinceReset[bucket] = source
            qnnContextHashesSinceReset[bucket] = sha256
        }
    }

    private fun recordSuccessfulGenerator(
        backend: InferenceBackend,
        bucket: Int,
        rtf: Double,
        holder: SessionHolder,
    ) {
        synchronized(performanceLock) {
            generatorBackendsSinceReset += backend
            currentInferenceBackends += backend
        }
        synchronized(diagnosticsLock) {
            diagnosticBackend = backend.name
            diagnosticBucket = bucket
            diagnosticRtf = "%.4f".format(Locale.US, rtf)
            diagnosticContextSource = holder.qnnSource.orEmpty()
            diagnosticContextHashPrefix = holder.qnnContextSha256.orEmpty().take(12)
            diagnosticsDirty = true
        }
    }

    private fun disableGeneratorBackend(backend: InferenceBackend, reason: String) {
        // Inference may run on the single look-ahead producer. This re-entrant
        // session-use lock lets an inference failure disable its own backend,
        // while external quality/retry callbacks wait until ORT is no longer using it.
        synchronized(sessionUseLock) {
            synchronized(sessionLock) {
                val preferenceEditor = runtimePreferences().edit()
                when (backend) {
                    InferenceBackend.QNN_HTP -> {
                        sharedQnnDisabledForProcess = true
                        sharedV1QnnSessions.values.distinct().forEach { it.close() }
                        sharedV1QnnSessions.clear()
                        v1QnnAcousticSessions.values.forEach { it.close() }
                        v1QnnAcousticSessions.clear()
                        sourceSpectrumSessions.values.forEach { it.close() }
                        sourceSpectrumSessions.clear()
                        qnnGeneratorSessions.values.forEach { it.close() }
                        qnnGeneratorSessions.clear()
                        preferenceEditor.putBoolean(QNN_DISABLED_KEY, true)
                    }
                    InferenceBackend.NNAPI -> {
                        nnapiGeneratorSession?.close()
                        nnapiGeneratorSession = null
                        preferenceEditor.putBoolean(NNAPI_DISABLED_KEY, true)
                    }
                    InferenceBackend.CPU -> return
                }
                nnapiSlowRuns.set(0)
                lastGeneratorBackend = InferenceBackend.CPU
                val diagnosticFailure = "${backend.name}: $reason"
                synchronized(diagnosticsLock) {
                    diagnosticBackend = "${backend.name}_FAILED"
                    diagnosticBucket = null
                    diagnosticRtf = ""
                    diagnosticContextSource = ""
                    diagnosticContextHashPrefix = ""
                    diagnosticFailureReason = diagnosticFailure
                    diagnosticsDirty = true
                }
                preferenceEditor
                    .putString(DIAGNOSTIC_BACKEND_KEY, "${backend.name}_FAILED")
                    .putInt(DIAGNOSTIC_BUCKET_KEY, -1)
                    .putString(DIAGNOSTIC_RTF_KEY, "")
                    .putString(DIAGNOSTIC_CONTEXT_SOURCE_KEY, "")
                    .putString(DIAGNOSTIC_CONTEXT_HASH_KEY, "")
                    .putString(DIAGNOSTIC_FAILURE_KEY, diagnosticFailure)
                    .putLong(DIAGNOSTIC_TIMESTAMP_KEY, System.currentTimeMillis())
                    .apply()
                Log.w(TAG, "Persisted $backend fallback: $reason")
            }
        }
    }

    private fun runtimePreferences() =
        context.getSharedPreferences(RUNTIME_PREFERENCES, Context.MODE_PRIVATE)

    private fun intendedBackendCacheIdentity(): String = when {
        shouldUseQnn() -> backendCacheIdentity(InferenceBackend.QNN_HTP)
        shouldUseNnapi() -> backendCacheIdentity(InferenceBackend.NNAPI)
        else -> backendCacheIdentity(InferenceBackend.CPU)
    }

    private fun backendCacheIdentity(backend: InferenceBackend): String = when (backend) {
        InferenceBackend.QNN_HTP -> {
            val retryGeneration = runtimePreferences().getLong(QNN_RETRY_GENERATION_KEY, 0L)
            qnnPcmCacheIdentity(qnnContextCacheId, retryGeneration)
        }
        InferenceBackend.NNAPI ->
            "NNAPI:masked-fp32-v2:${BuildConfig.KOKORO_FRONT_MODEL_SHA256}:" +
                BuildConfig.KOKORO_GENERATOR_MODEL_SHA256
        InferenceBackend.CPU ->
            "CPU:split-fp32-v2:${BuildConfig.KOKORO_FRONT_MODEL_SHA256}:" +
                BuildConfig.KOKORO_GENERATOR_MODEL_SHA256
    }

    private fun beginInferenceBackendTracking() {
        synchronized(performanceLock) { currentInferenceBackends.clear() }
    }

    private fun observedBackendCacheIdentity(): String? {
        val backendNames = synchronized(performanceLock) { currentInferenceBackends.mapTo(linkedSetOf()) { it.name } }
        // Mixed-backend recursive/fallback audio is valid to play but is never
        // cached under a single accelerator identity.
        return soleCacheBackend(backendNames)?.let { backendCacheIdentity(InferenceBackend.valueOf(it)) }
    }

    private fun finalizeInferenceDiagnostics() {
        val backendNames = synchronized(performanceLock) { currentInferenceBackends.map { it.name }.sorted() }
        if (backendNames.size <= 1) return
        synchronized(diagnosticsLock) {
            diagnosticBackend = "MIXED:${backendNames.joinToString("+")}"
            diagnosticBucket = null
            diagnosticContextSource = ""
            diagnosticContextHashPrefix = ""
            diagnosticsDirty = true
        }
    }

    private fun beginUtteranceDiagnostics() {
        synchronized(diagnosticsLock) {
            diagnosticBackend = ""
            diagnosticBucket = null
            diagnosticRtf = ""
            diagnosticContextSource = ""
            diagnosticContextHashPrefix = ""
            diagnosticFailureReason = ""
            diagnosticsDirty = false
        }
    }

    private fun persistRuntimeDiagnostics() {
        val values = synchronized(diagnosticsLock) {
            if (!diagnosticsDirty) return
            diagnosticsDirty = false
            arrayOf(
                diagnosticBackend,
                diagnosticBucket?.toString().orEmpty(),
                diagnosticRtf,
                diagnosticContextSource,
                diagnosticContextHashPrefix,
                diagnosticFailureReason,
            )
        }
        runtimePreferences().edit()
            .putString(DIAGNOSTIC_BACKEND_KEY, values[0])
            .putInt(DIAGNOSTIC_BUCKET_KEY, values[1].toIntOrNull() ?: -1)
            .putString(DIAGNOSTIC_RTF_KEY, values[2])
            .putString(DIAGNOSTIC_CONTEXT_SOURCE_KEY, values[3])
            .putString(DIAGNOSTIC_CONTEXT_HASH_KEY, values[4])
            .putString(DIAGNOSTIC_FAILURE_KEY, values[5])
            .putLong(DIAGNOSTIC_TIMESTAMP_KEY, System.currentTimeMillis())
            .apply()
    }

    /** Visible to physical-device instrumentation so CPU fallback cannot masquerade as HTP. */
    internal fun activeBackendForTesting(): String = lastGeneratorBackend.name

    internal fun qnnBucketCountForTesting(): Int = synchronized(sessionLock) {
        (qnnGeneratorSessions.keys + v1QnnAcousticSessions.keys + sharedV1QnnSessions.keys).toSet().size
    }

    internal fun qnnBucketFramesForTesting(): Set<Int> =
        synchronized(sessionLock) {
            (qnnGeneratorSessions.keys + v1QnnAcousticSessions.keys + sharedV1QnnSessions.keys).toSet()
        }

    internal fun qnnContextCacheIdForTesting(): String = qnnContextCacheId

    internal fun resetGeneratorRtfForTesting() {
        synchronized(performanceLock) {
            maxGeneratorRtfSinceReset = Double.NaN
            generatorBackendsSinceReset.clear()
            currentInferenceBackends.clear()
            qnnContextSourcesSinceReset.clear()
            qnnContextHashesSinceReset.clear()
        }
    }

    internal fun maxGeneratorRtfForTesting(): Double =
        synchronized(performanceLock) { maxGeneratorRtfSinceReset }

    internal fun qnnContextSourcesForTesting(): Map<Int, String> =
        synchronized(performanceLock) { qnnContextSourcesSinceReset.toMap() }

    internal fun qnnContextHashesForTesting(): Map<Int, String> =
        synchronized(performanceLock) { qnnContextHashesSinceReset.toMap() }

    internal fun generatorBackendsForTesting(): Set<String> =
        synchronized(performanceLock) { generatorBackendsSinceReset.mapTo(linkedSetOf()) { it.name } }

    internal fun rejectQnnQualityForTesting(reason: String) {
        disableGeneratorBackend(InferenceBackend.QNN_HTP, reason)
    }

    private fun registerActiveRun(options: OrtSession.RunOptions) {
        activeRuns += options
    }

    private fun unregisterActiveRun(options: OrtSession.RunOptions) {
        activeRuns -= options
    }

    fun cancel() {
        synchronized(sessionLock) { cancelRequested = true }
        activeRuns.toList().forEach { options ->
            try {
                options.setTerminate(true)
            } catch (problem: Exception) {
                Log.w(TAG, "Unable to terminate the active ONNX Runtime run", problem)
            }
        }
    }

    private fun cached(key: CacheKey): ByteArray? = synchronized(cacheLock) { pcmCache[key] }

    private fun cache(key: CacheKey, pcm: ByteArray) = synchronized(cacheLock) {
        pcmCache.put(key, pcm)?.let { pcmCacheSize -= it.size }
        pcmCacheSize += pcm.size
        val iterator = pcmCache.entries.iterator()
        while (pcmCacheSize > PCM_CACHE_BYTES && iterator.hasNext()) {
            pcmCacheSize -= iterator.next().value.size
            iterator.remove()
        }
    }

    private fun splitForLowLatency(input: String, speed: Float): List<String> {
        val (firstLimit, followingLimit) = chunkTokenLimitsForSpeed(speed)
        return splitForLatencyForTesting(input, firstLimit, followingLimit) { value ->
            KokoroTokenizer.tokenize(context, value).size
        }
    }

    /** Reports existing near-silent PCM at an edge; it never trims, extends, or inserts audio. */
    private fun pcmEdgeQuietSamples(pcm: ByteArray, leading: Boolean): Int {
        if (pcm.size < Short.SIZE_BYTES) return 0
        var peak = 0
        var index = 0
        while (index + 1 < pcm.size) {
            val sample = ((pcm[index + 1].toInt() shl 8) or (pcm[index].toInt() and 0xff)).toShort().toInt()
            peak = max(peak, abs(sample))
            index += Short.SIZE_BYTES
        }
        if (peak == 0) return pcm.size / Short.SIZE_BYTES
        val threshold = max(1, (peak * EDGE_ACTIVE_RELATIVE_RMS).roundToInt())
        val samples = pcm.size / Short.SIZE_BYTES
        var quiet = 0
        var sampleIndex = if (leading) 0 else samples - 1
        while (sampleIndex in 0 until samples) {
            val byteIndex = sampleIndex * Short.SIZE_BYTES
            val sample = ((pcm[byteIndex + 1].toInt() shl 8) or (pcm[byteIndex].toInt() and 0xff)).toShort().toInt()
            if (abs(sample) > threshold) break
            quiet++
            sampleIndex += if (leading) 1 else -1
        }
        return quiet
    }

    /**
     * Plans source text first, then maps only those selected source boundaries onto the already
     * produced full-sentence G2P stream. Mapping failure falls back to the existing phoneme
     * planner rather than independently phonemizing/rendering a fragment.
     */
    private fun sourcePlannedPhonemeParts(
        sourceText: String,
        phonemes: String,
        locale: Locale,
        speed: Float,
    ): List<String> {
        val (firstLimit, followingLimit) = chunkTokenLimitsForSpeed(speed)
        val sourceWords = sourceWordPattern.findAll(sourceText).count().coerceAtLeast(1)
        val fullTokens = KokoroTokenizer.tokenize(context, phonemes).size.coerceAtLeast(1)
        fun estimatedTokens(value: String): Int =
            (sourceWordPattern.findAll(value).count() * fullTokens.toFloat() / sourceWords)
                .roundToInt().coerceAtLeast(1)
        val sourceParts = splitSourceForLatencyForTesting(
            sourceText,
            firstLimit,
            followingLimit,
            ::estimatedTokens,
        )
        if (sourceParts.size < 2) return splitForLowLatency(phonemes, speed)

        val prefixCounts = mutableListOf<Int>()
        val prefix = StringBuilder()
        sourceParts.dropLast(1).forEach { sourcePart ->
            if (prefix.isNotEmpty()) prefix.append(' ')
            prefix.append(sourcePart)
            val prefixPhonemes = phonemizer.phonemize(prefix.toString(), locale)
            prefixCounts += KokoroTokenizer.tokenize(context, prefixPhonemes).size
        }
        return mapSourceBoundariesToPhonemesForTesting(
            phonemes,
            prefixCounts,
        ) { value -> KokoroTokenizer.tokenize(context, value).size }
            ?: run {
                Log.i(TAG, "stage=source_boundary_map fallback=phoneme_planner source_parts=${sourceParts.size}")
                splitForLowLatency(phonemes, speed)
            }
    }

    private fun splitOversizedPhonemes(input: String): PhonemeSplit? {
        val totalTokens = KokoroTokenizer.tokenize(context, input).size
        if (totalTokens < 2) return null
        val target = max(1, totalTokens / 2)
        val boundary = preferredTextBoundary(
            input = input,
            targetTokens = target,
            hardTokens = max(target, totalTokens * 2 / 3),
            minimumTailTokens = 1,
        ) { value -> KokoroTokenizer.tokenize(context, value).size }
            ?: fallbackTokenBoundary(input, target, 1) { value ->
                KokoroTokenizer.tokenize(context, value).size
            }
            ?: return null
        val head = input.substring(0, boundary).trim()
        val tail = input.substring(boundary).trim()
        if (head.isEmpty() || tail.isEmpty()) return null
        val headTokens = KokoroTokenizer.tokenize(context, head).size
        val tailTokens = KokoroTokenizer.tokenize(context, tail).size
        if (headTokens !in 1 until totalTokens || tailTokens !in 1 until totalTokens) return null
        return PhonemeSplit(head, tail, waveformSeamAfter(head))
    }

    private fun installAsset(name: String, installedName: String = name): File =
        synchronized(packagedAssetInstallLock) {
            val destination = File(context.noBackupFilesDir, installedName)
            if (destination.isFile && destination.length() > 1_000_000) return@synchronized destination
            val temporary = File(destination.parentFile, "$installedName.$instanceId.part")
            try {
                context.assets.open(name).use { input ->
                    FileOutputStream(temporary).use { output -> input.copyTo(output) }
                }
                if (!temporary.renameTo(destination)) {
                    temporary.copyTo(destination, overwrite = true)
                }
                destination
            } finally {
                if (temporary.exists() && !temporary.delete()) {
                    Log.w(TAG, "Unable to delete private asset-install partial $temporary")
                }
            }
        }

    override fun close() {
        val firstClose = synchronized(sessionLock) {
            if (closed) false else {
                closed = true
                true
            }
        }
        if (!firstClose) return

        cancel()
        // synthesizeChunks owns this monitor for its full lifetime. Waiting here
        // guarantees no session is closed while ORT is still executing it.
        synchronized(this) {
            synchronized(sessionLock) {
                frontSession?.close()
                frontSession = null
                v1ModelSession?.close()
                v1ModelSession = null
                cpuGeneratorSession?.close()
                cpuGeneratorSession = null
                nnapiGeneratorSession?.close()
                nnapiGeneratorSession = null
                v1IstftSession?.close()
                v1IstftSession = null
                v1QnnAcousticSessions.values.forEach { it.close() }
                v1QnnAcousticSessions.clear()
                sourceSpectrumSessions.values.forEach { it.close() }
                sourceSpectrumSessions.clear()
                sharedV1QnnSessions.values.distinct().forEach { it.close() }
                sharedV1QnnSessions.clear()
                qnnGeneratorSessions.values.forEach { it.close() }
                qnnGeneratorSessions.clear()
                qnnEpDevices = null
                mappedModels.clear()
            }
            synchronized(cacheLock) {
                pcmCache.clear()
                pcmCacheSize = 0
            }
        }
    }
}

/** Loads the model's own ID mapping, rather than inferring IDs from character order. */
internal object KokoroTokenizer {
    @Volatile private var vocabulary: Map<Char, Long>? = null

    fun tokenize(context: Context, value: String): LongArray {
        val loaded = vocabulary ?: synchronized(this) {
            vocabulary ?: context.assets.open(BuildConfig.KOKORO_TOKENIZER_ASSET).bufferedReader().use { reader ->
                val json = JSONObject(reader.readText()).getJSONObject("model").getJSONObject("vocab")
                buildMap {
                    val keys = json.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        if (key.length == 1) put(key[0], json.getLong(key))
                    }
                }.also { vocabulary = it }
            }
        }
        return tokenize(loaded, value)
    }

    internal fun tokenize(vocabulary: Map<Char, Long>, value: String): LongArray =
        value.mapNotNull { vocabulary[it] }.toLongArray()
}

/** Reads the current v1.0 raw float32 [510, 256] voice profiles. */
internal class VoiceStyleStore(private val context: Context) {
    private val cache = ConcurrentHashMap<String, FloatArray>()

    fun style(id: String, phonemeCount: Int): Array<FloatArray> {
        val all = cache.getOrPut(id) { read("voices_v1/$id.bin") }
        val row = min(509, max(0, phonemeCount)) * 256
        return arrayOf(all.copyOfRange(row, row + 256))
    }

    private fun read(path: String): FloatArray = context.assets.open(path).use { input ->
        val bytes = input.readBytes()
        require(bytes.size == 510 * 256 * Float.SIZE_BYTES) {
            "Unexpected Kokoro v1.0 voice tensor size for $path: ${bytes.size}"
        }
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { floats ->
            FloatArray(floats.remaining()).also { floats.get(it) }
        }
    }
}
