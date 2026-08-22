package com.local.kokorotts

import android.os.SystemClock
import android.os.Bundle
import android.os.Build
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.Collections
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

@RunWith(AndroidJUnit4::class)
class KokoroRuntimeInstrumentedTest {
    private data class RenderResult(
        val samples: ShortArray,
        val elapsedMs: Long,
        val chunkSamples: List<Int>,
    )

    private data class CaseSpec(
        val voice: KokoroVoice,
        val text: String,
        val speed: Float = 1.0f,
        val requiredQnnBucket: Int? = null,
    )

    private data class AudioMetrics(
        val durationDelta: Double,
        val peak: Int,
        val nrmse: Double,
        val correlation: Double,
        val roughnessRatio: Double,
    )

    private data class CaseReceipt(
        val voice: String,
        val textSha256: String,
        val reference: RenderResult,
        val candidate: RenderResult,
        val metrics: AudioMetrics,
        val qnnMaxGeneratorRtf: Double,
        val speed: Float,
        val qnnBackends: Set<String>,
        val qnnContextSources: Map<Int, String>,
        val qnnContextHashes: Map<Int, String>,
    )

    @Test
    fun qnnHtpMatchesClearCpuReference() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val marker = File(context.filesDir, "qnn-audio-gate.txt")
        val pendingMarker = File(context.filesDir, "qnn-audio-gate.txt.part")
        assertTrue("Unable to delete stale QNN validation marker", !marker.exists() || marker.delete())
        assertTrue("Unable to delete stale partial QNN marker", !pendingMarker.exists() || pendingMarker.delete())
        // A prior failed qualification must not prevent the explicit physical
        // rerun from exercising HTP. This rotates the cache generation too.
        KokoroSynthesizer.requestQnnRetry(context)

        assumeTrue("This APK was not built with the public Maven QNN runtime", BuildConfig.QNN_EP_INCLUDED)
        assumeTrue("This APK does not contain the embedded QNN fallback set", BuildConfig.KOKORO_QNN_AOT_INCLUDED)
        assertFalse("Android must use self-contained embedded contexts", BuildConfig.KOKORO_QNN_SHARED_INCLUDED)
        assertEquals("Unexpected QNN runtime", "com.qualcomm.qti:qnn-runtime:2.48.0", BuildConfig.QNN_RUNTIME_COORDINATE)
        assertEquals(
            "Unexpected QNN context producer",
            "QAIRT_2.48.40_V1_POW_X2_MUL_CPU_SOURCE_SPECTRUM_PER_BUCKET_S24_QUALIFIED",
            BuildConfig.KOKORO_QNN_CONTEXT_PRODUCER,
        )
        assertFalse(
            "QNN validation is forbidden on an emulator",
            Build.FINGERPRINT.contains("generic", ignoreCase = true) ||
                Build.MODEL.contains("emulator", ignoreCase = true),
        )
        assumeTrue(
            "The packaged QNN candidate targets Snapdragon 8 Gen 3 (SM8650)",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true),
        )
        assumeTrue(
            "The release device gate requires a Galaxy S24 Ultra (SM-S928 family)",
            Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val cases = listOf(
            CaseSpec(
                VoiceCatalog.find("af_heart"),
                "Hi.",
                speed = 2.5f,
                requiredQnnBucket = 64,
            ),
            CaseSpec(
                VoiceCatalog.find("af_heart"),
                "Ready.",
                speed = 1.3f,
                requiredQnnBucket = 96,
            ),
            CaseSpec(
                VoiceCatalog.find("bf_emma"),
                "Ready.",
                speed = 1.0f,
            ),
            CaseSpec(
                VoiceCatalog.find("am_adam"),
                "This is an example of speech synthesis in English.",
                speed = 1.3f,
            ),
            CaseSpec(
                VoiceCatalog.find("bm_george"),
                "Numbers like 24, 3.14159, and 2026 should sound natural, not noisy.",
                speed = 1.3f,
            ),
        )
        var observedBucketSessions = 0
        val observedBucketFrames = sortedSetOf<Int>()
        val observedContextSources = sortedMapOf<Int, String>()
        val observedContextHashes = sortedMapOf<Int, String>()
        val receipts = mutableListOf<CaseReceipt>()
        var contextCacheId = ""
        KokoroSynthesizer(context, BackendPreference.CPU).use { cpu ->
            KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { qnn ->
                cpu.prepare()
                qnn.prepare()
                contextCacheId = qnn.qnnContextCacheIdForTesting()
                cases.forEach { case ->
                    // Retain the established monolithic clear-CPU oracle for genuinely short
                    // one-window cases. Progressive sentences use the cadence-specific one-front,
                    // one-generator oracle so their reference cannot inherit independent chunks.
                    val reference = if (case.text.length <= 16) {
                        render(cpu, case.text, case.voice, case.speed)
                    } else {
                        renderUnsplitGlobal(cpu, case.text, case.voice, case.speed)
                    }
                    qnn.resetGeneratorRtfForTesting()
                    val candidate = render(qnn, case.text, case.voice, case.speed)
                    assertEquals("QNN inference fell back to CPU", "QNN_HTP", qnn.activeBackendForTesting())
                    val qnnMaxGeneratorRtf = qnn.maxGeneratorRtfForTesting()
                    assertTrue("QNN generator did not publish a finite RTF", qnnMaxGeneratorRtf.isFinite())
                    val qnnBackends = qnn.generatorBackendsForTesting()
                    assertEquals("A generator chunk did not run on QNN HTP", setOf("QNN_HTP"), qnnBackends)
                    val qnnContextSources = qnn.qnnContextSourcesForTesting()
                    val qnnContextHashes = qnn.qnnContextHashesForTesting()
                    assertTrue("QNN inference did not report a packaged context", qnnContextSources.isNotEmpty())
                    assertEquals("QNN source/hash buckets differ", qnnContextSources.keys, qnnContextHashes.keys)
                    assertTrue(
                        "QNN inference used a non-packaged source: $qnnContextSources",
                        qnnContextSources.values.all { it == KokoroSynthesizer.qnnSessionSourceForTesting() },
                    )
                    case.requiredQnnBucket?.let { requiredBucket ->
                        assertTrue(
                            "Deterministic case did not execute packaged T=$requiredBucket",
                            requiredBucket in qnnContextSources,
                        )
                    }
                    qnnContextHashes.forEach { (bucket, hash) ->
                        val expected = when (bucket) {
                            64 -> BuildConfig.KOKORO_QNN_B64_CONTEXT_SHA256
                            96 -> BuildConfig.KOKORO_QNN_B96_CONTEXT_SHA256
                            128 -> BuildConfig.KOKORO_QNN_B128_CONTEXT_SHA256
                            192 -> BuildConfig.KOKORO_QNN_B192_CONTEXT_SHA256
                            208 -> BuildConfig.KOKORO_QNN_B208_CONTEXT_SHA256
                            224 -> BuildConfig.KOKORO_QNN_B224_CONTEXT_SHA256
                            256 -> BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256
                            320 -> BuildConfig.KOKORO_QNN_B320_CONTEXT_SHA256
                            384 -> BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256
                            512 -> BuildConfig.KOKORO_QNN_B512_CONTEXT_SHA256
                            640 -> BuildConfig.KOKORO_QNN_B640_CONTEXT_SHA256
                            else -> error("Unexpected QNN AOT bucket T=$bucket")
                        }
                        assertEquals("Packaged QNN context hash changed for T=$bucket", expected, hash)
                    }
                    observedContextSources.putAll(qnnContextSources)
                    observedContextHashes.putAll(qnnContextHashes)
                    try {
                        receipts += CaseReceipt(
                            voice = case.voice.id,
                            textSha256 = sha256(case.text.toByteArray(Charsets.UTF_8)),
                            reference = reference,
                            candidate = candidate,
                            metrics = assertAudioSafe(case.voice.id, reference.samples, candidate.samples),
                            qnnMaxGeneratorRtf = qnnMaxGeneratorRtf,
                            speed = case.speed,
                            qnnBackends = qnnBackends,
                            qnnContextSources = qnnContextSources,
                            qnnContextHashes = qnnContextHashes,
                        )
                    } catch (qualityFailure: AssertionError) {
                        qnn.rejectQnnQualityForTesting("physical-device audio gate failed for ${case.voice.id}")
                        throw qualityFailure
                    }
                    observedBucketSessions = maxOf(observedBucketSessions, qnn.qnnBucketCountForTesting())
                    observedBucketFrames += qnn.qnnBucketFramesForTesting()
                    assertTrue(
                        "QNN generator used a non-packaged bucket: $observedBucketFrames",
                        observedBucketFrames.all { it in setOf(64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640) },
                    )
                    assertTrue("Embedded QNN LRU exceeded its three-session cap", observedBucketSessions <= 3)
                }
            }
        }
        assertTrue("Short/common/progressive sentence QNN routes were not all exercised: $observedBucketFrames",
            observedBucketFrames.containsAll(setOf(64, 96, 128, 192)))
        assertEquals("Missing packaged QNN source evidence", observedContextSources.keys, observedContextHashes.keys)
        val installedApk = File(context.applicationInfo.sourceDir)
        val receiptLines = buildList {
            add("receiptVersion=6")
            add("result=PASSED")
            add("testedAtUtc=${Instant.now()}")
            add("applicationId=${BuildConfig.APPLICATION_ID}")
            add("versionCode=${BuildConfig.VERSION_CODE}")
            add("versionName=${BuildConfig.VERSION_NAME}")
            add("apkBytes=${installedApk.length()}")
            add("apkSha256=${sha256(installedApk)}")
            add("frontModelAsset=${BuildConfig.KOKORO_FRONT_MODEL_ASSET}")
            add("frontModelSha256=${BuildConfig.KOKORO_FRONT_MODEL_SHA256}")
            add("generatorModelAsset=${BuildConfig.KOKORO_GENERATOR_MODEL_ASSET}")
            add("generatorModelSha256=${BuildConfig.KOKORO_GENERATOR_MODEL_SHA256}")
            add("ortCoordinate=${BuildConfig.ORT_RUNTIME_COORDINATE}")
            add("qnnProviderCoordinate=${BuildConfig.QNN_PROVIDER_COORDINATE}")
            add("qnnRuntimeCoordinate=${BuildConfig.QNN_RUNTIME_COORDINATE}")
            add("qnnContextCacheId=$contextCacheId")
            add("qnnContextProducer=${BuildConfig.KOKORO_QNN_CONTEXT_PRODUCER}")
            add("qnnSourceSpectrumManifest=${BuildConfig.KOKORO_QNN_SOURCE_SPECTRUM_MANIFEST}")
            add("qnnSessionSource=${KokoroSynthesizer.qnnSessionSourceForTesting()}")
            add("qnnPerformancePolicy=${KokoroSynthesizer.qnnPerformancePolicyForTesting()}")
            add("qnnPrecisionPolicy=${KokoroSynthesizer.qnnPrecisionPolicyForTesting()}")
            add("qnnAssignmentPolicy=${KokoroSynthesizer.qnnAssignmentPolicyForTesting()}")
            add("qnnB256ContextAsset=${BuildConfig.KOKORO_QNN_B256_CONTEXT_ASSET}")
            add("qnnB256ContextSha256=${BuildConfig.KOKORO_QNN_B256_CONTEXT_SHA256}")
            add("qnnB384ContextAsset=${BuildConfig.KOKORO_QNN_B384_CONTEXT_ASSET}")
            add("qnnB384ContextSha256=${BuildConfig.KOKORO_QNN_B384_CONTEXT_SHA256}")
            add("qnnObservedContextSources=${observedContextSources.receiptMap()}")
            add("qnnObservedContextHashes=${observedContextHashes.receiptMap()}")
            add("deviceFingerprint=${Build.FINGERPRINT}")
            add("deviceManufacturer=${Build.MANUFACTURER}")
            add("deviceModel=${Build.MODEL}")
            add("deviceName=${Build.DEVICE}")
            add("deviceSoc=${Build.SOC_MODEL}")
            add("deviceSdk=${Build.VERSION.SDK_INT}")
            add("deviceAbis=${Build.SUPPORTED_ABIS.joinToString(",")}")
            add("backend=QNN_HTP")
            add("cases=${receipts.size}")
            add("buckets=${observedBucketFrames.joinToString(",")}")
            add("maxBucketSessions=$observedBucketSessions")
            receipts.forEachIndexed { index, receipt ->
                val prefix = "case.$index"
                add("$prefix.voice=${receipt.voice}")
                add("$prefix.textSha256=${receipt.textSha256}")
                add("$prefix.speed=${receipt.speed}")
                add("$prefix.referenceSamples=${receipt.reference.samples.size}")
                add("$prefix.candidateSamples=${receipt.candidate.samples.size}")
                add("$prefix.cpuElapsedMs=${receipt.reference.elapsedMs}")
                add("$prefix.qnnElapsedMs=${receipt.candidate.elapsedMs}")
                add("$prefix.qnnMaxGeneratorRtf=${receipt.qnnMaxGeneratorRtf.receiptDecimal()}")
                add("$prefix.qnnBackends=${receipt.qnnBackends.sorted().joinToString(",")}")
                add("$prefix.qnnContextSources=${receipt.qnnContextSources.toSortedMap().receiptMap()}")
                add("$prefix.qnnContextHashes=${receipt.qnnContextHashes.toSortedMap().receiptMap()}")
                add("$prefix.durationDelta=${receipt.metrics.durationDelta.receiptDecimal()}")
                add("$prefix.peak=${receipt.metrics.peak}")
                add("$prefix.nrmse=${receipt.metrics.nrmse.receiptDecimal()}")
                add("$prefix.correlation=${receipt.metrics.correlation.receiptDecimal()}")
                add("$prefix.roughnessRatio=${receipt.metrics.roughnessRatio.receiptDecimal()}")
            }
        }
        pendingMarker.writeText(receiptLines.joinToString("\n", postfix = "\n"), Charsets.UTF_8)
        assertTrue("Unable to publish QNN validation marker atomically", pendingMarker.renameTo(marker))
        println(
            "KOKORO_QNN_AUDIO_GATE PASSED cases=${cases.size} " +
                "buckets=$observedBucketFrames max_bucket_sessions=$observedBucketSessions " +
                "apk_sha256=${receiptLines.first { it.startsWith("apkSha256=") }.substringAfter('=')}",
        )
    }

    private fun render(
        synthesizer: KokoroSynthesizer,
        text: String,
        voice: KokoroVoice,
        speed: Float,
    ): RenderResult {
        val output = ByteArrayOutputStream()
        val chunkSamples = mutableListOf<Int>()
        val started = SystemClock.elapsedRealtime()
        synthesizer.synthesizeChunks(text, voice, speed) { pcm ->
            output.write(pcm)
            chunkSamples += pcm.size / Short.SIZE_BYTES
            true
        }
        val elapsedMs = SystemClock.elapsedRealtime() - started
        val samples = decodePcm16(output.toByteArray())
        return RenderResult(samples, elapsedMs, chunkSamples)
    }

    private fun renderUnsplitGlobal(
        synthesizer: KokoroSynthesizer,
        text: String,
        voice: KokoroVoice,
        speed: Float,
    ): RenderResult {
        val capture = synthesizer.captureUnsplitGlobalFrontForTesting(text, voice, speed)
        val bytes = KokoroSynthesizer.encodePcm16(capture.samples)
        return RenderResult(
            samples = decodePcm16(bytes),
            elapsedMs = capture.elapsedMs,
            chunkSamples = listOf(bytes.size / Short.SIZE_BYTES),
        )
    }

    private fun decodePcm16(bytes: ByteArray): ShortArray =
        ShortArray(bytes.size / Short.SIZE_BYTES) { index ->
            val offset = index * 2
            ((bytes[offset].toInt() and 0xff) or (bytes[offset + 1].toInt() shl 8)).toShort()
        }

    private fun encodePcm16(samples: ShortArray): ByteArray =
        ByteArray(samples.size * Short.SIZE_BYTES).also { bytes ->
            samples.forEachIndexed { index, sample ->
                val value = sample.toInt()
                bytes[index * 2] = value.toByte()
                bytes[index * 2 + 1] = (value ushr 8).toByte()
            }
        }

    private fun assertAudioSafe(label: String, reference: ShortArray, candidate: ShortArray): AudioMetrics {
        val durationDelta = kotlin.math.abs(candidate.size - reference.size).toDouble() / reference.size.coerceAtLeast(1)
        assertTrue("$label duration drift=$durationDelta", durationDelta <= 0.03)
        val length = minOf(reference.size, candidate.size)
        // The deterministic B64 route deliberately uses "Hi." at 2.5x and is
        // shorter than the former 10k-sample floor after edge-silence trimming.
        assertTrue("$label produced too little audio", length > 2_400)
        var refEnergy = 0.0
        var diffEnergy = 0.0
        var sumRef = 0.0
        var sumCandidate = 0.0
        var peak = 0
        for (i in 0 until length) {
            val ref = reference[i].toDouble()
            val value = candidate[i].toDouble()
            refEnergy += ref * ref
            diffEnergy += (value - ref) * (value - ref)
            sumRef += ref
            sumCandidate += value
            peak = maxOf(peak, kotlin.math.abs(candidate[i].toInt()))
        }
        val meanRef = sumRef / length
        val meanCandidate = sumCandidate / length
        var covariance = 0.0
        var varianceRef = 0.0
        var varianceCandidate = 0.0
        var roughReference = 0.0
        var roughCandidate = 0.0
        for (i in 0 until length) {
            val centeredRef = reference[i] - meanRef
            val centeredCandidate = candidate[i] - meanCandidate
            covariance += centeredRef * centeredCandidate
            varianceRef += centeredRef * centeredRef
            varianceCandidate += centeredCandidate * centeredCandidate
            if (i >= 2) {
                val refSecond = reference[i] - 2.0 * reference[i - 1] + reference[i - 2]
                val candidateSecond = candidate[i] - 2.0 * candidate[i - 1] + candidate[i - 2]
                roughReference += refSecond * refSecond
                roughCandidate += candidateSecond * candidateSecond
            }
        }
        val nrmse = kotlin.math.sqrt(diffEnergy / refEnergy.coerceAtLeast(1.0))
        val correlation = covariance / kotlin.math.sqrt((varianceRef * varianceCandidate).coerceAtLeast(1.0))
        val roughnessRatio = (roughCandidate / varianceCandidate.coerceAtLeast(1.0)) /
            (roughReference / varianceRef.coerceAtLeast(1.0)).coerceAtLeast(1e-9)
        val rmsRatio = kotlin.math.sqrt(varianceCandidate / varianceRef.coerceAtLeast(1.0))
        println("KOKORO_QNN_CASE $label nrmse=$nrmse correlation=$correlation roughness_ratio=$roughnessRatio")
        assertTrue("$label clipping peak=$peak", peak < (Short.MAX_VALUE * 0.98).toInt())
        assertTrue("$label is effectively static", varianceCandidate / length >= 4.0)
        assertTrue("$label RMS ratio=$rmsRatio", rmsRatio in 0.5..2.0)
        assertTrue("$label excess high-frequency roughness=$roughnessRatio", roughnessRatio <= 1.25)
        return AudioMetrics(durationDelta, peak, nrmse, correlation, roughnessRatio)
    }

    private fun Double.receiptDecimal(): String = String.format(Locale.US, "%.9f", this)

    private fun Map<Int, String>.receiptMap(): String =
        entries.joinToString(",") { (bucket, value) -> "$bucket:$value" }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().toHex()
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(Locale.US, it) }

    /** Captures the exact CPU and HTP PCM without sending either stream to an
     * Android audio sink. This is intentionally a one-case diagnostic so a
     * failed physical audio gate leaves inspectable evidence on the device. */
    @Test
    fun captureQnnAndCpuPcmWithoutPlayback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("QNN runtime is not packaged", BuildConfig.QNN_EP_INCLUDED && BuildConfig.KOKORO_QNN_AOT_INCLUDED)
        assumeTrue(
            "Capture requires the physical Galaxy S24 Ultra target",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true) &&
                Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val text = "This is an example of speech synthesis in English."
        val voice = VoiceCatalog.find("af_heart")
        val speed = InstrumentationRegistry.getArguments().getString("captureSpeed")?.toFloatOrNull() ?: 1.3f
        val reference = KokoroSynthesizer(context, BackendPreference.CPU).use { cpu ->
            cpu.prepare()
            render(cpu, text, voice, speed)
        }
        val candidate: RenderResult
        val backend: String
        val buckets: Set<Int>
        KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { qnn ->
            qnn.prepare()
            qnn.resetGeneratorRtfForTesting()
            candidate = render(qnn, text, voice, speed)
            backend = qnn.activeBackendForTesting()
            buckets = qnn.qnnContextSourcesForTesting().keys
        }
        fun writePcm(file: File, samples: ShortArray) {
            val bytes = ByteArray(samples.size * 2)
            samples.forEachIndexed { index, sample ->
                val value = sample.toInt()
                bytes[index * 2] = value.toByte()
                bytes[index * 2 + 1] = (value ushr 8).toByte()
            }
            file.writeBytes(bytes)
        }
        val cpuFile = File(context.filesDir, "qnn-diagnostic-cpu-s16le-24k.pcm")
        val qnnFile = File(context.filesDir, "qnn-diagnostic-htp-s16le-24k.pcm")
        writePcm(cpuFile, reference.samples)
        writePcm(qnnFile, candidate.samples)
        File(context.filesDir, "qnn-diagnostic.txt").writeText(
            "voice=${voice.id}\nspeed=$speed\ntext=$text\n" +
                "cpuSamples=${reference.samples.size}\ncpuElapsedMs=${reference.elapsedMs}\n" +
                "qnnSamples=${candidate.samples.size}\nqnnElapsedMs=${candidate.elapsedMs}\n" +
                "backend=$backend\nbuckets=${buckets.sorted().joinToString(",")}\n",
            Charsets.UTF_8,
        )
        assertEquals("HTP capture fell back to CPU", "QNN_HTP", backend)
        assertAudioSafe("physical-capture-${voice.id}", reference.samples, candidate.samples)
    }

    /** Distinguishes a model/onset defect from Android sink startup loss without playing audio. */
    @Test
    fun captureEffectiveRateOpeningEdgeWithoutPlayback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("QNN runtime is not packaged", BuildConfig.QNN_EP_INCLUDED && BuildConfig.KOKORO_QNN_AOT_INCLUDED)
        val arguments = InstrumentationRegistry.getArguments()
        val text = arguments.getString("reproText")
            ?: "US Treasury Secretary Scott Bessent made a fresh attempt to rein-in long-term borrowing costs."
        val voice = VoiceCatalog.find(arguments.getString("reproVoice") ?: "am_puck")
        val evidenceDir = File(checkNotNull(context.getExternalFilesDir(null)), "opening-edge-v40").apply {
            mkdirs()
            listFiles().orEmpty().forEach { it.delete() }
        }
        val report = StringBuilder("text=$text\nvoice=${voice.id}\n")
        KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { synthesizer ->
            synthesizer.prepare()
            listOf(1.0f, 1.3f).forEach { speed ->
                val capture = synthesizer.captureOpeningEdgeForTesting(text, voice, speed)
                File(evidenceDir, "speed-$speed-raw-s16le-24k.pcm").writeBytes(
                    KokoroSynthesizer.encodePcm16(capture.rawSamples),
                )
                File(evidenceDir, "speed-$speed-trimmed-s16le-24k.pcm").writeBytes(
                    KokoroSynthesizer.encodePcm16(capture.trimmedSamples),
                )
                val retainedBeforeActive = capture.detectedActiveStartSample?.minus(capture.removedLeadingSamples)
                report.append("speed.$speed.frames=${capture.frames}\n")
                report.append("speed.$speed.rawSamples=${capture.rawSamples.size}\n")
                report.append("speed.$speed.trimmedSamples=${capture.trimmedSamples.size}\n")
                report.append("speed.$speed.detectedActiveStartSample=${capture.detectedActiveStartSample}\n")
                report.append("speed.$speed.removedLeadingSamples=${capture.removedLeadingSamples}\n")
                report.append("speed.$speed.retainedBeforeActiveSamples=$retainedBeforeActive\n")
                assertEquals("Opening-edge capture left QNN", "QNN_HTP", synthesizer.activeBackendForTesting())
                assertTrue("Leading trim reached detected speech", retainedBeforeActive == null || retainedBeforeActive >= 0)
            }
        }
        File(evidenceDir, "report.txt").writeText(report.toString(), Charsets.UTF_8)
        println("KOKORO_OPENING_EDGE evidence=$evidenceDir")
    }

    /** Compares the streamed HTP plan with a one-front/one-generator sentence reference.
     * The CPU side is deliberately not the old independently chunked fallback. Captures remain
     * in app-private storage for the required subjective cadence comparison. */
    @Test
    fun captureGlobalFrontSlicedCadenceAgainstUnsplitReference() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("QNN runtime is not packaged", BuildConfig.QNN_EP_INCLUDED && BuildConfig.KOKORO_QNN_AOT_INCLUDED)
        assumeTrue(
            "Capture requires the physical Galaxy S24 Ultra target",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true) &&
                Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val text = "This is an example of speech synthesis in English."
        val voice = VoiceCatalog.find("af_heart")
        val speed = InstrumentationRegistry.getArguments().getString("captureSpeed")?.toFloatOrNull() ?: 1.0f
        val referenceCapture = KokoroSynthesizer(context, BackendPreference.CPU).use { cpu ->
            cpu.prepare()
            cpu.captureUnsplitGlobalFrontForTesting(text, voice, speed)
        }
        val referenceBytes = KokoroSynthesizer.encodePcm16(referenceCapture.samples)
        val reference = RenderResult(
            samples = decodePcm16(referenceBytes),
            elapsedMs = referenceCapture.elapsedMs,
            chunkSamples = listOf(referenceBytes.size / Short.SIZE_BYTES),
        )

        val candidate: RenderResult
        val backend: String
        val buckets: Set<Int>
        KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { qnn ->
            qnn.prepare()
            qnn.resetGeneratorRtfForTesting()
            candidate = render(qnn, text, voice, speed)
            backend = qnn.activeBackendForTesting()
            buckets = qnn.qnnContextSourcesForTesting().keys
        }
        assertEquals("HTP cadence capture fell back to CPU", "QNN_HTP", backend)
        assertTrue("Cadence route did not stream multiple cores", candidate.chunkSamples.size >= 2)
        assertTrue("Cadence route did not use B128/B192", buckets.containsAll(setOf(128, 192)))
        val metrics = assertAudioSafe("global-front-cadence-${voice.id}", reference.samples, candidate.samples)

        fun envelope(samples: ShortArray, start: Int, end: Int): DoubleArray {
            val window = 480
            val hop = 120
            if (end - start < window) return doubleArrayOf()
            return DoubleArray(1 + (end - start - window) / hop) { index ->
                val offset = start + index * hop
                var energy = 0.0
                for (sample in offset until offset + window) {
                    val value = samples[sample].toDouble()
                    energy += value * value
                }
                kotlin.math.sqrt(energy / window)
            }
        }

        fun correlation(first: DoubleArray, second: DoubleArray): Double {
            val length = minOf(first.size, second.size)
            if (length == 0) return Double.NaN
            val firstMean = first.take(length).average()
            val secondMean = second.take(length).average()
            var covariance = 0.0
            var firstEnergy = 0.0
            var secondEnergy = 0.0
            for (index in 0 until length) {
                val a = first[index] - firstMean
                val b = second[index] - secondMean
                covariance += a * b
                firstEnergy += a * a
                secondEnergy += b * b
            }
            return covariance / kotlin.math.sqrt((firstEnergy * secondEnergy).coerceAtLeast(1e-12))
        }

        val commonSamples = minOf(reference.samples.size, candidate.samples.size)
        fun logEnvelope(samples: ShortArray): DoubleArray =
            envelope(samples, 0, samples.size).let { values ->
                DoubleArray(values.size) { index ->
                    kotlin.math.ln1p(values[index] * 100.0 / Short.MAX_VALUE)
                }
            }

        val referenceEnvelope = logEnvelope(reference.samples)
        val candidateEnvelope = logEnvelope(candidate.samples)
        fun alignedCorrelation(lag: Int, candidateStart: Int, candidateEnd: Int): Double {
            val referenceStart = candidateStart + maxOf(0, lag)
            val shiftedCandidateStart = candidateStart + maxOf(0, -lag)
            val requested = candidateEnd - candidateStart
            val length = minOf(
                requested,
                referenceEnvelope.size - referenceStart,
                candidateEnvelope.size - shiftedCandidateStart,
            )
            if (length <= 0) return Double.NaN
            return correlation(
                referenceEnvelope.copyOfRange(referenceStart, referenceStart + length),
                candidateEnvelope.copyOfRange(shiftedCandidateStart, shiftedCandidateStart + length),
            )
        }
        val maxLagFrames = 20 // bounded to +/-100 ms at the 5 ms envelope hop
        val bestLag = (-maxLagFrames..maxLagFrames).maxBy { lag ->
            alignedCorrelation(lag, 0, minOf(referenceEnvelope.size, candidateEnvelope.size))
        }
        val wholeEnvelopeCorrelation = alignedCorrelation(
            bestLag,
            0,
            minOf(referenceEnvelope.size, candidateEnvelope.size),
        )
        val seam = candidate.chunkSamples.first()
        val envelopeHop = 120
        val seamRadiusFrames = (KokoroSynthesizer.SAMPLE_RATE / 2) / envelopeHop
        val seamFrame = seam / envelopeHop
        val seamEnvelopeCorrelation = alignedCorrelation(
            bestLag,
            maxOf(0, seamFrame - seamRadiusFrames),
            minOf(candidateEnvelope.size, seamFrame + seamRadiusFrames),
        )
        val nearStart = maxOf(1, seam - KokoroSynthesizer.SAMPLE_RATE / 10)
        val nearEnd = minOf(candidate.samples.lastIndex, seam + KokoroSynthesizer.SAMPLE_RATE / 10)
        val nearSteps = DoubleArray(nearEnd - nearStart) { index ->
            kotlin.math.abs(
                candidate.samples[nearStart + index + 1].toInt() -
                    candidate.samples[nearStart + index].toInt(),
            ).toDouble()
        }.sortedArray()
        val joinStep = kotlin.math.abs(candidate.samples[seam].toInt() - candidate.samples[seam - 1].toInt())
        val nearbyP95Step = nearSteps[(nearSteps.lastIndex * 95) / 100]
        File(context.filesDir, "global-front-unsplit-cpu-s16le-24k.pcm").writeBytes(referenceBytes)
        File(context.filesDir, "global-front-sliced-htp-s16le-24k.pcm").writeBytes(encodePcm16(candidate.samples))
        File(context.filesDir, "global-front-cadence-gate.txt").writeText(
            "voice=${voice.id}\nspeed=$speed\ntext=$text\n" +
                "frontFrames=${referenceCapture.frames}\nreferenceSamples=${reference.samples.size}\n" +
                "candidateSamples=${candidate.samples.size}\ncandidateChunkSamples=${candidate.chunkSamples.joinToString(",")}\n" +
                "referenceElapsedMs=${reference.elapsedMs}\ncandidateElapsedMs=${candidate.elapsedMs}\n" +
                "backend=$backend\nbuckets=${buckets.sorted().joinToString(",")}\n" +
                "waveformCorrelation=${metrics.correlation}\nnrmse=${metrics.nrmse}\n" +
                "roughnessRatio=${metrics.roughnessRatio}\nenvelopeLagMs=${bestLag * 5}\n" +
                "wholeEnvelopeCorrelation=$wholeEnvelopeCorrelation\n" +
                "seamEnvelopeCorrelation=$seamEnvelopeCorrelation\n" +
                "joinStep=$joinStep\nnearbyP95Step=$nearbyP95Step\n",
            Charsets.UTF_8,
        )
        println(
            "KOKORO_GLOBAL_FRONT_CADENCE waveform_corr=${metrics.correlation} " +
                "envelope_corr=$wholeEnvelopeCorrelation lag_ms=${bestLag * 5} " +
                "seam_envelope_corr=$seamEnvelopeCorrelation join=$joinStep/$nearbyP95Step " +
                "chunks=${candidate.chunkSamples}",
        )
        // Direct sample correlation is phase-sensitive and the qualified HTP suffix is not
        // sample-identical to CPU. Cadence is gated on the amplitude contour while the raw pair
        // above remains the authoritative listening artifact.
        assertTrue("Whole-sentence aligned envelope correlation=$wholeEnvelopeCorrelation", wholeEnvelopeCorrelation >= 0.95)
        assertTrue("Seam aligned envelope correlation=$seamEnvelopeCorrelation", seamEnvelopeCorrelation >= 0.90)
        assertTrue("Global-front seam step $joinStep exceeds nearby p95 $nearbyP95Step", joinStep <= nearbyP95Step)
    }

    @Test
    fun captureShortSplitQnnAndCpuPcmWithoutPlayback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("QNN runtime is not packaged", BuildConfig.QNN_EP_INCLUDED && BuildConfig.KOKORO_QNN_AOT_INCLUDED)
        assumeTrue(
            "Capture requires the physical Galaxy S24 Ultra target",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true) &&
                Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val text = "Ready."
        val voice = VoiceCatalog.find("af_heart")
        val speed = InstrumentationRegistry.getArguments().getString("captureSpeed")?.toFloatOrNull() ?: 1.3f
        val reference = KokoroSynthesizer(context, BackendPreference.CPU).use { cpu ->
            cpu.prepare()
            render(cpu, text, voice, speed)
        }
        val candidate: RenderResult
        val backend: String
        val buckets: Set<Int>
        KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { qnn ->
            qnn.prepare()
            qnn.resetGeneratorRtfForTesting()
            candidate = render(qnn, text, voice, speed)
            backend = qnn.activeBackendForTesting()
            buckets = qnn.qnnContextSourcesForTesting().keys
        }
        fun writePcm(file: File, samples: ShortArray) {
            val bytes = ByteArray(samples.size * 2)
            samples.forEachIndexed { index, sample ->
                val value = sample.toInt()
                bytes[index * 2] = value.toByte()
                bytes[index * 2 + 1] = (value ushr 8).toByte()
            }
            file.writeBytes(bytes)
        }
        writePcm(File(context.filesDir, "qnn-short-cpu-s16le-24k.pcm"), reference.samples)
        writePcm(File(context.filesDir, "qnn-short-htp-s16le-24k.pcm"), candidate.samples)
        File(context.filesDir, "qnn-short-diagnostic.txt").writeText(
            "voice=${voice.id}\nspeed=$speed\ntext=$text\n" +
                "cpuSamples=${reference.samples.size}\ncpuElapsedMs=${reference.elapsedMs}\n" +
                "qnnSamples=${candidate.samples.size}\nqnnElapsedMs=${candidate.elapsedMs}\n" +
                "backend=$backend\nbuckets=${buckets.sorted().joinToString(",")}\n",
            Charsets.UTF_8,
        )
        assertEquals("HTP capture fell back to CPU", "QNN_HTP", backend)
        assertAudioSafe("physical-short-${voice.id}", reference.samples, candidate.samples)
    }

    /** Runs an arbitrary embedded EPContext copied to filesDir and writes its
     * raw float waveform without playing it. This lets host-side graph variants
     * be qualified on the S24 without repeatedly repackaging a multi-gigabyte
     * APK. */
    @Test
    fun captureExternalQnnCandidateWithoutPlayback() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("QNN runtime is not packaged", BuildConfig.QNN_EP_INCLUDED)
        assumeTrue(
            "Capture requires the physical Galaxy S24 Ultra target",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true) &&
                Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val arguments = InstrumentationRegistry.getArguments()
        val candidateName = arguments.getString("candidateFile") ?: "qnn-candidate.ctx.onnx"
        val bucket = requireNotNull(arguments.getString("candidateBucket")?.toIntOrNull()) {
            "candidateBucket instrumentation argument is required"
        }
        val candidateKind = arguments.getString("candidateKind") ?: "acoustic"
        val fullWaveform = candidateKind == "full"
        val rawOutput = candidateKind == "raw"
        val text = arguments.getString("candidateText") ?: "This is an example of speech synthesis in English."
        val voice = VoiceCatalog.find(arguments.getString("candidateVoice") ?: "af_heart")
        val speed = arguments.getString("captureSpeed")?.toFloatOrNull() ?: 1.24f
        val warmupRuns = arguments.getString("candidateWarmupRuns")?.toIntOrNull() ?: 0
        val candidateFile = File(context.filesDir, candidateName)
        assertTrue("Missing external QNN candidate $candidateFile", candidateFile.isFile)
        val harmonicSourceFile = arguments.getString("harmonicSourceFile")?.let { name ->
            File(context.filesDir, name).also { file ->
                assertTrue("Missing CPU harmonic-source candidate $file", file.isFile)
            }
        }
        val phonemes = EnglishPhonemizer(context).phonemize(text, voice.locale)
        val result = KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { synthesizer ->
            synthesizer.captureExternalQnnCandidateForTesting(
                phonemes = phonemes,
                voice = voice.id,
                speed = speed,
                bucket = bucket,
                contextFile = candidateFile,
                harmonicSourceFile = harmonicSourceFile,
                fullWaveform = fullWaveform,
                rawOutput = rawOutput,
                warmupRuns = warmupRuns,
            )
        }
        val bytes = ByteBuffer.allocate(result.samples.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        result.samples.forEach(bytes::putFloat)
        File(context.filesDir, "external-qnn-candidate-f32le.raw").writeBytes(bytes.array())
        val finite = result.samples.count { it.isFinite() }
        val finiteValues = result.samples.filter { it.isFinite() }
        val peak = finiteValues.maxOfOrNull { kotlin.math.abs(it) } ?: Float.NaN
        val rms = if (finiteValues.isEmpty()) Double.NaN else kotlin.math.sqrt(
            finiteValues.sumOf { it.toDouble() * it.toDouble() } / finiteValues.size,
        )
        File(context.filesDir, "external-qnn-candidate.txt").writeText(
            "candidate=$candidateName\nkind=$candidateKind\n" +
                "voice=${voice.id}\nspeed=$speed\ntext=$text\nphonemes=$phonemes\n" +
                "frames=${result.frames}\nbucket=$bucket\nwarmupRuns=$warmupRuns\nsamples=${result.samples.size}\n" +
                "finite=$finite\npeak=$peak\nrms=$rms\nelapsedMs=${result.elapsedMs}\n",
            Charsets.UTF_8,
        )
        println(
            "KOKORO_EXTERNAL_QNN candidate=$candidateName kind=$candidateKind " +
                "T=${result.frames} B=$bucket finite=$finite/${result.samples.size} peak=$peak rms=$rms " +
                "elapsed_ms=${result.elapsedMs}",
        )
        assertTrue("External QNN candidate returned no samples", result.samples.isNotEmpty())
    }

    @Test
    fun optimizedCpuProducesStreamablePcm() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        KokoroSynthesizer(context, BackendPreference.CPU).use { synthesizer ->
            synthesizer.prepare()
            var firstChunkMs = 0L
            var chunks = 0
            var pcmBytes = 0
            val started = SystemClock.elapsedRealtime()
            synthesizer.synthesizeChunks(
                "This is an example of speech synthesis in English.",
                VoiceCatalog.default,
                1.0f,
            ) { pcm ->
                if (firstChunkMs == 0L) firstChunkMs = SystemClock.elapsedRealtime() - started
                chunks++
                pcmBytes += pcm.size
                pcm.any { it != 0.toByte() }
            }
            println("KOKORO_CLEAR_AUDIO cpu first_chunk_ms=$firstChunkMs chunks=$chunks pcm_bytes=$pcmBytes")
            assertTrue("normal Settings sample should stream a bounded opening", chunks >= 2)
            assertTrue("expected audible PCM", pcmBytes > 10_000)

            val cachedStarted = SystemClock.elapsedRealtime()
            var cachedBytes = 0
            synthesizer.synthesizeChunks(
                "This is an example of speech synthesis in English.",
                VoiceCatalog.default,
                1.0f,
            ) { pcm -> cachedBytes += pcm.size; true }
            val cachedMs = SystemClock.elapsedRealtime() - cachedStarted
            println("KOKORO_CLEAR_AUDIO cached_ms=$cachedMs cached_bytes=$cachedBytes")
            assertTrue("repeat cache changed the audio size", cachedBytes == pcmBytes)
            assertTrue("repeat cache was not faster than first synthesis", cachedMs < firstChunkMs)
        }
    }

    @Test
    fun nnapiPathProducesPcmOrFallsBackSafely() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        KokoroSynthesizer(context, BackendPreference.NNAPI).use { synthesizer ->
            var pcmBytes = 0
            synthesizer.synthesizeChunks("Hello world.", VoiceCatalog.default, 1.0f) { pcm ->
                pcmBytes += pcm.size
                true
            }
            assertTrue("NNAPI/CPU fallback returned no PCM", pcmBytes > 10_000)
        }
    }

    @Test
    fun androidTextToSpeechApiCreatesARealWaveFile() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val initialized = CountDownLatch(1)
        var initStatus = TextToSpeech.ERROR
        lateinit var tts: TextToSpeech
        instrumentation.runOnMainSync {
            tts = TextToSpeech(context, { status ->
                initStatus = status
                initialized.countDown()
            }, context.packageName)
        }
        assertTrue("Android did not bind the Kokoro engine", initialized.await(15, TimeUnit.SECONDS))
        assertTrue("Kokoro engine initialization failed", initStatus == TextToSpeech.SUCCESS)

        val completed = CountDownLatch(1)
        var synthesisError = false
        val utteranceId = "native-api-contract"
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) = Unit
            override fun onDone(id: String?) { if (id == utteranceId) completed.countDown() }
            @Deprecated("Legacy Android callback")
            override fun onError(id: String?) { synthesisError = true; completed.countDown() }
        })
        val output = File(context.cacheDir, "kokoro-native-api.wav").apply { delete() }
        val accepted = tts.synthesizeToFile(
            "This is an example of speech synthesis in English.",
            Bundle(),
            output,
            utteranceId,
        )
        assertTrue("Android rejected the synthesis request", accepted == TextToSpeech.SUCCESS)
        assertTrue("Android TTS synthesis timed out", completed.await(30, TimeUnit.SECONDS))
        tts.shutdown()
        assertTrue("Android reported a synthesis error", !synthesisError)
        assertTrue("Framework WAV is empty", output.length() > 10_000)
        assertTrue("Framework did not write a RIFF/WAVE file", output.inputStream().use {
            val header = ByteArray(4)
            it.read(header) == header.size && String(header, Charsets.US_ASCII) == "RIFF"
        })
    }

    @Test
    fun androidTextToSpeechStreamsProgressiveFirstPcm() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val initialized = CountDownLatch(1)
        var initStatus = TextToSpeech.ERROR
        lateinit var tts: TextToSpeech
        instrumentation.runOnMainSync {
            tts = TextToSpeech(context, { status ->
                initStatus = status
                initialized.countDown()
            }, context.packageName)
        }
        assertTrue("Android did not bind the Kokoro engine", initialized.await(15, TimeUnit.SECONDS))
        assertEquals("Kokoro engine initialization failed", TextToSpeech.SUCCESS, initStatus)

        val voice = tts.voices.firstOrNull { it.name == "af_heart" }
        assertTrue("Android did not expose af_heart", voice != null)
        assertEquals("Android rejected af_heart", TextToSpeech.SUCCESS, tts.setVoice(voice))
        assertEquals("Android rejected normal speech rate", TextToSpeech.SUCCESS, tts.setSpeechRate(1.0f))

        // Let the service's bounded B128/B192 preparation reach the same resident-idle state
        // used by @Voice between one-sentence requests.
        Thread.sleep(4_500)

        val firstAudio = CountDownLatch(1)
        val completed = CountDownLatch(1)
        val requestAt = AtomicLong(-1L)
        val beginAt = AtomicLong(-1L)
        val firstAudioAt = AtomicLong(-1L)
        val playbackAt = AtomicLong(-1L)
        val doneAt = AtomicLong(-1L)
        val audioBytes = AtomicLong(0L)
        val doneCount = AtomicInteger(0)
        val errorCode = AtomicInteger(0)
        val utteranceId = "progressive-first-pcm"
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onBeginSynthesis(id: String?, sampleRateInHz: Int, audioFormat: Int, channelCount: Int) {
                if (id == utteranceId) beginAt.compareAndSet(-1L, SystemClock.elapsedRealtime())
            }

            override fun onAudioAvailable(id: String?, audio: ByteArray?) {
                if (id != utteranceId || audio == null) return
                audioBytes.addAndGet(audio.size.toLong())
                if (firstAudioAt.compareAndSet(-1L, SystemClock.elapsedRealtime())) firstAudio.countDown()
            }

            override fun onStart(id: String?) {
                if (id == utteranceId) playbackAt.compareAndSet(-1L, SystemClock.elapsedRealtime())
            }

            override fun onDone(id: String?) {
                if (id != utteranceId) return
                doneAt.set(SystemClock.elapsedRealtime())
                doneCount.incrementAndGet()
                completed.countDown()
            }

            @Deprecated("Legacy Android callback")
            override fun onError(id: String?) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.ERROR)
                    completed.countDown()
                }
            }

            override fun onError(id: String?, code: Int) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, code)
                    completed.countDown()
                }
            }

            override fun onStop(id: String?, interrupted: Boolean) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.STOPPED)
                    completed.countDown()
                }
            }
        })

        requestAt.set(SystemClock.elapsedRealtime())
        val accepted = tts.speak(
            "This is an example of speech synthesis in English.",
            TextToSpeech.QUEUE_FLUSH,
            Bundle(),
            utteranceId,
        )
        assertEquals("Android rejected the progressive speech request", TextToSpeech.SUCCESS, accepted)
        assertTrue("Android did not expose first PCM within 10 seconds", firstAudio.await(10, TimeUnit.SECONDS))
        assertTrue("Android did not complete progressive playback", completed.await(15, TimeUnit.SECONDS))
        instrumentation.runOnMainSync { tts.shutdown() }

        val firstMs = firstAudioAt.get() - requestAt.get()
        val beginMs = beginAt.get() - requestAt.get()
        val playbackMs = playbackAt.get() - requestAt.get()
        val totalMs = doneAt.get() - requestAt.get()
        val bytes = audioBytes.get()
        val receipt = buildString {
            append("firstPcmMs=$firstMs\n")
            append("beginSynthesisMs=$beginMs\n")
            append("playbackMs=$playbackMs\n")
            append("doneMs=$totalMs\n")
            append("audioBytes=$bytes\n")
            append("doneCount=${doneCount.get()}\n")
            append("errorCode=${errorCode.get()}\n")
        }
        File(context.filesDir, "first-audio-service-gate.txt").writeText(receipt, Charsets.UTF_8)
        println(
            "KOKORO_FIRST_PCM_SERVICE first_ms=$firstMs begin_ms=$beginMs playback_ms=$playbackMs " +
                "done_ms=$totalMs bytes=$bytes done_count=${doneCount.get()} error=${errorCode.get()}",
        )
        assertEquals("Android reported a synthesis/playback error", 0, errorCode.get())
        assertEquals("Android completion callback count changed", 1, doneCount.get())
        assertTrue("First PCM was not materially below v1.20 B224", firstMs in 1..1_000)
        assertTrue("Synthesis begin callback was missing", beginMs in 0..firstMs)
        assertTrue("Playback began before the request or after completion", playbackMs in 0..totalMs)
        assertTrue("Android completed before first PCM", totalMs > firstMs)
        assertTrue("Progressive output was truncated ($bytes bytes)", bytes in 90_000L..140_000L)
    }

    /**
     * Records the framework-visible PCM cadence for the same one-sentence path used by @Voice.
     * The projected starvation value treats every onAudioAvailable payload as immediately
     * playable 24 kHz mono PCM and therefore makes no hidden AudioTrack buffering assumption.
     */
    @Test
    fun androidTextToSpeechMeasuresProgressiveContinuity() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val caseName = InstrumentationRegistry.getArguments().getString("continuityCase") ?: "standard"
        val text = when (caseName) {
            "standard" -> "This is an example of speech synthesis in English."
            "medium" -> "Numbers like 24, 3.14159, and 2026 should sound natural, not noisy."
            "long" -> "After the morning rain had cleared, the travelers crossed the quiet valley and followed the lanterns toward the old stone bridge before sunset."
            else -> error("Unknown continuityCase=$caseName")
        }
        val initialized = CountDownLatch(1)
        var initStatus = TextToSpeech.ERROR
        lateinit var tts: TextToSpeech
        instrumentation.runOnMainSync {
            tts = TextToSpeech(context, { status ->
                initStatus = status
                initialized.countDown()
            }, context.packageName)
        }
        assertTrue("Android did not bind the Kokoro engine", initialized.await(15, TimeUnit.SECONDS))
        assertEquals("Kokoro engine initialization failed", TextToSpeech.SUCCESS, initStatus)

        val voice = tts.voices.firstOrNull { it.name == "af_heart" }
        assertTrue("Android did not expose af_heart", voice != null)
        assertEquals("Android rejected af_heart", TextToSpeech.SUCCESS, tts.setVoice(voice))
        assertEquals("Android rejected normal speech rate", TextToSpeech.SUCCESS, tts.setSpeechRate(1.0f))
        Thread.sleep(4_500)

        val completed = CountDownLatch(1)
        val requestAtNanos = AtomicLong(-1L)
        val doneAtNanos = AtomicLong(-1L)
        val errorCode = AtomicInteger(0)
        val events = Collections.synchronizedList(mutableListOf<Pair<Long, Int>>())
        val utteranceId = "progressive-continuity"
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) = Unit

            override fun onAudioAvailable(id: String?, audio: ByteArray?) {
                if (id == utteranceId && audio != null) {
                    events += SystemClock.elapsedRealtimeNanos() to audio.size
                }
            }

            override fun onDone(id: String?) {
                if (id == utteranceId) {
                    doneAtNanos.set(SystemClock.elapsedRealtimeNanos())
                    completed.countDown()
                }
            }

            @Deprecated("Legacy Android callback")
            override fun onError(id: String?) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.ERROR)
                    completed.countDown()
                }
            }

            override fun onError(id: String?, code: Int) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, code)
                    completed.countDown()
                }
            }

            override fun onStop(id: String?, interrupted: Boolean) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.STOPPED)
                    completed.countDown()
                }
            }
        })

        requestAtNanos.set(SystemClock.elapsedRealtimeNanos())
        val accepted = tts.speak(
            text,
            TextToSpeech.QUEUE_FLUSH,
            Bundle(),
            utteranceId,
        )
        assertEquals("Android rejected the continuity request", TextToSpeech.SUCCESS, accepted)
        assertTrue("Android did not complete the continuity request", completed.await(15, TimeUnit.SECONDS))
        instrumentation.runOnMainSync { tts.shutdown() }
        assertEquals("Android reported a synthesis/playback error", 0, errorCode.get())

        val observed = synchronized(events) { events.toList() }
        assertTrue("Android exposed no PCM callbacks", observed.isNotEmpty())
        val firstAt = observed.first().first
        var projectedQueueEnd = firstAt
        var projectedStarvationNanos = 0L
        var maxInterCallbackNanos = 0L
        var totalBytes = 0L
        observed.forEachIndexed { index, (at, bytes) ->
            if (index > 0) {
                maxInterCallbackNanos = maxOf(maxInterCallbackNanos, at - observed[index - 1].first)
                if (at > projectedQueueEnd) {
                    projectedStarvationNanos += at - projectedQueueEnd
                    projectedQueueEnd = at
                }
            }
            projectedQueueEnd += bytes.toLong() * 1_000_000_000L /
                (KokoroSynthesizer.SAMPLE_RATE * Short.SIZE_BYTES)
            totalBytes += bytes
        }
        val eventLines = observed.mapIndexed { index, (at, bytes) ->
            val offsetMicros = (at - requestAtNanos.get()) / 1_000L
            "$index:$offsetMicros:$bytes"
        }
        val receipt = buildString {
            append("case=$caseName\n")
            append("textChars=${text.length}\n")
            append("firstPcmMs=${(firstAt - requestAtNanos.get()) / 1_000_000.0}\n")
            append("doneMs=${(doneAtNanos.get() - requestAtNanos.get()) / 1_000_000.0}\n")
            append("audioBytes=$totalBytes\n")
            append("callbackCount=${observed.size}\n")
            append("maxInterCallbackMs=${maxInterCallbackNanos / 1_000_000.0}\n")
            append("projectedStarvationMs=${projectedStarvationNanos / 1_000_000.0}\n")
            append("events=index:requestOffsetUs:bytes\n")
            eventLines.forEach { append(it).append('\n') }
        }
        File(context.filesDir, "continuity-service-gate.txt").writeText(receipt, Charsets.UTF_8)
        println(
            "KOKORO_CONTINUITY first_ms=${(firstAt - requestAtNanos.get()) / 1_000_000.0} " +
                "done_ms=${(doneAtNanos.get() - requestAtNanos.get()) / 1_000_000.0} " +
                "bytes=$totalBytes callbacks=${observed.size} " +
                "projected_starvation_ms=${projectedStarvationNanos / 1_000_000.0}",
        )
        val minimumBytes = when (caseName) {
            "standard" -> 90_000L
            "medium" -> 200_000L
            "long" -> 250_000L
            else -> error("Unknown continuityCase=$caseName")
        }
        assertTrue("Continuity output was truncated ($totalBytes bytes)", totalBytes >= minimumBytes)
        assertEquals("Framework-visible PCM queue starved", 0L, projectedStarvationNanos)
        if (caseName == "standard") {
            val firstPcmNanos = firstAt - requestAtNanos.get()
            assertTrue("First PCM was not materially below v1.20", firstPcmNanos in 1..1_100_000_000L)
        }
    }

    @Test
    fun captureWhaleSnailLoudnessAndCallbackRepro() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assumeTrue(
            "Capture requires the physical Galaxy S24 Ultra target",
            Build.VERSION.SDK_INT >= 31 && Build.SOC_MODEL.equals("SM8650", ignoreCase = true) &&
                Build.MODEL.uppercase(Locale.US).startsWith("SM-S928"),
        )
        val arguments = InstrumentationRegistry.getArguments()
        val text = arguments.getString("reproText")
            ?: "The tide slipped away leaving the whale and the snail on the shore."
        val voice = VoiceCatalog.find(
            arguments.getString("reproVoice") ?: VoiceCatalog.preferredDefaultId(context),
        )
        val requestRatePercent = arguments.getString("reproRequestRatePercent")?.toIntOrNull() ?: 100
        val requireCallbackMatch = arguments.getString("requireCallbackMatch")
            ?.equals("true", ignoreCase = true) ?: (requestRatePercent == 100)
        val deliveryMultiplier = ExpressionSettings.deliverySpeed(context)
        val modelSpeed = KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(
            KokoroSynthesizer.modelSpeedForRequest(requestRatePercent),
            deliveryMultiplier,
        )
        val evidenceDir = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "whale-snail-volume-repro-v${BuildConfig.VERSION_CODE}",
        ).apply {
            mkdirs()
            listFiles().orEmpty().forEach { it.delete() }
        }

        data class DirectPass(
            val label: String,
            val diagnostics: List<ChunkOutputDiagnostics>,
            val pcm: ByteArray,
            val consumerEvents: List<Pair<Long, Int>>,
            val elapsedMs: Double,
        )

        fun captureDirect(synthesizer: KokoroSynthesizer, label: String): DirectPass {
            val diagnostics = mutableListOf<ChunkOutputDiagnostics>()
            val output = ByteArrayOutputStream()
            val events = mutableListOf<Pair<Long, Int>>()
            val started = SystemClock.elapsedRealtimeNanos()
            synthesizer.synthesizeChunks(
                text,
                voice,
                modelSpeed,
                diagnostics = { diagnostics += it },
            ) { pcm ->
                events += (SystemClock.elapsedRealtimeNanos() - started) to pcm.size
                output.write(pcm)
                true
            }
            return DirectPass(
                label = label,
                diagnostics = diagnostics.toList(),
                pcm = output.toByteArray(),
                consumerEvents = events.toList(),
                elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0,
            )
        }

        fun pcmSample(bytes: ByteArray, index: Int): Int {
            val offset = index * Short.SIZE_BYTES
            return ((bytes[offset].toInt() and 0xff) or (bytes[offset + 1].toInt() shl 8)).toShort().toInt()
        }

        fun nearbyP95Step(bytes: ByteArray, boundary: Int): Int {
            val samples = bytes.size / Short.SIZE_BYTES
            val start = maxOf(1, boundary - KokoroSynthesizer.SAMPLE_RATE / 10)
            val end = minOf(samples - 1, boundary + KokoroSynthesizer.SAMPLE_RATE / 10)
            val steps = (start..end).map { index ->
                kotlin.math.abs(pcmSample(bytes, index) - pcmSample(bytes, index - 1))
            }.sorted()
            return steps[(steps.lastIndex * 95) / 100]
        }

        fun windowRms(bytes: ByteArray, start: Int, count: Int): Double {
            var energy = 0.0
            for (index in start until start + count) {
                val sample = pcmSample(bytes, index).toDouble()
                energy += sample * sample
            }
            return kotlin.math.sqrt(energy / count)
        }

        fun writeDirect(pass: DirectPass) {
            File(evidenceDir, "${pass.label}-post-s16le-24k.pcm").writeBytes(pass.pcm)
            val preOutput = ByteArrayOutputStream()
            val preJoiner = GlobalPcmOverlapJoiner()
            val gainCsv = StringBuilder("window_start_sample,chunk,local_start_sample,gain,pre_rms,post_rms\n")
            var coreStart = 0
            pass.diagnostics.forEach { diagnostic ->
                File(evidenceDir, "${pass.label}-chunk-${diagnostic.index}-pre-s16le-24k.pcm")
                    .writeBytes(diagnostic.preStabilizerPcm)
                File(evidenceDir, "${pass.label}-chunk-${diagnostic.index}-post-s16le-24k.pcm")
                    .writeBytes(diagnostic.postStabilizerPcm)
                File(evidenceDir, "${pass.label}-chunk-${diagnostic.index}-delivered-s16le-24k.pcm")
                    .writeBytes(diagnostic.deliveredPcm)
                preOutput.write(
                    preJoiner.stitch(
                        diagnostic.preStabilizerPcm,
                        diagnostic.leadingOverlapSamples,
                        diagnostic.trailingOverlapSamples,
                    ),
                )
                val samples = diagnostic.preStabilizerPcm.size / Short.SIZE_BYTES
                val windowStart = coreStart - diagnostic.leadingOverlapSamples
                val meterWindow = KokoroSynthesizer.SAMPLE_RATE / 100
                for (localStart in 0 until samples step meterWindow) {
                    val count = minOf(meterWindow, samples - localStart)
                    val gainIndex = localStart + count / 2
                    val gain = if (diagnostic.rampSamples > 0 && gainIndex < diagnostic.rampSamples) {
                        diagnostic.startGain + (diagnostic.appliedGain - diagnostic.startGain) *
                            ((gainIndex + 1).toFloat() / diagnostic.rampSamples)
                    } else {
                        diagnostic.appliedGain
                    }
                    gainCsv.append(windowStart + localStart).append(',')
                        .append(diagnostic.index).append(',').append(localStart).append(',')
                        .append(gain).append(',')
                        .append(windowRms(diagnostic.preStabilizerPcm, localStart, count)).append(',')
                        .append(windowRms(diagnostic.postStabilizerPcm, localStart, count)).append('\n')
                }
                coreStart += diagnostic.coreSamples
            }
            preJoiner.requireComplete()
            val prePcm = preOutput.toByteArray()
            File(evidenceDir, "${pass.label}-pre-s16le-24k.pcm").writeBytes(prePcm)
            File(evidenceDir, "${pass.label}-gain-envelope.csv").writeText(gainCsv.toString(), Charsets.UTF_8)

            val report = buildString {
                append("label=${pass.label}\ntext=$text\nvoice=${voice.id}\n")
                append("requestRatePercent=$requestRatePercent\ndeliveryMultiplier=$deliveryMultiplier\n")
                append("modelSpeed=$modelSpeed\nelapsedMs=${pass.elapsedMs}\n")
                append("preSha256=${sha256(prePcm)}\npostSha256=${sha256(pass.pcm)}\n")
                append("chunks=${pass.diagnostics.size}\n")
                var boundary = 0
                pass.diagnostics.forEach { diagnostic ->
                    val samples = diagnostic.preStabilizerPcm.size / Short.SIZE_BYTES
                    append("chunk.${diagnostic.index}.windowStartSample=${boundary - diagnostic.leadingOverlapSamples}\n")
                    append("chunk.${diagnostic.index}.semanticCoreStartSample=$boundary\n")
                    append("chunk.${diagnostic.index}.windowSamples=$samples\n")
                    append("chunk.${diagnostic.index}.coreSamples=${diagnostic.coreSamples}\n")
                    append("chunk.${diagnostic.index}.leadingOverlapSamples=${diagnostic.leadingOverlapSamples}\n")
                    append("chunk.${diagnostic.index}.trailingOverlapSamples=${diagnostic.trailingOverlapSamples}\n")
                    append("chunk.${diagnostic.index}.deliveredSamples=${diagnostic.deliveredPcm.size / Short.SIZE_BYTES}\n")
                    append("chunk.${diagnostic.index}.activeRms=${diagnostic.activeRms}\n")
                    append("chunk.${diagnostic.index}.referenceRms=${diagnostic.referenceRms}\n")
                    append("chunk.${diagnostic.index}.requestedGain=${diagnostic.requestedGain}\n")
                    append("chunk.${diagnostic.index}.startGain=${diagnostic.startGain}\n")
                    append("chunk.${diagnostic.index}.appliedGain=${diagnostic.appliedGain}\n")
                    append("chunk.${diagnostic.index}.rampSamples=${diagnostic.rampSamples}\n")
                    append("chunk.${diagnostic.index}.peak=${diagnostic.peak}\n")
                    append("chunk.${diagnostic.index}.preSha256=${sha256(diagnostic.preStabilizerPcm)}\n")
                    append("chunk.${diagnostic.index}.postSha256=${sha256(diagnostic.postStabilizerPcm)}\n")
                    if (diagnostic.index > 0) {
                        val preStep = kotlin.math.abs(pcmSample(prePcm, boundary) - pcmSample(prePcm, boundary - 1))
                        val postStep = kotlin.math.abs(pcmSample(pass.pcm, boundary) - pcmSample(pass.pcm, boundary - 1))
                        append("join.${diagnostic.index - 1}:${diagnostic.index}.sample=$boundary\n")
                        append("join.${diagnostic.index - 1}:${diagnostic.index}.preStep=$preStep\n")
                        append("join.${diagnostic.index - 1}:${diagnostic.index}.preNearbyP95=${nearbyP95Step(prePcm, boundary)}\n")
                        append("join.${diagnostic.index - 1}:${diagnostic.index}.postStep=$postStep\n")
                        append("join.${diagnostic.index - 1}:${diagnostic.index}.postNearbyP95=${nearbyP95Step(pass.pcm, boundary)}\n")
                    }
                    boundary += diagnostic.coreSamples
                }
                append("consumerEvents=index:offsetUs:bytes\n")
                var deliveryBoundary = 0
                pass.consumerEvents.forEachIndexed { index, (offset, bytes) ->
                    append("$index:${offset / 1_000L}:$bytes:deliveryStartSample=$deliveryBoundary\n")
                    deliveryBoundary += bytes / Short.SIZE_BYTES
                }
            }
            File(evidenceDir, "${pass.label}-report.txt").writeText(report, Charsets.UTF_8)
        }

        lateinit var cold: DirectPass
        lateinit var cached: DirectPass
        lateinit var directBackend: String
        lateinit var directBuckets: Set<Int>
        KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { synthesizer ->
            synthesizer.prepare()
            cold = captureDirect(synthesizer, "direct-cold")
            cached = captureDirect(synthesizer, "direct-cache-hit")
            directBackend = synthesizer.activeBackendForTesting()
            directBuckets = synthesizer.qnnContextSourcesForTesting().keys
        }
        writeDirect(cold)
        writeDirect(cached)

        val initialized = CountDownLatch(1)
        val completed = CountDownLatch(1)
        var initStatus = TextToSpeech.ERROR
        lateinit var tts: TextToSpeech
        instrumentation.runOnMainSync {
            tts = TextToSpeech(context, { status ->
                initStatus = status
                initialized.countDown()
            }, context.packageName)
        }
        assertTrue("Android did not bind the Kokoro engine", initialized.await(15, TimeUnit.SECONDS))
        assertEquals("Kokoro engine initialization failed", TextToSpeech.SUCCESS, initStatus)
        val androidVoice = tts.voices.firstOrNull { it.name == voice.id }
        assertTrue("Android did not expose ${voice.id}", androidVoice != null)
        assertEquals("Android rejected ${voice.id}", TextToSpeech.SUCCESS, tts.setVoice(androidVoice))
        assertEquals(
            "Android rejected repro speech rate",
            TextToSpeech.SUCCESS,
            tts.setSpeechRate(requestRatePercent / 100f),
        )
        Thread.sleep(4_500)

        val requestAt = AtomicLong(SystemClock.elapsedRealtimeNanos())
        val beginAt = AtomicLong(-1L)
        val playbackAt = AtomicLong(-1L)
        val doneAt = AtomicLong(-1L)
        val errorCode = AtomicInteger(0)
        val callbackEvents = Collections.synchronizedList(mutableListOf<Pair<Long, ByteArray>>())
        val utteranceId = "whale-snail-volume-repro"
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onBeginSynthesis(id: String?, sampleRateInHz: Int, audioFormat: Int, channelCount: Int) {
                if (id == utteranceId) beginAt.compareAndSet(-1L, SystemClock.elapsedRealtimeNanos())
            }

            override fun onAudioAvailable(id: String?, audio: ByteArray?) {
                if (id == utteranceId && audio != null) {
                    callbackEvents += (SystemClock.elapsedRealtimeNanos() to audio.copyOf())
                }
            }

            override fun onStart(id: String?) {
                if (id == utteranceId) playbackAt.compareAndSet(-1L, SystemClock.elapsedRealtimeNanos())
            }

            override fun onDone(id: String?) {
                if (id == utteranceId) {
                    doneAt.set(SystemClock.elapsedRealtimeNanos())
                    completed.countDown()
                }
            }

            @Deprecated("Legacy Android callback")
            override fun onError(id: String?) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.ERROR)
                    completed.countDown()
                }
            }

            override fun onError(id: String?, code: Int) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, code)
                    completed.countDown()
                }
            }

            override fun onStop(id: String?, interrupted: Boolean) {
                if (id == utteranceId) {
                    errorCode.compareAndSet(0, TextToSpeech.STOPPED)
                    completed.countDown()
                }
            }
        })
        requestAt.set(SystemClock.elapsedRealtimeNanos())
        assertEquals(
            "Android rejected the exact repro",
            TextToSpeech.SUCCESS,
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, Bundle(), utteranceId),
        )
        assertTrue("Android did not complete the exact repro", completed.await(30, TimeUnit.SECONDS))
        instrumentation.runOnMainSync { tts.shutdown() }

        val observedCallbacks = synchronized(callbackEvents) { callbackEvents.toList() }
        val callbackPcm = ByteArrayOutputStream().also { output ->
            observedCallbacks.forEach { (_, pcm) -> output.write(pcm) }
        }.toByteArray()
        var bufferedUntilNanos = -1L
        var projectedCallbackStarvationNanos = 0L
        observedCallbacks.forEach { (atNanos, pcm) ->
            if (bufferedUntilNanos < 0L) bufferedUntilNanos = atNanos
            if (atNanos > bufferedUntilNanos) {
                projectedCallbackStarvationNanos += atNanos - bufferedUntilNanos
            }
            val audioNanos = (pcm.size / Short.SIZE_BYTES).toLong() * 1_000_000_000L /
                KokoroSynthesizer.SAMPLE_RATE
            bufferedUntilNanos = maxOf(bufferedUntilNanos, atNanos) + audioNanos
        }
        val directPeak = (cold.pcm.indices step Short.SIZE_BYTES).maxOf { offset ->
            kotlin.math.abs(
                ((cold.pcm[offset].toInt() and 0xff) or (cold.pcm[offset + 1].toInt() shl 8))
                    .toShort().toInt(),
            )
        }
        val clippedSamples = (cold.pcm.indices step Short.SIZE_BYTES).count { offset ->
            val sample = ((cold.pcm[offset].toInt() and 0xff) or
                (cold.pcm[offset + 1].toInt() shl 8)).toShort().toInt()
            sample == Short.MIN_VALUE.toInt() || sample == Short.MAX_VALUE.toInt()
        }
        val referenceActiveRms = checkNotNull(cold.diagnostics.first().activeRms)
        val continuationActiveRms = cold.diagnostics.drop(1).mapNotNull { it.activeRms }
        assertEquals(
            "Every continuation must contain measurable active speech",
            cold.diagnostics.size - 1,
            continuationActiveRms.size,
        )
        val minimumContinuationRmsRatio = continuationActiveRms.minOf { it / referenceActiveRms }
        File(evidenceDir, "android-callback-s16le-24k.pcm").writeBytes(callbackPcm)
        File(evidenceDir, "android-callback-report.txt").writeText(
            buildString {
                append("text=$text\nvoice=${voice.id}\nrequestRatePercent=$requestRatePercent\n")
                append("deliveryMultiplier=$deliveryMultiplier\nmodelSpeed=$modelSpeed\n")
                append("directBackend=$directBackend\ndirectBuckets=${directBuckets.sorted().joinToString(",")}\n")
                append("requestToBeginMs=${(beginAt.get() - requestAt.get()) / 1_000_000.0}\n")
                append("requestToPlaybackMs=${(playbackAt.get() - requestAt.get()) / 1_000_000.0}\n")
                append("requestToDoneMs=${(doneAt.get() - requestAt.get()) / 1_000_000.0}\n")
                append("callbackCount=${observedCallbacks.size}\ncallbackBytes=${callbackPcm.size}\n")
                append("callbackSha256=${sha256(callbackPcm)}\n")
                append("directPostSha256=${sha256(cold.pcm)}\n")
                append("callbackMatchesDirect=${callbackPcm.contentEquals(cold.pcm)}\n")
                append("requireCallbackMatch=$requireCallbackMatch\n")
                append("cacheHitMatchesCold=${cached.pcm.contentEquals(cold.pcm)}\n")
                append("errorCode=${errorCode.get()}\n")
                append("directSamples=${cold.pcm.size / Short.SIZE_BYTES}\n")
                append("semanticCoreSamples=${cold.diagnostics.sumOf { it.coreSamples }}\n")
                append("directPeak=$directPeak\nclippedSamples=$clippedSamples\n")
                append("minimumContinuationRmsRatio=$minimumContinuationRmsRatio\n")
                append("projectedCallbackStarvationMs=${projectedCallbackStarvationNanos / 1_000_000.0}\n")
                append("callbacks=index:requestOffsetUs:bytes\n")
                observedCallbacks.forEachIndexed { index, (at, pcm) ->
                    append("$index:${(at - requestAt.get()) / 1_000L}:${pcm.size}\n")
                }
            },
            Charsets.UTF_8,
        )

        println(
            "KOKORO_WHALE_SNAIL voice=${voice.id} speed=$modelSpeed backend=$directBackend " +
                "chunks=${cold.diagnostics.size} callback_bytes=${callbackPcm.size} evidence=$evidenceDir",
        )
        assertEquals("Exact repro did not remain on QNN HTP", "QNN_HTP", directBackend)
        assertTrue("Exact repro did not exercise a window join", cold.diagnostics.size >= 2)
        assertTrue("Cache-hit PCM differs from the cold path", cached.pcm.contentEquals(cold.pcm))
        assertEquals("Android reported an exact-repro error", 0, errorCode.get())
        assertTrue("Android callback PCM was empty", callbackPcm.isNotEmpty())
        assertEquals("Framework callback queue starved", 0L, projectedCallbackStarvationNanos)
        assertEquals("Exact repro PCM clipped", 0, clippedSamples)
        assertTrue(
            "Continuation became nearly inaudible (active RMS ratio $minimumContinuationRmsRatio)",
            minimumContinuationRmsRatio >= 0.5,
        )
        if (requireCallbackMatch) {
            assertTrue("Android callback PCM differs from the qualified direct path", callbackPcm.contentEquals(cold.pcm))
        }
        assertEquals(
            "Shared-context joining changed the global sentence cadence",
            cold.diagnostics.sumOf { it.coreSamples },
            cold.pcm.size / Short.SIZE_BYTES,
        )
        var semanticBoundary = 0
        cold.diagnostics.forEach { diagnostic ->
            if (diagnostic.index > 0) {
                val halfOverlap = diagnostic.leadingOverlapSamples
                assertTrue("Exact repro boundary did not use shared global overlap", halfOverlap > 0)
                val previous = cold.diagnostics[diagnostic.index - 1]
                val crossfadeStart = semanticBoundary - halfOverlap
                val previousOverlapStart = previous.postStabilizerPcm.size / Short.SIZE_BYTES - 2 * halfOverlap
                assertEquals(
                    "Crossfade start changed the previous window hand-off sample",
                    pcmSample(previous.postStabilizerPcm, previousOverlapStart - 1),
                    pcmSample(cold.pcm, crossfadeStart - 1),
                )
                assertEquals(
                    "Crossfade start did not continue the previous window exactly",
                    pcmSample(previous.postStabilizerPcm, previousOverlapStart),
                    pcmSample(cold.pcm, crossfadeStart),
                )
                val crossfadeEnd = semanticBoundary + halfOverlap
                assertEquals(
                    "Crossfade end did not reach the continuation window exactly",
                    pcmSample(diagnostic.postStabilizerPcm, 2 * halfOverlap - 1),
                    pcmSample(cold.pcm, crossfadeEnd - 1),
                )
                assertEquals(
                    "Crossfade end changed the continuation hand-off sample",
                    pcmSample(diagnostic.postStabilizerPcm, 2 * halfOverlap),
                    pcmSample(cold.pcm, crossfadeEnd),
                )
                val semanticStep = kotlin.math.abs(
                    pcmSample(cold.pcm, semanticBoundary) - pcmSample(cold.pcm, semanticBoundary - 1),
                )
                val nearbyP95 = nearbyP95Step(cold.pcm, semanticBoundary)
                // A click is a robust local outlier, not merely any sample above the exact p95.
                // Loudness ramps can lower the neighborhood p95 while also reducing the join's
                // absolute step. Retain a 2x local bound with a 1024-count quantization/noise
                // floor; the original broken join (20975 versus 2527) remains decisively caught.
                val maximumNormalStep = maxOf(nearbyP95 * 2, 1024)
                assertTrue(
                    "Exact repro semantic join discontinuity is $semanticStep/$nearbyP95 " +
                        "(limit $maximumNormalStep)",
                    semanticStep <= maximumNormalStep,
                )
            }
            semanticBoundary += diagnostic.coreSamples
        }
    }

    @Test
    fun diagnoseProgressiveCandidateBoundaries() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val voice = VoiceCatalog.find("af_heart")
        val phonemes = EnglishPhonemizer(context).phonemize(
            "This is an example of speech synthesis in English.",
            voice.locale,
        )
        for (target in listOf(26, 30, 32, 34, 36, 38, 40, 42)) {
            val chunks = KokoroSynthesizer.splitForLatencyForTesting(
                phonemes,
                target,
                94,
            ) { value -> KokoroTokenizer.tokenize(context, value).size }
            println(
                "KOKORO_BOUNDARY target=$target " +
                    chunks.map { chunk ->
                        "tokens=${KokoroTokenizer.tokenize(context, chunk).size},chars=${chunk.length},text={$chunk}"
                    }.joinToString(" | "),
            )
        }
    }

    /** Plan-only regression: this runs G2P/front but never generates, records, or plays audio. */
    @Test
    fun longArticleRequestStaysGloballyConditionedInsidePackagedWindows() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue(
            "Packaged arm64 QNN planning is unavailable",
            BuildConfig.QNN_EP_INCLUDED && BuildConfig.KOKORO_QNN_AOT_INCLUDED &&
                Build.SUPPORTED_ABIS.contains("arm64-v8a"),
        )
        val text = "It also pushed yields on the 30-year bond lower by as much as 10 basis " +
            "points to 5.18%, moving it away from its highest level since 2007. Twenty-year " +
            "yields also dropped, leaving investors with tepid demand for a $16 billion " +
            "auction of the securities."
        assertEquals(251, text.length)
        val inspection = KokoroSynthesizer(context, BackendPreference.QNN_HTP).use { synthesizer ->
            synthesizer.inspectGlobalPlanForTesting(
                text = text,
                voice = VoiceCatalog.find("am_puck"),
                speed = KokoroSynthesizer.modelSpeedForRequest(100),
            )
        }

        assertEquals("Unexpected full-request G2P shape", 317, inspection.phonemeChars)
        assertEquals("Source planner regression", 4, inspection.initialChunks)
        assertTrue("Article request fell back to independently conditioned chunks", inspection.globallyConditioned)
        assertTrue(
            "Latency-safe refinement did not split the oversized article cores",
            inspection.finalChunks > inspection.initialChunks,
        )
        assertEquals(inspection.finalChunks, inspection.coreFrames.size)
        assertEquals(inspection.finalChunks, inspection.windowFrames.size)
        assertEquals(inspection.finalChunks, inspection.qnnBuckets.size)
        assertEquals(inspection.finalChunks, inspection.leadingOverlapFrames.size)
        assertEquals(inspection.finalChunks, inspection.trailingOverlapFrames.size)
        assertTrue("A planned window exceeds B640: ${inspection.windowFrames}", inspection.windowFrames.all { it <= 640 })
        assertTrue("A planned tail has no packaged QNN bucket: ${inspection.qnnBuckets}", inspection.qnnBuckets.all { it != null })
        assertTrue(
            "A continuation exceeds the prewarmed B192 latency envelope: ${inspection.coreFrames}",
            inspection.coreFrames.drop(1).all { it <= 160 },
        )
        assertTrue(
            "Article plan escaped the prewarmed B128/B192 set: ${inspection.qnnBuckets}",
            inspection.qnnBuckets.all { it in setOf(128, 192) },
        )
        assertTrue(
            "An internal core has no left shared context: ${inspection.leadingOverlapFrames}",
            inspection.leadingOverlapFrames.drop(1).all { it == 2 },
        )
        assertTrue(
            "An internal core has no right shared context: ${inspection.trailingOverlapFrames}",
            inspection.trailingOverlapFrames.dropLast(1).all { it == 2 },
        )
    }
}
