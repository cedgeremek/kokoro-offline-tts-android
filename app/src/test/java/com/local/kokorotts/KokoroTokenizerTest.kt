package com.local.kokorotts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class KokoroTokenizerTest {
    private val vocab = mapOf('h' to 50L, '\u025b' to 86L, 'l' to 54L, 'o' to 57L, '!' to 5L)

    @Test fun tokenizesEnglishIpaAndPunctuation() {
        assertEquals(5, KokoroTokenizer.tokenize(vocab, "h\u025blo!").size)
    }

    @Test fun ignoresUnsupportedCodepoints() {
        assertTrue(KokoroTokenizer.tokenize(vocab, "hello\uD83D\uDE00").isNotEmpty())
    }

    @Test fun generatorFramesUseOnlyPackagedAotBuckets() {
        assertEquals(64, KokoroSynthesizer.aotBucketForFrames(1))
        assertEquals(64, KokoroSynthesizer.aotBucketForFrames(63))
        assertEquals(64, KokoroSynthesizer.aotBucketForFrames(64))
        assertEquals(96, KokoroSynthesizer.aotBucketForFrames(65))
        assertEquals(96, KokoroSynthesizer.aotBucketForFrames(96))
        assertEquals(128, KokoroSynthesizer.aotBucketForFrames(97))
        assertEquals(128, KokoroSynthesizer.aotBucketForFrames(128))
        assertEquals(192, KokoroSynthesizer.aotBucketForFrames(129))
        assertEquals(192, KokoroSynthesizer.aotBucketForFrames(141))
        assertEquals(192, KokoroSynthesizer.aotBucketForFrames(192))
        assertEquals(208, KokoroSynthesizer.aotBucketForFrames(193))
        assertEquals(208, KokoroSynthesizer.aotBucketForFrames(208))
        assertEquals(224, KokoroSynthesizer.aotBucketForFrames(209))
        assertEquals(224, KokoroSynthesizer.aotBucketForFrames(224))
        assertEquals(256, KokoroSynthesizer.aotBucketForFrames(225))
        assertEquals(256, KokoroSynthesizer.aotBucketForFrames(256))
        assertEquals(320, KokoroSynthesizer.aotBucketForFrames(257))
        assertEquals(320, KokoroSynthesizer.aotBucketForFrames(320))
        assertEquals(384, KokoroSynthesizer.aotBucketForFrames(321))
        assertEquals(384, KokoroSynthesizer.aotBucketForFrames(384))
        assertEquals(512, KokoroSynthesizer.aotBucketForFrames(385))
        assertEquals(512, KokoroSynthesizer.aotBucketForFrames(512))
        assertEquals(640, KokoroSynthesizer.aotBucketForFrames(513))
        assertEquals(640, KokoroSynthesizer.aotBucketForFrames(640))
        assertNull(KokoroSynthesizer.aotBucketForFrames(641))
    }

    @Test fun repairedContextsAllReturnAcousticFeaturesForCpuIstft() {
        listOf(64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640).forEach { bucket ->
            assertTrue(!KokoroSynthesizer.isFullWaveformQnnBucket(bucket))
        }
    }

    @Test fun contiguousShortBucketsPreserveOriginalPunctuationSpans() {
        val frames = mapOf(
            "First clause, Second clause." to 96,
            "Third clause." to 63,
            "Third clause. Fourth clause?" to 128,
            "Fifth clause!" to 80,
            "Fifth clause! Sixth clause." to 641,
        )
        val spans = listOf(
            "First clause,",
            "Second clause.",
            "Third clause.",
            "Fourth clause?",
            "Fifth clause!",
            "Sixth clause.",
        )
        val result = KokoroSynthesizer.coalescePostDurationForTesting(
            spans = spans,
            tokenCount = { it.length },
            frameCount = { frames[it] ?: 96 },
        )
        assertEquals(
            listOf(
                "First clause,",
                "Second clause.",
                "Third clause.",
                "Fourth clause?",
                "Fifth clause!",
                "Sixth clause.",
            ),
            result,
        )
    }

    @Test fun postDurationCoalescingRejectsOversizeAndDoesNotRecurse() {
        val spans = listOf("a.", "b.", "c.")
        val result = KokoroSynthesizer.coalescePostDurationForTesting(
            spans = spans,
            tokenCount = { if (it == "a. b.") 510 else 1 },
            frameCount = { if (it == "a.") 63 else 96 },
        )
        assertEquals(spans, result)
    }

    @Test fun lowLatencyChunkLimitsScaleWithSpeechSpeed() {
        assertEquals(16 to 18, KokoroSynthesizer.chunkTokenLimitsForSpeed(0.8f))
        assertEquals(20 to 23, KokoroSynthesizer.chunkTokenLimitsForSpeed(1.0f))
        assertEquals(50 to 50, KokoroSynthesizer.chunkTokenLimitsForSpeed(2.5f))
        assertThrows(IllegalArgumentException::class.java) {
            KokoroSynthesizer.chunkTokenLimitsForSpeed(0f)
        }
    }

    @Test fun androidSpeechRateUsesFastEngineMultiplierAndSafeModelClamp() {
        assertEquals(1.04f, KokoroSynthesizer.modelSpeedForRequest(80), 0.0001f)
        assertEquals(1.3f, KokoroSynthesizer.modelSpeedForRequest(100), 0.0001f)
        assertEquals(1.95f, KokoroSynthesizer.modelSpeedForRequest(150), 0.0001f)
        assertEquals(2.5f, KokoroSynthesizer.modelSpeedForRequest(200), 0.0001f)
        assertEquals(2.5f, KokoroSynthesizer.modelSpeedForRequest(250), 0.0001f)
        assertEquals(1.04f, KokoroSynthesizer.modelSpeedForRequest(1), 0.0001f)
        assertEquals(2.5f, KokoroSynthesizer.modelSpeedForRequest(999), 0.0001f)
    }

    @Test fun requestBoundaryMovesExistingPauseAheadOfTheFastOnset() {
        assertEquals(
            KokoroSynthesizer.SAMPLE_RATE * 120 / 1_000 to
                KokoroSynthesizer.SAMPLE_RATE * 30 / 1_000,
            KokoroSynthesizer.requestBoundaryEdgeSamplesForTesting(),
        )
        assertEquals(
            KokoroSynthesizer.SAMPLE_RATE * 150 / 1_000,
            KokoroSynthesizer.requestBoundaryEdgeSamplesForTesting().let { it.first + it.second },
        )
    }

    @Test fun deliverySpeedMultiplierStaysWithinTheExistingSafeModelRange() {
        assertEquals(1.495f, KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(1.3f, 1.15f), 0.0001f)
        assertEquals(1.105f, KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(1.3f, 0.85f), 0.0001f)
        assertEquals(2.5f, KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(2.5f, 1.15f), 0.0001f)
        assertEquals(0.8f, KokoroSynthesizer.modelSpeedWithDeliveryMultiplier(0.8f, 0.01f), 0.0001f)
    }

    @Test fun sourcePlannerPrefersFinishedVerbPhraseBeforeRelativeClause() {
        val input = "A few speed boats were racing which caused the crowd to cheer loudly as the announcer watched from the bridge."
        fun words(value: String): Int = Regex("[A-Za-z]+").findAll(value).count()
        val chunks = KokoroSynthesizer.splitSourceForLatencyForTesting(input, 6, 12, ::words)
        assertTrue("source planner did not preserve the complete verb phrase: $chunks", chunks.first().endsWith("racing"))
        assertTrue("relative clause was not kept with its connector: $chunks", chunks[1].startsWith("which"))
    }

    @Test fun sourcePlannerRejectsDeterminerAuxiliaryPrepositionAndOpenConnectorCuts() {
        fun words(value: String): Int = Regex("[A-Za-z]+").findAll(value).count()
        val input = "The boats were racing and the crowd was waiting for the announcer to finish the long race report tonight."
        val chunks = KokoroSynthesizer.splitSourceForLatencyForTesting(input, 4, 12, ::words)
        assertTrue(chunks.none { it.endsWith("The", true) || it.endsWith("were", true) || it.endsWith("and", true) || it.endsWith("for", true) })
    }

    @Test fun sourceBoundariesMapBackToTheExistingFullPhonemeStream() {
        fun tokens(value: String): Int = value.count { !it.isWhitespace() }
        assertEquals(
            listOf("aa", "bb", "cc dd"),
            KokoroSynthesizer.mapSourceBoundariesToPhonemesForTesting("aa bb cc dd", listOf(2, 4), ::tokens),
        )
    }

    @Test fun sentenceAwarePlannerStreamsUsefulOpeningAndBatchesContinuation() {
        fun words(value: String): Int = value.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
        fun sentence(prefix: String, count: Int): String =
            (1..count).joinToString(" ") { "$prefix$it" } + "."

        val settingsSample = sentence("sample", 55)
        val settingsChunks = KokoroSynthesizer.splitForLatencyForTesting(settingsSample, 30, 72, ::words)
        assertEquals(listOf(30, 25), settingsChunks.map(::words))
        assertEquals(55, settingsChunks.sumOf(::words))

        val opening = "\u201c${sentence("open", 78)}\u201d"
        val remainder = sentence("finish", 96)
        val chunks = KokoroSynthesizer.splitForLatencyForTesting(
            "$opening $remainder",
            30,
            72,
            ::words,
        )
        assertEquals(listOf(30, 48, 96), chunks.map(::words))
        assertEquals(174, chunks.sumOf(::words))
        assertTrue("continuation did not preserve the first sentence boundary", chunks[1].endsWith(".\u201d"))
        assertEquals(remainder, chunks[2])
    }

    @Test fun progressiveOpeningKeepsOnlyGenuinelyShortUtterancesWhole() {
        fun words(value: String): Int = value.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
        fun utterance(count: Int): String =
            (1..count).joinToString(" ") { "word$it" } + "."

        val short = utterance(29)
        assertEquals(listOf(short), KokoroSynthesizer.splitForLatencyForTesting(short, 20, 72, ::words))

        for (count in listOf(30, 88, 108)) {
            val planned = KokoroSynthesizer.splitForLatencyForTesting(utterance(count), 20, 72, ::words)
            assertTrue("input with $count tokens did not receive a progressive opening", planned.size > 1)
            assertEquals(20, planned.first().let(::words))
            assertEquals(count, planned.sumOf(::words))
        }
    }

    @Test fun longUnpunctuatedInputStillUsesSemanticBatchesInsteadOf48TokenFragments() {
        val input = (1..200).joinToString(" ") { "word$it" }
        fun words(value: String): Int = value.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
        val chunks = KokoroSynthesizer.splitForLatencyForTesting(input, 30, 72, ::words)
        assertEquals(listOf(30, 72, 98), chunks.map(::words))
    }

    @Test fun progressiveOpeningDoesNotDangleAConnectorPhoneme() {
        val phonemes = "ðˌɪs ɪz ɐn ɪɡzˈæmpəl ʌv spˈiʧ sˈɪnθəsɪs ɪn ˈɪŋɡlɪʃ."
        val chunks = KokoroSynthesizer.splitForLatencyForTesting(phonemes, 26, 94) { it.length }
        assertTrue("opening should end after example, not of", chunks.first().endsWith("ɪɡzˈæmpəl"))
        assertTrue("continuation should retain the connector", chunks[1].startsWith("ʌv "))
        assertEquals(phonemes.replace(" ", ""), chunks.joinToString("").replace(" ", ""))
    }

    @Test fun globalFrontBoundariesUseExactBosTokenEosDurations() {
        val input = "ab cd ef"
        val boundaries = KokoroSynthesizer.durationAlignedBoundariesForTesting(
            input = input,
            chunks = listOf("ab", "cd", "ef"),
            tokenDurations = longArrayOf(1, 2, 3, 4, 5, 6, 7, 8),
        ) { value -> value.count { !it.isWhitespace() } }
        assertEquals(
            listOf(0 to 0, 3 to 12, 6 to 30, input.length to 72),
            boundaries,
        )
    }

    @Test fun oversizedGlobalOpeningMovesOnlyToEarlierSafeWordBoundary() {
        val input = "aa of cc"
        assertEquals(
            3 to 6,
            KokoroSynthesizer.boundedDurationOpeningForTesting(
                input = input,
                originalBoundary = 6 to 10,
                tokenDurations = LongArray(8) { 1L },
                maxFrames = 10,
                countTokens = { value -> value.count { !it.isWhitespace() } },
                isSafeTail = { tail -> tail != "of" },
            ),
        )
    }

    @Test fun undersizedGlobalOpeningMovesForwardToFirstSafeDurationBoundary() {
        val input = "aa of cc dd"
        assertEquals(
            9 to 14,
            KokoroSynthesizer.expandedDurationOpeningForTesting(
                input = input,
                originalBoundary = 3 to 6,
                stopBeforeChar = input.length,
                tokenDurations = LongArray(10) { 1L },
                minFrames = 12,
                maxFrames = 16,
                countTokens = { value -> value.count { !it.isWhitespace() } },
                isSafeTail = { tail -> tail != "of" },
            ),
        )
        assertNull(
            KokoroSynthesizer.expandedDurationOpeningForTesting(
                input = input,
                originalBoundary = 3 to 6,
                stopBeforeChar = input.length,
                tokenDurations = LongArray(10) { 1L },
                minFrames = 12,
                maxFrames = 12,
                countTokens = { value -> value.count { !it.isWhitespace() } },
                isSafeTail = { tail -> tail != "of" },
            ),
        )
    }

    @Test fun oversizedGlobalContinuationUsesPackagedBucketWithSharedContext() {
        assertEquals(128, KokoroSynthesizer.globalOpeningWindowFramesForTesting(126))
        assertEquals(192, KokoroSynthesizer.globalOpeningWindowFramesForTesting(127))
        assertEquals(192, KokoroSynthesizer.globalOpeningWindowFramesForTesting(128))
        assertEquals(192, KokoroSynthesizer.globalContinuationWindowFramesForTesting(188))
        assertEquals(208, KokoroSynthesizer.globalContinuationWindowFramesForTesting(189))
        assertEquals(208, KokoroSynthesizer.globalContinuationWindowFramesForTesting(192))
        assertEquals(256, KokoroSynthesizer.globalContinuationWindowFramesForTesting(193))
        assertEquals(256, KokoroSynthesizer.globalContinuationWindowFramesForTesting(212))
        assertEquals(640, KokoroSynthesizer.globalContinuationWindowFramesForTesting(608))
        assertEquals(640, KokoroSynthesizer.globalContinuationWindowFramesForTesting(609))
        assertEquals(640, KokoroSynthesizer.globalContinuationWindowFramesForTesting(636))
        assertNull(KokoroSynthesizer.globalContinuationWindowFramesForTesting(637))
        assertNull(KokoroSynthesizer.globalContinuationWindowFramesForTesting(639))
    }

    @Test fun globalContextAllocatorReservesBothSidesOfEveryInternalCore() {
        val sentenceFrames = 700
        val cores = listOf(0 to 128, 128 to 226, 226 to 338, 338 to 492, 492 to 700)
        val windows = listOf(192, 192, 192, 192, 256)
        val contexts = cores.mapIndexed { index, core ->
            checkNotNull(
                KokoroSynthesizer.globalContextWindowForTesting(
                    sentenceFrames = sentenceFrames,
                    coreStart = core.first,
                    coreEnd = core.second,
                    windowFrames = windows[index],
                    opening = index == 0,
                ),
            )
        }

        assertEquals(
            listOf(0 to 192, 74 to 266, 186 to 378, 336 to 528, 444 to 700),
            contexts,
        )
        for (index in 0 until cores.lastIndex) {
            val previousRightContext = contexts[index].second - cores[index].second
            val nextLeftContext = cores[index + 1].first - contexts[index + 1].first
            assertTrue("boundary $index has no right overlap", previousRightContext >= 2)
            assertTrue("boundary $index has no left overlap", nextLeftContext >= 2)
        }
    }

    @Test fun oversizedGlobalCoreIsRefinedAgainstTheSameSentenceDurationVector() {
        val input = "opening one two three four tail"
        val initialChunks = listOf("opening", "one two three four", "tail")
        // The semantic core spans 122..682 (T=560), while every individual word remains
        // representable in the prewarmed B192/32-frame envelope.
        val durations = longArrayOf(1, 60, 70, 70, 70, 70, 10, 1)
        fun words(value: String): Int = Regex("[A-Za-z]+").findAll(value).count()

        val refined = checkNotNull(
            KokoroSynthesizer.refineGlobalDurationPartsForTesting(
                input = input,
                chunks = initialChunks,
                tokenDurations = durations,
                countTokens = ::words,
                splitChunk = { part ->
                    val pieces = part.split(Regex("\\s+")).filter(String::isNotEmpty)
                    if (pieces.size < 2) null else {
                        val middle = pieces.size / 2
                        pieces.take(middle).joinToString(" ") to pieces.drop(middle).joinToString(" ")
                    }
                },
            ),
        )
        assertEquals(listOf("opening", "one", "two", "three", "four", "tail"), refined.chunks)
        assertEquals(3, refined.refinements)
        assertEquals(0, refined.bridgeCoalescences)
        assertEquals(
            listOf(
                0 to 0,
                8 to 122,
                12 to 262,
                16 to 402,
                22 to 542,
                27 to 682,
                input.length to 704,
            ),
            refined.boundaries,
        )
        assertEquals(
            listOf(122, 140, 140, 140, 140, 22),
            refined.boundaries.zipWithNext().map { (start, end) -> end.second - start.second },
        )
    }

    @Test fun latencySafeGlobalRefinementCoalescesTinyBridgeAndKeepsHeavyCoresPrewarmed() {
        val heavy = (1..8).map { "heavy$it" }
        val input = (listOf("opening", "previous", "bridge") + heavy).joinToString(" ")
        val initialChunks = listOf("opening", "previous", "bridge", heavy.joinToString(" "))
        // Initial exact cores are 68, 84, 22, and 1024 frames. The 22-frame bridge cannot
        // cover the following large inference, while 84+22 remains a context-safe B192 core.
        val durations = longArrayOf(
            1, 33, 42, 11,
            61, 62, 61, 62,
            66, 67, 66, 66,
            1,
        )
        fun words(value: String): Int = Regex("[A-Za-z]+[0-9]*").findAll(value).count()
        val refined = checkNotNull(
            KokoroSynthesizer.refineGlobalDurationPartsForTesting(
                input = input,
                chunks = initialChunks,
                tokenDurations = durations,
                countTokens = ::words,
                splitChunk = { part ->
                    val pieces = part.split(Regex("\\s+")).filter(String::isNotEmpty)
                    if (pieces.size < 2) null else {
                        val middle = pieces.size / 2
                        pieces.take(middle).joinToString(" ") to pieces.drop(middle).joinToString(" ")
                    }
                },
            ),
        )

        assertEquals(
            listOf(
                "opening",
                "previous bridge",
                "heavy1",
                "heavy2",
                "heavy3",
                "heavy4",
                "heavy5",
                "heavy6",
                "heavy7",
                "heavy8",
            ),
            refined.chunks,
        )
        assertEquals(7, refined.refinements)
        assertEquals(1, refined.bridgeCoalescences)
        val cores = refined.boundaries.zipWithNext().map { (start, end) -> end.second - start.second }
        assertEquals(listOf(68, 106, 122, 124, 122, 124, 132, 134, 132, 134), cores)
        assertTrue("A continuation escaped the prewarmed B192 envelope: $cores", cores.drop(1).all { it <= 160 })
    }

    @Test fun unsplittableOversizedGlobalCoreRetainsIndependentFallback() {
        val input = "opening one two three four tail"
        assertNull(
            KokoroSynthesizer.refineGlobalDurationPartsForTesting(
                input = input,
                chunks = listOf("opening", "one two three four", "tail"),
                tokenDurations = longArrayOf(1, 86, 128, 128, 128, 128, 10, 1),
                countTokens = { value -> Regex("[A-Za-z]+").findAll(value).count() },
                splitChunk = { null },
            ),
        )
    }

    @Test fun seamTrimUsesSustainedActivityAndRetainsRequestedQuietEdges() {
        val leading = 2_400
        val active = 720
        val trailing = 3_000
        val waveform = FloatArray(leading + active + trailing)
        for (index in leading until leading + active) waveform[index] = 0.25f

        val trimmed = KokoroSynthesizer.trimArtificialEdgeSilence(
            waveform,
            keepLeadingSamples = 720,
            keepTrailingSamples = 960,
        )
        assertTrue(trimmed.size < waveform.size)
        assertTrue(trimmed.size >= active + 720 + 960)
        assertTrue(trimmed.any { it == 0.25f })
        assertEquals(active, trimmed.count { it == 0.25f })

        val impulseOnly = FloatArray(4_000).also { it[2_000] = 0.25f }
        assertTrue(
            "A single-sample impulse must not be mistaken for sustained speech",
            KokoroSynthesizer.trimArtificialEdgeSilence(impulseOnly, 720, 960) === impulseOnly,
        )
    }

    @Test fun seamJoinCrossfadesOnlyContinuationAndPreservesPunctuationPause() {
        val first = FloatArray(720 + 20_000).also { values ->
            for (index in 0 until 720) values[index] = 0.2f
        }
        val second = FloatArray(12_000 + 720).also { values ->
            for (index in 12_000 until values.size) values[index] = -0.2f
        }

        fun gapMillis(seam: WaveformSeam): Int {
            val joined = KokoroSynthesizer.joinWaveformsAtSeam(first, second, seam)
            assertTrue(joined.all { it.isFinite() })
            assertEquals(720, joined.count { it == 0.2f })
            assertEquals(720, joined.count { it == -0.2f })
            return (joined.size - 1_440) * 1_000 / KokoroSynthesizer.SAMPLE_RATE
        }

        val expectedGapMillis = linkedMapOf(
            WaveformSeam.REQUEST_BOUNDARY to 150,
            WaveformSeam.CONTINUATION to 70,
            WaveformSeam.COMMA to 430,
            WaveformSeam.SEMICOLON to 450,
            WaveformSeam.COLON to 595,
            WaveformSeam.PERIOD to 560,
            WaveformSeam.QUESTION to 575,
        )
        expectedGapMillis.forEach { (seam, expected) ->
            val actual = gapMillis(seam)
            assertTrue("$seam seam was ${actual}ms, expected about ${expected}ms", kotlin.math.abs(actual - expected) <= 15)
        }

        val continuation = KokoroSynthesizer.joinWaveformsAtSeam(first, second, WaveformSeam.CONTINUATION)
        assertTrue(continuation.all { it.isFinite() })
    }

    @Test fun punctuationClassifierHandlesEbookClosersDashesAndEllipsis() {
        assertEquals(WaveformSeam.PERIOD, KokoroSynthesizer.waveformSeamAfter("ready.\u201d"))
        assertEquals(WaveformSeam.PERIOD, KokoroSynthesizer.waveformSeamAfter("wait\u2026"))
        assertEquals(WaveformSeam.SEMICOLON, KokoroSynthesizer.waveformSeamAfter("aside\u2014"))
        assertEquals(WaveformSeam.QUESTION, KokoroSynthesizer.waveformSeamAfter("ready?]"))
    }

    @Test fun oneChunkLookaheadOverlapsOnlyConsumptionAndKeepsCallbackOnCaller() {
        val callerThread = Thread.currentThread().name
        val secondStarted = CountDownLatch(1)
        val releaseSecond = CountDownLatch(1)
        val activeProducers = AtomicInteger()
        val maximumProducers = AtomicInteger()
        val consumed = Collections.synchronizedList(mutableListOf<Int>())
        val callbackThreads = Collections.synchronizedList(mutableListOf<String>())

        KokoroSynthesizer.consumeOneChunkAhead(
            items = listOf(1, 2, 3),
            produce = { item ->
                val active = activeProducers.incrementAndGet()
                maximumProducers.updateAndGet { previous -> maxOf(previous, active) }
                try {
                    if (item == 2) {
                        secondStarted.countDown()
                        assertTrue(releaseSecond.await(1, TimeUnit.SECONDS))
                    }
                    item * 10
                } finally {
                    activeProducers.decrementAndGet()
                }
            },
            consume = { value ->
                callbackThreads += Thread.currentThread().name
                if (value == 10) {
                    assertTrue("second chunk was not prefetched during consumption", secondStarted.await(1, TimeUnit.SECONDS))
                    releaseSecond.countDown()
                }
                consumed += value
                true
            },
        )

        assertEquals(listOf(10, 20, 30), consumed)
        assertTrue(callbackThreads.all { it == callerThread })
        assertEquals(1, maximumProducers.get())
    }

    @Test fun oneChunkLookaheadPropagatesFailureAndJoinsCancelledProducer() {
        val failure = assertThrows(IllegalStateException::class.java) {
            KokoroSynthesizer.consumeOneChunkAhead(
                items = listOf(1, 2),
                produce = { item -> if (item == 2) error("prefetch failed") else item },
                consume = { true },
            )
        }
        assertEquals("prefetch failed", failure.message)

        val producerStarted = CountDownLatch(1)
        val releaseProducer = CountDownLatch(1)
        val producerExited = CountDownLatch(1)
        KokoroSynthesizer.consumeOneChunkAhead(
            items = listOf(1, 2),
            produce = { item ->
                if (item == 2) {
                    producerStarted.countDown()
                    try {
                        releaseProducer.await()
                    } finally {
                        producerExited.countDown()
                    }
                }
                item
            },
            consume = {
                assertTrue(producerStarted.await(1, TimeUnit.SECONDS))
                false
            },
            cancelProducer = { releaseProducer.countDown() },
        )
        assertTrue("cancelled producer was not joined", producerExited.await(1, TimeUnit.SECONDS))
    }

    @Test fun qnnRetryTouchesOnlyCurrentGenerationControlKeys() {
        assertEquals(
            setOf(
                "qnn_disabled_v1_powmul_source_spectrum_v13",
                "last_failure_reason",
                "qnn_retry_generation_v1_powmul_source_spectrum_v13",
            ),
            KokoroSynthesizer.qnnRetryKeysForTesting(),
        )
    }

    @Test fun embeddedAndroidPowerPolicyIsExplicit() {
        assertEquals(
            "embedded.provider=balanced;embedded.vtcm_mb=8_for_B192_B512_B640;" +
                "embedded.run.qnn.perf_mode=burst;external_shared_weights=disabled_on_android;" +
                "prewarm=B128_B192_B208_max3_lru3;q8_fallback=lazy_only;" +
                "global_front=duration_aligned_prewarmed_B128_B192_max160_bridge_coalesce_refine_depth16;" +
                "opening_runway=duration_min112_exact_B128_promotes_B192;" +
                "request_edge=leading120ms_trailing30ms_fixed150ms;" +
                "pcm=robust_p995_soft_limiter;loudness=speech_band_median_1db_corridor_6db_cap;" +
                "cpu_prefix=source_spectrum_only;cpu_suffix=istft_only",
            KokoroSynthesizer.qnnPerformancePolicyForTesting(),
        )
        assertEquals(
            "PACKAGED_EMBEDDED_AOT_QNN248_MASKED_V1",
            KokoroSynthesizer.qnnSessionSourceForTesting(),
        )
    }

    @Test fun qnnPrewarmRetainsReadingBucketsAlongsideLastValidBucket() {
        val expected = listOf(128, 192, 208)
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(192))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(256))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(224))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(320))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(null))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(225))
        assertEquals(expected, KokoroSynthesizer.qnnPrewarmBuckets(640))
    }

    @Test fun qnnAutoTargetIsLimitedToPhysicalS24UltraFamily() {
        assertTrue(KokoroSynthesizer.isTargetQnnDevice(35, "SM8650", "SM-S928U1", listOf("arm64-v8a")))
        assertTrue(KokoroSynthesizer.isTargetQnnDevice(35, "sm8650", "sm-s928b", listOf("arm64-v8a")))
        assertTrue(!KokoroSynthesizer.isTargetQnnDevice(35, "SM8650", "CPH2583", listOf("arm64-v8a")))
        assertTrue(!KokoroSynthesizer.isTargetQnnDevice(30, "SM8650", "SM-S928U1", listOf("arm64-v8a")))
        assertTrue(!KokoroSynthesizer.isTargetQnnDevice(35, "SM8650", "SM-S928U1", listOf("x86_64")))
    }

    @Test fun pcmCacheSeparatesRetriesAndRejectsMixedBackends() {
        val firstAttempt = KokoroSynthesizer.qnnPcmCacheIdentity("context-a", 0L)
        val retry = KokoroSynthesizer.qnnPcmCacheIdentity("context-a", 1L)
        assertTrue(firstAttempt != retry)
        assertEquals("QNN_HTP", KokoroSynthesizer.soleCacheBackend(setOf("QNN_HTP")))
        assertNull(KokoroSynthesizer.soleCacheBackend(setOf("QNN_HTP", "CPU")))
        assertNull(KokoroSynthesizer.soleCacheBackend(emptySet()))
    }

    @Test fun pcmHeadroomAttenuatesOnlyOverRangeAudio() {
        assertEquals(1f, KokoroSynthesizer.pcmHeadroomGain(floatArrayOf(-0.5f, 0.25f)), 0f)
        val gain = KokoroSynthesizer.pcmHeadroomGain(floatArrayOf(-1.49f, 0.2f, 1.1f))
        assertEquals(0.95f / 1.49f, gain, 1e-6f)
        assertTrue(1.49f * gain <= 0.950001f)

        val pcm = KokoroSynthesizer.encodePcm16(floatArrayOf(-1.49f, 0.2f, 1.1f))
        val samples = ShortArray(pcm.size / 2) { index ->
            val offset = index * 2
            ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort()
        }
        assertTrue(samples.maxOf { kotlin.math.abs(it.toInt()) } <= (32767 * 0.951).toInt())
    }

    @Test fun pcmEncodingLimitsAnImpulseWithoutTurningDownFollowingSpeech() {
        val waveform = FloatArray(24_000) { index ->
            (0.2 * kotlin.math.sin(index * 2.0 * Math.PI * 220.0 / 24_000.0)).toFloat()
        }.also { it[1_000] = 12f }
        val encoded = KokoroSynthesizer.encodePcm16WithDiagnostics(waveform)
        assertEquals("an isolated click still attenuated the entire chunk", 1f, encoded.gain, 0f)
        assertTrue(encoded.limitedSamples > 0)
        assertEquals(12f, encoded.absolutePeak, 0f)

        fun sample(index: Int): Int {
            val offset = index * Short.SIZE_BYTES
            return ((encoded.pcm[offset].toInt() and 0xff) or
                (encoded.pcm[offset + 1].toInt() shl 8)).toShort().toInt()
        }
        val ordinaryIndex = 1_137
        val expected = (waveform[ordinaryIndex] * Short.MAX_VALUE).toInt()
        assertTrue(kotlin.math.abs(sample(ordinaryIndex) - expected) <= 1)
        assertTrue(
            (encoded.pcm.indices step Short.SIZE_BYTES).maxOf { offset ->
                kotlin.math.abs(
                    ((encoded.pcm[offset].toInt() and 0xff) or
                        (encoded.pcm[offset + 1].toInt() shl 8)).toShort().toInt(),
                )
            } <= (Short.MAX_VALUE * 0.951f).toInt(),
        )
    }

    @Test fun chunkLoudnessStabilizerBoundsExcursionsWithoutChangingCadenceOrClipping() {
        fun tone(amplitude: Float): ByteArray = KokoroSynthesizer.encodePcm16(
            FloatArray(KokoroSynthesizer.SAMPLE_RATE / 5) { index ->
                (amplitude * kotlin.math.sin(index * 2.0 * Math.PI * 220.0 / KokoroSynthesizer.SAMPLE_RATE)).toFloat()
            },
        )
        fun peak(pcm: ByteArray): Int = (pcm.indices step Short.SIZE_BYTES).maxOf { offset ->
            val sample = ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort().toInt()
            kotlin.math.abs(sample)
        }

        val stabilizer = ChunkLoudnessStabilizer()
        val reference = tone(0.25f)
        val quiet = tone(0.025f)
        val loud = tone(0.8f)
        assertEquals(1f, stabilizer.stabilize(reference).appliedGain, 0f)

        val quietResult = stabilizer.stabilize(quiet)
        assertTrue("quiet chunk was not recognized as an excursion", quietResult.requestedGain > 5f)
        assertTrue("quiet correction exceeded the 6 dB chunk limit", quietResult.appliedGain <= 1.996f)
        assertTrue("quiet correction was incorrectly clamped to unity", quietResult.appliedGain > 1f)
        assertEquals("stabilization must not change audio duration", quiet.size, quietResult.pcm.size)
        assertTrue("stabilization introduced clipping", peak(quietResult.pcm) <= (Short.MAX_VALUE * 0.951f).toInt())

        val loudResult = stabilizer.stabilize(loud)
        assertTrue("loud correction exceeded the 6 dB chunk limit", loudResult.appliedGain >= 0.501f)
        assertTrue("downward correction was not applied", loudResult.appliedGain < 1f)
        assertEquals("stabilization must not change audio duration", loud.size, loudResult.pcm.size)
        assertTrue("stabilization introduced clipping", peak(loudResult.pcm) <= (Short.MAX_VALUE * 0.951f).toInt())
    }

    @Test fun chunkLoudnessStabilizerLeavesNormalProsodyAndUnmeteredPcmUntouched() {
        fun tone(amplitude: Float): ByteArray = KokoroSynthesizer.encodePcm16(
            FloatArray(KokoroSynthesizer.SAMPLE_RATE / 10) { index ->
                (amplitude * kotlin.math.sin(index * 2.0 * Math.PI * 180.0 / KokoroSynthesizer.SAMPLE_RATE)).toFloat()
            },
        )
        val stabilizer = ChunkLoudnessStabilizer()
        stabilizer.stabilize(tone(0.25f))
        val ordinary = tone(0.23f) // About 0.7 dB below the reference: preserve it exactly.
        assertTrue(stabilizer.stabilize(ordinary).pcm.contentEquals(ordinary))
        val silence = ByteArray(480)
        assertTrue(stabilizer.stabilize(silence).pcm === silence)
    }

    @Test fun chunkLoudnessStabilizerIsNotFooledByLowFrequencyEnergy() {
        fun mixed(speechAmplitude: Double, rumbleAmplitude: Double): ByteArray =
            KokoroSynthesizer.encodePcm16(
                FloatArray(KokoroSynthesizer.SAMPLE_RATE / 2) { index ->
                    (speechAmplitude * kotlin.math.sin(
                        index * 2.0 * Math.PI * 440.0 / KokoroSynthesizer.SAMPLE_RATE,
                    ) + rumbleAmplitude * kotlin.math.sin(
                        index * 2.0 * Math.PI * 35.0 / KokoroSynthesizer.SAMPLE_RATE,
                    )).toFloat()
                },
            )

        val stabilizer = ChunkLoudnessStabilizer()
        stabilizer.stabilize(mixed(speechAmplitude = 0.20, rumbleAmplitude = 0.0))
        val quietSpeechWithRumble = stabilizer.stabilize(
            mixed(speechAmplitude = 0.035, rumbleAmplitude = 0.20),
        )
        assertTrue(
            "speech-band meter treated rumble as audible speech",
            quietSpeechWithRumble.requestedGain > 2f,
        )
        assertTrue(quietSpeechWithRumble.appliedGain > 1.5f)
        assertTrue(quietSpeechWithRumble.rampSamples >= KokoroSynthesizer.SAMPLE_RATE / 20)
    }

    @Test fun globalPcmOverlapJoinerRemovesTheWindowStepWithoutChangingCadence() {
        fun constantPcm(sample: Int, count: Int): ByteArray = ByteArray(count * Short.SIZE_BYTES).also { pcm ->
            for (index in 0 until count) {
                val offset = index * Short.SIZE_BYTES
                pcm[offset] = (sample and 0xff).toByte()
                pcm[offset + 1] = (sample shr 8).toByte()
            }
        }
        fun decode(pcm: ByteArray): IntArray = IntArray(pcm.size / Short.SIZE_BYTES) { index ->
            val offset = index * Short.SIZE_BYTES
            ((pcm[offset].toInt() and 0xff) or (pcm[offset + 1].toInt() shl 8)).toShort().toInt()
        }

        val joiner = GlobalPcmOverlapJoiner()
        val first = joiner.stitch(
            constantPcm(sample = 5_000, count = 1_100),
            leadingHalfOverlapSamples = 0,
            trailingHalfOverlapSamples = 100,
        )
        val second = joiner.stitch(
            constantPcm(sample = 1_000, count = 1_100),
            leadingHalfOverlapSamples = 100,
            trailingHalfOverlapSamples = 0,
        )
        joiner.requireComplete()
        val output = decode(first + second)

        assertEquals("shared overlap changed the global sample count", 2_000, output.size)
        assertEquals(5_000, output.first())
        assertEquals(1_000, output.last())
        assertEquals("first crossfade hand-off is discontinuous", 0, kotlin.math.abs(output[900] - output[899]))
        assertEquals("second crossfade hand-off is discontinuous", 0, kotlin.math.abs(output[1_100] - output[1_099]))
        assertTrue(
            "raised-cosine overlap retained an audible step",
            output.asList().zipWithNext().maxOf { (a, b) -> kotlin.math.abs(b - a) } <= 40,
        )
    }

    @Test fun pcmHeadroomRejectsEmptyStaticAndNonFiniteAudio() {
        assertThrows(IllegalStateException::class.java) {
            KokoroSynthesizer.pcmHeadroomGain(floatArrayOf())
        }
        assertThrows(IllegalStateException::class.java) {
            KokoroSynthesizer.pcmHeadroomGain(floatArrayOf(0.1f, 0.1f, 0.1f))
        }
        assertThrows(IllegalStateException::class.java) {
            KokoroSynthesizer.pcmHeadroomGain(floatArrayOf(0f, Float.NaN))
        }
    }
}
