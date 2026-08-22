import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// The QNN plugin is loaded into the stock ORT Android runtime at runtime. Keep
// all three official artifacts pinned because their JNI/EP ABI must agree.
val ortCoordinate = "com.microsoft.onnxruntime:onnxruntime-android:1.24.3"
// Keep the packaged QAIRT 2.48 contexts on their matching runtime while the
// local compatibility AAR changes only QNN 2.4's Samsung FastRPC node probe.
// See third_party/qualcomm-qnn-samsung-compat/PROVENANCE.md.
val qnnProviderCoordinate = "local:onnxruntime-android-qnn:2.4.0-samsung-sm8650"
val qnnRuntimeCoordinate = "com.qualcomm.qti:qnn-runtime:2.48.0"
val frontModelAsset = "kokoro-v1-front.fp16io.onnx"
// The prepared-front fallback consumes the three sentence-level cut tensors plus
// validity masks.  The monolithic q8 graph accepts text/style/speed instead and
// cannot faithfully recover a globally conditioned slice after a QNN failure.
val generatorModelAsset = "kokoro-generator.masked-dynamic.fp32.onnx"
val istftModelAsset = "kokoro-v1-istft.fp32.onnx"
val repairedQnnBuckets = listOf(64, 96, 128, 192, 208, 224, 256, 320, 384, 512, 640)
fun repairedContextAsset(bucket: Int) = "kokoro-v1-neural-vocoder-b$bucket.qnn248.powmul.ctx.onnx"
fun sourceSpectrumAsset(bucket: Int) = "kokoro-v1-source-spectrum-b$bucket.fp32.onnx"
val qnnB64ContextAsset = repairedContextAsset(64)
val qnnB96ContextAsset = repairedContextAsset(96)
val qnnB128ContextAsset = repairedContextAsset(128)
val qnnB192ContextAsset = repairedContextAsset(192)
val qnnB208ContextAsset = repairedContextAsset(208)
val qnnB224ContextAsset = repairedContextAsset(224)
val qnnB256ContextAsset = repairedContextAsset(256)
val qnnB320ContextAsset = repairedContextAsset(320)
val qnnB384ContextAsset = repairedContextAsset(384)
val qnnB512ContextAsset = repairedContextAsset(512)
val qnnB640ContextAsset = repairedContextAsset(640)
val sharedQnnAcousticBinaryAsset = "kokoro-v1-b64.shared.ctx_qnn.bin"
val sharedQnnMidBinaryAsset = "kokoro-v1-b192.shared.ctx_qnn.bin"
val sharedQnnLargeBinaryAsset = "kokoro-v1-b512.shared.ctx_qnn.bin"
val sharedQnnWrapperAssets = mapOf(
    64 to "kokoro-v1-b64.shared.ctx.onnx",
    96 to "kokoro-v1-b96.shared.ctx.onnx",
    128 to "kokoro-v1-b128.shared.ctx.onnx",
    192 to "kokoro-v1-b192.shared.ctx.onnx",
    208 to "kokoro-v1-b208.shared.ctx.onnx",
    224 to "kokoro-v1-b224.shared.ctx.onnx",
    256 to "kokoro-v1-b256.shared.ctx.onnx",
    320 to "kokoro-v1-b320.shared.ctx.onnx",
    384 to "kokoro-v1-b384.shared.ctx.onnx",
    512 to "kokoro-v1-b512.shared.ctx.onnx",
    640 to "kokoro-v1-b640.shared.ctx.onnx",
)
val qnnContextProducer = "QAIRT_2.48.40_V1_POW_X2_MUL_CPU_SOURCE_SPECTRUM_PER_BUCKET_S24_QUALIFIED"
val kokoroModelAsset = "kokoro-v1.0-q8.onnx"
val kokoroModelExpectedSha256 = "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478"
val kokoroTokenizerAsset = "kokoro-v1.0-tokenizer.json"
val kokoroTokenizerExpectedSha256 = "77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34"
val misakiNativeExpectedSha256 = "2e60a254358206283af794d6bfcf79416bac5fc4d5bc6a4907773afb8d20889f"
val espeakNativeExpectedSha256 = "1a887ac7fad29783369630020708aeeeecfbc2036b8ac7e27898d11bdaba673d"
val englishVoiceIds = listOf(
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
val englishVoiceExpectedSha256 = mapOf(
    "af_alloy" to "c4a6b876047fd7fb472edf4ebd63cfac7c3b958a7cae7c106e8f038ca6308c45",
    "af_aoede" to "4a004c33430762e2461eedb2013fad808ef4ab3121f5300f554476caf58d8361",
    "af_bella" to "f69d836209b78eb8c66e75e3cda491e26ea838a3674257e9d4e5703cbaf55c8b",
    "af_heart" to "d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b",
    "af_jessica" to "a240a5e3c15b43563d6e923bdca8ef5613a23471d9b77653694012435df23bd8",
    "af_kore" to "9be5221b6a941c04b561959b8ff0b06e809444dcc4ab7e75a7b23606f691819e",
    "af_nicole" to "cd2191ab31b914ed7b318416b0e4440fdf392ddad9106a060819aa600a64f59a",
    "af_nova" to "18778272caa0d0eebaea251c35fd635f038434f9eee5e691d02a174bd328414f",
    "af_river" to "00a2bcf82b1d86e8f19902ede58c65ccf6c0e43b44b7d74fad54e5d8933c9c30",
    "af_sarah" to "4409fbc125afabacc615d94db5398d847006a737b0247d6892b7a9a0007a2f0a",
    "af_sky" to "4435255c9744f3f31659e0d714ab7689bf65d9e77ec1cce060f083912614f0b9",
    "am_adam" to "162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716",
    "am_echo" to "3968b92c3c4cd1c4416dbded36c13eaa388a90d5788d02a13e4d781f5f8cf3c3",
    "am_eric" to "e8b5be17edd1e3636901ce7598baafe2dc8dd8ff707a0c23bf9e461add7e2832",
    "am_fenrir" to "c27989f741f7ee34d273a39d8a595cc0837d35f5ced9a29b7cc162614616df43",
    "am_liam" to "52403be32fd047c6a44517cb0bcd6b134f2a18baa73e70ef41651e0eab921ade",
    "am_michael" to "1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1",
    "am_onyx" to "da5d135b424164916d75a68ffb4c2abce3d7d5ccc82dd1ee6cf447ce286145e6",
    "am_puck" to "fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0",
    "am_santa" to "61150cf726ab6c5ed7a99f90a304f91f5a72c00c592e89ec94e5df11c319227a",
    "bf_alice" to "08afa6ba24da61ea5e8efa139e5aadc938d83f0a6da5a900adaf763ac1da5573",
    "bf_emma" to "669fe0647f9dd04fcab92f1439a40eeb4c8b4ab1f82e4996fe3d918ce4a63b73",
    "bf_isabella" to "3754352c4aaa46d17f27654ab7518d65b62ad6163a0f55a5f4330c2da2c4e94f",
    "bf_lily" to "5e0ee32ebe64a467124976b14e69590746f1c4ce41a12b587a50c862edfea335",
    "bm_daniel" to "6b3194bbceffb746733cbc22c8f593dd44e401a71d53895a2dca891bc595a1e8",
    "bm_fable" to "f889083196807b4adb15e9204252165f503b8d33d3982e681c52443c49d798f1",
    "bm_george" to "c4b235a4c1f2cd3b939fed08b899ce9385638b763f7b73a59616c4fc9bd6c9bc",
    "bm_lewis" to "b8f671cef828c30e66fdf0b0756a76bba58f6bb3398cbbf27058642acbcedb97",
)
// Filled only with independently hashed, completed QAIRT artifacts. A partial
// set keeps QNN disabled and excluded from the APK rather than silently mixing
// graph generations.
val expectedQnnContextSha256 = mapOf(
    "kokoro-v1-neural-vocoder-b64.qnn248.powmul.ctx.onnx" to "349e770c06399cb0051d513a85bf3e477b80fd26246b6b4dc2f1e1c40549cd2b",
    "kokoro-v1-neural-vocoder-b96.qnn248.powmul.ctx.onnx" to "364e7fea26648b7da9f1b05423a5214db04006bb538601fdb1e780b63004a31e",
    "kokoro-v1-neural-vocoder-b128.qnn248.powmul.ctx.onnx" to "5a307d4a85f49227ae92fdd5b4cf6193e6ece6a48b44a7dd4090a69b64cc4940",
    "kokoro-v1-neural-vocoder-b192.qnn248.powmul.ctx.onnx" to "2add50ecc4decb903288e08f1816f21b64f7f6207881f0057d6261e33afa6a0c",
    "kokoro-v1-neural-vocoder-b208.qnn248.powmul.ctx.onnx" to "41ed2be1ce623a7dcf37d446bd33b6d444fe8f8604143e4703da4042617875fc",
    "kokoro-v1-neural-vocoder-b224.qnn248.powmul.ctx.onnx" to "0ea097e10f540aa0c3d84b46485c31f855828977dba62f74a9952ec62c02c80c",
    "kokoro-v1-neural-vocoder-b256.qnn248.powmul.ctx.onnx" to "a50345bff3189508f89523b0502ebbd4e46d00c0026a35edc84bc57e9cce3db1",
    "kokoro-v1-neural-vocoder-b320.qnn248.powmul.ctx.onnx" to "7fa7bd8b6bf485de2b55d0bd2b778f3fb7feebef030bd53dd41af6f862ffe84b",
    "kokoro-v1-neural-vocoder-b384.qnn248.powmul.ctx.onnx" to "8b537eaa538890069b1d1657af9df987554ac0401be1fdedc09ad6fc0be29bbb",
    "kokoro-v1-neural-vocoder-b512.qnn248.powmul.ctx.onnx" to "f3b944d3ffb78a481c6bf86a3e359a35e8e25a8559b28f388b6b1c9c99a3d7c6",
    "kokoro-v1-neural-vocoder-b640.qnn248.powmul.ctx.onnx" to "b78df4e656cf1c01b1d4bef0ce958b16051dc7d49fd302b64ad40b73a2917c64",
)
val expectedQnnContextBytes = mapOf(
    "kokoro-v1-neural-vocoder-b64.qnn248.powmul.ctx.onnx" to 52_302_596L,
    "kokoro-v1-neural-vocoder-b96.qnn248.powmul.ctx.onnx" to 52_921_094L,
    "kokoro-v1-neural-vocoder-b128.qnn248.powmul.ctx.onnx" to 67_166_985L,
    "kokoro-v1-neural-vocoder-b192.qnn248.powmul.ctx.onnx" to 54_592_263L,
    "kokoro-v1-neural-vocoder-b208.qnn248.powmul.ctx.onnx" to 102_417_159L,
    "kokoro-v1-neural-vocoder-b224.qnn248.powmul.ctx.onnx" to 104_874_761L,
    "kokoro-v1-neural-vocoder-b256.qnn248.powmul.ctx.onnx" to 101_991_175L,
    "kokoro-v1-neural-vocoder-b320.qnn248.powmul.ctx.onnx" to 120_582_922L,
    "kokoro-v1-neural-vocoder-b384.qnn248.powmul.ctx.onnx" to 181_326_604L,
    "kokoro-v1-neural-vocoder-b512.qnn248.powmul.ctx.onnx" to 119_239_436L,
    "kokoro-v1-neural-vocoder-b640.qnn248.powmul.ctx.onnx" to 135_107_340L,
)
val expectedSourceSpectrumSha256 = mapOf(
    "kokoro-v1-source-spectrum-b64.fp32.onnx" to "055759052a1d180b44b9777c77ce487c0fd0ca1702d8db3a39201c4752e996fb",
    "kokoro-v1-source-spectrum-b96.fp32.onnx" to "b2d3710396798d604177c7e3a3af19414244c3f8b95e9ab6f92c66929471c85b",
    "kokoro-v1-source-spectrum-b128.fp32.onnx" to "156840388433459eacdd6020920f286ba2361d2cb35175cfbfba3d6c693e1ff5",
    "kokoro-v1-source-spectrum-b192.fp32.onnx" to "c055f87b5feb25336059bbe562981235d55c58cee1b4b86ec69bb16e737dae65",
    "kokoro-v1-source-spectrum-b208.fp32.onnx" to "2e996a49f192175f6e3b3cad2fd795774e1f185a62a158dca0e6b2741c25972f",
    "kokoro-v1-source-spectrum-b224.fp32.onnx" to "d9d348b6222314bef082d31546790f71783ebed3d9d0b5174e1e71c1b19a0a8b",
    "kokoro-v1-source-spectrum-b256.fp32.onnx" to "47275a5b9875bb876f7c4e73bbad6ec37f2c1fcd95ed97146dffec1cb3c02deb",
    "kokoro-v1-source-spectrum-b320.fp32.onnx" to "d681855a0036c7527f969e13479d372fc18b6fd687639fe1660834ecb6698ef2",
    "kokoro-v1-source-spectrum-b384.fp32.onnx" to "0a8a4fac6769b93b1d949a5eeea05923a89253d8cb55e5476f6c29232eb12bf8",
    "kokoro-v1-source-spectrum-b512.fp32.onnx" to "2ceb7de19c969fe76c24f29083abb765b83d407fea856a51289fa529f4054bc0",
    "kokoro-v1-source-spectrum-b640.fp32.onnx" to "ddd37f7fefea30166de653799297962a9e1d7cc505f3eb05b9244bcc6fa2b2db",
)
val expectedSourceSpectrumBytes = mapOf(
    "kokoro-v1-source-spectrum-b64.fp32.onnx" to 17_579L,
    "kokoro-v1-source-spectrum-b96.fp32.onnx" to 17_579L,
    "kokoro-v1-source-spectrum-b128.fp32.onnx" to 17_587L,
    "kokoro-v1-source-spectrum-b192.fp32.onnx" to 17_587L,
    "kokoro-v1-source-spectrum-b208.fp32.onnx" to 17_587L,
    "kokoro-v1-source-spectrum-b224.fp32.onnx" to 17_587L,
    "kokoro-v1-source-spectrum-b256.fp32.onnx" to 17_587L,
    "kokoro-v1-source-spectrum-b320.fp32.onnx" to 17_588L,
    "kokoro-v1-source-spectrum-b384.fp32.onnx" to 17_588L,
    "kokoro-v1-source-spectrum-b512.fp32.onnx" to 17_588L,
    "kokoro-v1-source-spectrum-b640.fp32.onnx" to 17_588L,
)
val expectedSharedQnnSha256 = mapOf(
    sharedQnnAcousticBinaryAsset to "f9f87e1d2ec0918354ad00070617d86f3d105dd6217154d107c6802aab5172bf",
    sharedQnnMidBinaryAsset to "09a8c0744a0d9e82eed4c5a5f6a13bcbfdbb3134a20769dfec424bb8a645b92f",
    sharedQnnLargeBinaryAsset to "7086cf86826df0c8d2c57c30eba97d0bfdf39c866e2b921c8c9a398eea022a65",
    sharedQnnWrapperAssets.getValue(64) to "1d5560ae256cf4fa711b7f62c845675492d47c8fbe4020ef472f21c232065224",
    sharedQnnWrapperAssets.getValue(96) to "edf9a4d522ecf0c8853565722a06bcfef2af1b203106a3a2638c8e6b4722eb21",
    sharedQnnWrapperAssets.getValue(128) to "e477aa02b3443d1f4575d225b9dbf99ccce5106aa3b7f905c14b61bfdffbada6",
    sharedQnnWrapperAssets.getValue(192) to "1ecda9b3433e7b6ef8a3ee62ed7c4ecdf07bb2fc39803a85c4ad13751c305157",
    sharedQnnWrapperAssets.getValue(208) to "e0755da7be9b6aed1abac1202aa1862cd80fbb9dbeefb142e4789332e1252f49",
    sharedQnnWrapperAssets.getValue(224) to "f70a2f228666d57c119d00628453eec5aa1a6f740657f953af268a7f061bf80e",
    sharedQnnWrapperAssets.getValue(256) to "d7c9e1b2f5d247e2c562a8c529329a326c459e63e9074f8c5036b33db7ecf640",
    sharedQnnWrapperAssets.getValue(320) to "ed453a6fc461cbae05f0268876f3af5fa44152c35678a933eed653f9dc23ae99",
    sharedQnnWrapperAssets.getValue(384) to "ca308c8d69d915af605eb321afa7333db8691b130896f5a5bc36b026bb5c7144",
    sharedQnnWrapperAssets.getValue(512) to "72fa757ad1bf6e932d24dc6f2c6ef4c265068cf9ccf8068115942528d58db9a3",
    sharedQnnWrapperAssets.getValue(640) to "58aa3b9be1598a5ab651834a5677298d47a95f26af04dc88420d2ffd3fc6faf8",
)
val expectedSharedQnnBytes = mapOf(
    sharedQnnAcousticBinaryAsset to 86_990_848L,
    sharedQnnMidBinaryAsset to 135_729_152L,
    sharedQnnLargeBinaryAsset to 253_485_056L,
    sharedQnnWrapperAssets.getValue(64) to 1_263L,
    sharedQnnWrapperAssets.getValue(96) to 1_263L,
    sharedQnnWrapperAssets.getValue(128) to 1_265L,
    sharedQnnWrapperAssets.getValue(192) to 1_263L,
    sharedQnnWrapperAssets.getValue(208) to 1_261L,
    sharedQnnWrapperAssets.getValue(224) to 1_261L,
    sharedQnnWrapperAssets.getValue(256) to 1_261L,
    sharedQnnWrapperAssets.getValue(320) to 1_262L,
    sharedQnnWrapperAssets.getValue(384) to 1_262L,
    sharedQnnWrapperAssets.getValue(512) to 1_262L,
    sharedQnnWrapperAssets.getValue(640) to 1_264L,
)
val modelFiles = listOf(frontModelAsset, generatorModelAsset).associateWith { asset ->
    file("src/main/assets/$asset").also {
        if (!it.isFile) {
            throw GradleException("Missing split Kokoro model asset: app/src/main/assets/$asset")
        }
    }
}
val qnnContextFiles = listOf(
    qnnB64ContextAsset, qnnB96ContextAsset, qnnB128ContextAsset,
    qnnB192ContextAsset, qnnB208ContextAsset, qnnB224ContextAsset,
    qnnB256ContextAsset, qnnB320ContextAsset, qnnB384ContextAsset,
    qnnB512ContextAsset, qnnB640ContextAsset,
).associateWith { asset ->
    file("src/main/assets/$asset")
}
val sourceSpectrumFiles = repairedQnnBuckets.associate { bucket ->
    sourceSpectrumAsset(bucket) to file("src/main/assets/${sourceSpectrumAsset(bucket)}")
}
fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}
val qnnContextSha256 = mutableMapOf<String, String>()
qnnContextFiles.forEach { (asset, contextFile) ->
    if (contextFile.exists()) {
        // v1 acoustic contexts are intentionally smaller than the old v0.19
        // waveform contexts (roughly 51–62 MiB each).  Validate exact size
        // below rather than applying the old 100 MiB lower bound.
        if (!contextFile.isFile || contextFile.length() < 1_000_000L) {
            throw GradleException("Invalid packaged QNN context size: app/src/main/assets/$asset")
        }
        if (contextFile.length() != expectedQnnContextBytes.getValue(asset)) {
            throw GradleException("Unexpected packaged QNN context byte count: app/src/main/assets/$asset")
        }
        val actualSha256 = sha256(contextFile)
        if (actualSha256 != expectedQnnContextSha256.getValue(asset)) {
            throw GradleException("Unexpected packaged QNN context SHA-256: app/src/main/assets/$asset")
        }
        qnnContextSha256[asset] = actualSha256
    }
}
val sourceSpectrumSha256 = mutableMapOf<String, String>()
sourceSpectrumFiles.forEach { (asset, sourceFile) ->
    if (sourceFile.exists()) {
        if (!sourceFile.isFile || sourceFile.length() != expectedSourceSpectrumBytes.getValue(asset)) {
            throw GradleException("Unexpected CPU source-spectrum byte count: app/src/main/assets/$asset")
        }
        val actualSha256 = sha256(sourceFile)
        if (actualSha256 != expectedSourceSpectrumSha256.getValue(asset)) {
            throw GradleException("Unexpected CPU source-spectrum SHA-256: app/src/main/assets/$asset")
        }
        sourceSpectrumSha256[asset] = actualSha256
    }
}
// All repaired suffix contexts and their exact CPU prefixes must be present.
// There is deliberately no source-model/JIT QNN fallback on the phone.
val qnnAotIncluded = qnnContextFiles.keys.all { qnnContextSha256.containsKey(it) } &&
    sourceSpectrumFiles.keys.all { sourceSpectrumSha256.containsKey(it) }
val sourceSpectrumManifest = if (qnnAotIncluded) {
    repairedQnnBuckets.joinToString("|") { bucket ->
        val asset = sourceSpectrumAsset(bucket)
        "$bucket,$asset,${sourceSpectrumSha256.getValue(asset)},${expectedSourceSpectrumBytes.getValue(asset)}"
    }
} else ""
val sharedQnnFiles = expectedSharedQnnSha256.keys.associateWith { asset -> file("src/main/assets/$asset") }
val presentSharedQnnFiles = sharedQnnFiles.filterValues { it.exists() }
if (presentSharedQnnFiles.isNotEmpty() && presentSharedQnnFiles.size != sharedQnnFiles.size) {
    throw GradleException("Partial shared QNN context set is unsafe: ${presentSharedQnnFiles.keys.sorted()}")
}
presentSharedQnnFiles.forEach { (asset, artifact) ->
    if (!artifact.isFile || artifact.length() != expectedSharedQnnBytes.getValue(asset)) {
        throw GradleException("Unexpected shared QNN artifact byte count: app/src/main/assets/$asset")
    }
    if (sha256(artifact) != expectedSharedQnnSha256.getValue(asset)) {
        throw GradleException("Unexpected shared QNN artifact SHA-256: app/src/main/assets/$asset")
    }
}
// Qualcomm's QNN EP explicitly disables external context weight sharing on
// Android. Keep the audited artifacts in the workspace, but ship and run the
// self-contained embedded context for each static bucket on Android.
val sharedQnnIncluded = false
fun sharedManifest(binary: String, buckets: List<Int>): String = buildString {
    append(binary).append('|').append(expectedSharedQnnSha256.getValue(binary)).append('|')
        .append(expectedSharedQnnBytes.getValue(binary))
    buckets.forEach { bucket ->
        val wrapper = sharedQnnWrapperAssets.getValue(bucket)
        append('|').append(bucket).append(',').append(wrapper).append(',')
            .append(expectedSharedQnnSha256.getValue(wrapper)).append(',')
            .append(expectedSharedQnnBytes.getValue(wrapper))
    }
}
val sharedQnnAcousticManifest = sharedManifest(sharedQnnAcousticBinaryAsset, listOf(64, 96, 128))
val sharedQnnMidManifest = sharedManifest(sharedQnnMidBinaryAsset, listOf(192, 208, 224, 256, 320, 384))
val sharedQnnLargeManifest = sharedManifest(sharedQnnLargeBinaryAsset, listOf(512, 640))
val frontModelSha256 = sha256(modelFiles.getValue(frontModelAsset))
val generatorModelSha256 = sha256(modelFiles.getValue(generatorModelAsset))
val kokoroModelFile = file("src/main/assets/$kokoroModelAsset")
val kokoroTokenizerFile = file("src/main/assets/$kokoroTokenizerAsset")
val istftModelFile = file("src/main/assets/$istftModelAsset")
val misakiNativeFile = file("src/main/jniLibs/arm64-v8a/libmisaki_android.so")
val espeakNativeFile = file("src/main/jniLibs/arm64-v8a/libttsespeak.so")
check(kokoroModelFile.length() == 92_361_116L && sha256(kokoroModelFile) == kokoroModelExpectedSha256) {
    "Pinned Kokoro v1.0 q8 model is missing or changed"
}
check(sha256(kokoroTokenizerFile) == kokoroTokenizerExpectedSha256) {
    "Pinned Kokoro v1.0 tokenizer is missing or changed"
}
check(istftModelFile.isFile && istftModelFile.length() == 16_201L) {
    "Pinned v1 iSTFT CPU suffix is missing or changed"
}
check(sha256(istftModelFile) == "c59d8a097d1ffc424d7d3bf88a31e5784956c8fa37b60814d960c8e23d99da7b") {
    "Pinned v1 iSTFT CPU suffix hash changed"
}
check(sha256(misakiNativeFile) == misakiNativeExpectedSha256) { "Native Misaki library changed" }
val misakiNativeByteText = misakiNativeFile.readBytes().toString(Charsets.ISO_8859_1)
listOf(
    "Java_com_local_kokorotts_NativeMisaki_initialize",
    "Java_com_local_kokorotts_NativeMisaki_phonemize",
).forEach { symbol ->
    check(misakiNativeByteText.contains(symbol)) {
        "Native Misaki library does not export the public-package JNI symbol: $symbol"
    }
}
listOf("C:\\Users\\", "com.derek").forEach { privateMarker ->
    check(!misakiNativeByteText.contains(privateMarker)) {
        "Native Misaki library contains a private build marker: $privateMarker"
    }
}
check(sha256(espeakNativeFile) == espeakNativeExpectedSha256) { "Native eSpeak library changed" }
englishVoiceIds.forEach { voice ->
    val voiceFile = file("src/main/assets/voices_v1/$voice.bin")
    check(voiceFile.length() == 522_240L && sha256(voiceFile) == englishVoiceExpectedSha256.getValue(voice)) {
        "Missing or invalid Kokoro v1.0 English voice: $voice"
    }
}

android {
    namespace = "com.local.kokorotts"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.local.kokorotts"
        minSdk = 27
        targetSdk = 35
        versionCode = 41
        versionName = "1.30.1-kokoro-v1.0-public-jni-repair"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += "arm64-v8a" }
        buildConfigField("boolean", "QNN_EP_INCLUDED", "true")
        buildConfigField("String", "KOKORO_MODEL_ASSET", "\"$kokoroModelAsset\"")
        buildConfigField("String", "KOKORO_MODEL_SHA256", "\"$kokoroModelExpectedSha256\"")
        buildConfigField("String", "KOKORO_TOKENIZER_ASSET", "\"$kokoroTokenizerAsset\"")
        buildConfigField("String", "MISAKI_FRONTEND_SHA256", "\"$misakiNativeExpectedSha256\"")
        buildConfigField("String", "ESPEAK_NATIVE_SHA256", "\"$espeakNativeExpectedSha256\"")
        buildConfigField("String", "KOKORO_FRONT_MODEL_ASSET", "\"$frontModelAsset\"")
        buildConfigField("String", "KOKORO_FRONT_MODEL_SHA256", "\"$frontModelSha256\"")
        buildConfigField("String", "KOKORO_GENERATOR_MODEL_ASSET", "\"$generatorModelAsset\"")
        buildConfigField("String", "KOKORO_GENERATOR_MODEL_SHA256", "\"$generatorModelSha256\"")
        buildConfigField("String", "KOKORO_ISTFT_MODEL_ASSET", "\"$istftModelAsset\"")
        buildConfigField("String", "ORT_RUNTIME_COORDINATE", "\"$ortCoordinate\"")
        buildConfigField("String", "QNN_PROVIDER_COORDINATE", "\"$qnnProviderCoordinate\"")
        buildConfigField("String", "QNN_RUNTIME_COORDINATE", "\"$qnnRuntimeCoordinate\"")
        buildConfigField("boolean", "KOKORO_QNN_AOT_INCLUDED", qnnAotIncluded.toString())
        buildConfigField(
            "String",
            "KOKORO_QNN_SOURCE_SPECTRUM_MANIFEST",
            "\"$sourceSpectrumManifest\"",
        )
        buildConfigField("boolean", "KOKORO_QNN_SHARED_INCLUDED", sharedQnnIncluded.toString())
        buildConfigField("String", "KOKORO_QNN_SHARED_ACOUSTIC_MANIFEST", "\"${if (sharedQnnIncluded) sharedQnnAcousticManifest else ""}\"")
        buildConfigField("String", "KOKORO_QNN_SHARED_MID_MANIFEST", "\"${if (sharedQnnIncluded) sharedQnnMidManifest else ""}\"")
        buildConfigField("String", "KOKORO_QNN_SHARED_LARGE_MANIFEST", "\"${if (sharedQnnIncluded) sharedQnnLargeManifest else ""}\"")
        buildConfigField(
            "String",
            "KOKORO_QNN_B64_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB64ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B96_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB96ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B128_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB128ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B192_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB192ContextAsset else ""}\"",
        )
        buildConfigField("String", "KOKORO_QNN_B208_CONTEXT_ASSET", "\"${if (qnnAotIncluded) qnnB208ContextAsset else ""}\"")
        buildConfigField("String", "KOKORO_QNN_B224_CONTEXT_ASSET", "\"${if (qnnAotIncluded) qnnB224ContextAsset else ""}\"")
        buildConfigField(
            "String",
            "KOKORO_QNN_B64_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB64ContextAsset) else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B96_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB96ContextAsset) else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B128_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB128ContextAsset) else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B192_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB192ContextAsset) else ""}\"",
        )
        buildConfigField("String", "KOKORO_QNN_B208_CONTEXT_SHA256", "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB208ContextAsset) else ""}\"")
        buildConfigField("String", "KOKORO_QNN_B224_CONTEXT_SHA256", "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB224ContextAsset) else ""}\"")
        buildConfigField(
            "String",
            "KOKORO_QNN_B256_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB256ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B320_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB320ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B384_CONTEXT_ASSET",
            "\"${if (qnnAotIncluded) qnnB384ContextAsset else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B256_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB256ContextAsset) else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B320_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB320ContextAsset) else ""}\"",
        )
        buildConfigField(
            "String",
            "KOKORO_QNN_B384_CONTEXT_SHA256",
            "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB384ContextAsset) else ""}\"",
        )
        buildConfigField("String", "KOKORO_QNN_B512_CONTEXT_ASSET", "\"${if (qnnAotIncluded) qnnB512ContextAsset else ""}\"")
        buildConfigField("String", "KOKORO_QNN_B640_CONTEXT_ASSET", "\"${if (qnnAotIncluded) qnnB640ContextAsset else ""}\"")
        buildConfigField("String", "KOKORO_QNN_B512_CONTEXT_SHA256", "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB512ContextAsset) else ""}\"")
        buildConfigField("String", "KOKORO_QNN_B640_CONTEXT_SHA256", "\"${if (qnnAotIncluded) qnnContextSha256.getValue(qnnB640ContextAsset) else ""}\"")
        buildConfigField("String", "KOKORO_QNN_CONTEXT_PRODUCER", "\"$qnnContextProducer\"")
    }
    buildTypes { release { isMinifyEnabled = false } }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
        // Qualcomm's DSP loader needs native libraries extracted to real files.
        jniLibs.useLegacyPackaging = true
    }
    buildFeatures { buildConfig = true }
    androidResources {
        noCompress += "onnx"
        noCompress += "bin"
        ignoreAssetsPatterns += "kokoro-v0_19.int8.onnx"
        ignoreAssetsPatterns += "kokoro-v0_19.mobile-qop-int8.onnx"
        ignoreAssetsPatterns += "kokoro-v0_19.mobile2d.fp32.onnx"
        ignoreAssetsPatterns += "kokoro-v0_19.qnn-qdq.onnx"
        ignoreAssetsPatterns += "kokoro-generator.fp32.onnx"
        // Kept in the source checkout only for historical diagnostics. The
        // installed frontend exclusively uses the packaged Misaki US/GB data.
        ignoreAssetsPatterns += "cmudict_ipa.dict"
        // Retain the older FP32 contexts in the workspace for rollback, but
        // never package them alongside the mutually exclusive B256/B384 pair.
        ignoreAssetsPatterns += "kokoro-generator.masked-b320.qnn248.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-generator.masked-b640.qnn248.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-front.fp32.onnx"
        ignoreAssetsPatterns += "kokoro-generator.masked-b256.qnn248.fp32.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-generator.masked-b384.qnn248.fp32.ctx.onnx"
        // The replaced v1 split contexts are retained in the checkout as
        // source-validation controls only. Shipping uses the full-waveform
        // variants named above for B192/B256/B320/B384.
        ignoreAssetsPatterns += "kokoro-v1-acoustic-b192.qnn248.fp32-vtcm8.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-acoustic-b256.qnn248.fp32.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-acoustic-b320.qnn248.fp32.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-acoustic-b384.qnn248.fp32.ctx.onnx"
        // Retain the proven B192 FP32 context in the checkout as an immediate
        // rollback control, but package only the native-FP16 candidate above.
        ignoreAssetsPatterns += "kokoro-v1-generator-b192.qnn248.fp32-vtcm8.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b208.qnn248.fp32.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b224.qnn248.fp32.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b256.qnn248.fp32.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b320.qnn248.fp32.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b384.qnn248.fp32.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b512.qnn248.fp32-vtcm8-mode0.full.ctx.onnx"
        ignoreAssetsPatterns += "kokoro-v1-generator-b640.qnn248.fp32-vtcm8-mode0.full.ctx.onnx"
        // Superseded v1.18 contexts are preserved as rollback/debug controls,
        // but never coexist in the repaired APK.
        listOf(
            "kokoro-v1-acoustic-b64.qnn248.fp16.ctx.onnx",
            "kokoro-v1-acoustic-b96.qnn248.fp16.ctx.onnx",
            "kokoro-v1-acoustic-b128.qnn248.fp16.ctx.onnx",
            "kokoro-v1-generator-b192.qnn248.native-fp16-vtcm8.full.ctx.onnx",
            "kokoro-v1-generator-b208.qnn248.native-fp16.full.ctx.onnx",
            "kokoro-v1-generator-b224.qnn248.native-fp16.full.ctx.onnx",
            "kokoro-v1-generator-b256.qnn248.native-fp16.full.ctx.onnx",
            "kokoro-v1-generator-b320.qnn248.native-fp16.full.ctx.onnx",
            "kokoro-v1-generator-b384.qnn248.native-fp16.full.ctx.onnx",
            "kokoro-v1-generator-b512.qnn248.native-fp16-vtcm8-mode0.full.ctx.onnx",
            "kokoro-v1-generator-b640.qnn248.native-fp16-vtcm8-mode0.full.ctx.onnx",
        ).forEach { ignoreAssetsPatterns += it }
        ignoreAssetsPatterns += "*.npy"
        ignoreAssetsPatterns += "*.mlex"
        ignoreAssetsPatterns += "vocab.json"
        if (!qnnAotIncluded) {
            // Never package a partial bucket set: shipping QNN is strictly no-JIT.
            ignoreAssetsPatterns += qnnB64ContextAsset
            ignoreAssetsPatterns += qnnB96ContextAsset
            ignoreAssetsPatterns += qnnB128ContextAsset
            ignoreAssetsPatterns += qnnB192ContextAsset
            ignoreAssetsPatterns += qnnB208ContextAsset
            ignoreAssetsPatterns += qnnB224ContextAsset
            ignoreAssetsPatterns += qnnB256ContextAsset
            ignoreAssetsPatterns += qnnB320ContextAsset
            ignoreAssetsPatterns += qnnB384ContextAsset
            ignoreAssetsPatterns += qnnB512ContextAsset
            ignoreAssetsPatterns += qnnB640ContextAsset
            sourceSpectrumFiles.keys.forEach { ignoreAssetsPatterns += it }
        }
        if (!sharedQnnIncluded) {
            expectedSharedQnnSha256.keys.forEach { ignoreAssetsPatterns += it }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation(ortCoordinate)
    implementation(files("libs/onnxruntime-android-qnn-2.4.0-samsung-sm8650.aar"))
    implementation(qnnRuntimeCoordinate)
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
