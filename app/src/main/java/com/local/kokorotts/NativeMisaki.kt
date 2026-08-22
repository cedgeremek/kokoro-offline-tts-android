package com.local.kokorotts

import android.content.Context
import java.io.File
import java.security.MessageDigest

/** JNI bridge for the pinned Misaki behavioral port and eSpeak OOV fallback. */
internal object NativeMisaki {
    private val lock = Any()
    @Volatile private var initialized = false

    init {
        System.loadLibrary("misaki_android")
    }

    fun ensureInitialized(context: Context) {
        if (initialized) return
        synchronized(lock) {
            if (initialized) return
            val installRoot = File(context.filesDir, "misaki-0.9.4-espeak-1.52.0")
            val dataDirectory = File(installRoot, "espeak-ng-data")
            val marker = File(installRoot, "asset.sha256")
            val assetIdentity = "${BuildConfig.MISAKI_FRONTEND_SHA256}:${BuildConfig.ESPEAK_NATIVE_SHA256}"
            if (!marker.isFile || marker.readText() != assetIdentity) {
                installRoot.deleteRecursively()
                copyAssetTree(context, "espeak-ng-data", dataDirectory)
                marker.parentFile?.mkdirs()
                marker.writeText(assetIdentity)
            }
            val library = File(context.applicationInfo.nativeLibraryDir, "libttsespeak.so")
            check(library.isFile) { "Packaged eSpeak library is missing" }
            check(sha256(library) == BuildConfig.ESPEAK_NATIVE_SHA256) {
                "Packaged eSpeak library identity changed"
            }
            val error = initialize(library.absolutePath, dataDirectory.absolutePath)
            check(error == null) { "Unable to initialize native Misaki frontend: $error" }
            initialized = true
        }
    }

    fun convert(text: String, british: Boolean): String = phonemize(text, british)

    private fun copyAssetTree(context: Context, assetPath: String, destination: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            destination.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destination.outputStream().use(input::copyTo)
            }
            return
        }
        check(destination.mkdirs() || destination.isDirectory) { "Unable to create $destination" }
        children.forEach { child ->
            copyAssetTree(context, "$assetPath/$child", File(destination, child))
        }
    }

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
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private external fun initialize(libraryPath: String, dataPath: String): String?
    private external fun phonemize(text: String, british: Boolean): String
}
