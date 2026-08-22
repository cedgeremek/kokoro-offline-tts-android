package com.local.kokorotts

import android.content.Context
import android.speech.tts.Voice
import java.util.Locale

internal data class KokoroVoice(val id: String, val locale: Locale)

internal object VoiceCatalog {
    private const val SETTINGS_NAME = "kokoro_user_settings"
    private const val DEFAULT_VOICE_KEY = "default_voice"

    private val entries = listOf(
        KokoroVoice("af_alloy", Locale.US), KokoroVoice("af_aoede", Locale.US),
        KokoroVoice("af_bella", Locale.US), KokoroVoice("af_heart", Locale.US),
        KokoroVoice("af_jessica", Locale.US), KokoroVoice("af_kore", Locale.US),
        KokoroVoice("af_nicole", Locale.US), KokoroVoice("af_nova", Locale.US),
        KokoroVoice("af_river", Locale.US), KokoroVoice("af_sarah", Locale.US),
        KokoroVoice("af_sky", Locale.US), KokoroVoice("am_adam", Locale.US),
        KokoroVoice("am_echo", Locale.US), KokoroVoice("am_eric", Locale.US),
        KokoroVoice("am_fenrir", Locale.US), KokoroVoice("am_liam", Locale.US),
        KokoroVoice("am_michael", Locale.US), KokoroVoice("am_onyx", Locale.US),
        KokoroVoice("am_puck", Locale.US), KokoroVoice("am_santa", Locale.US),
        KokoroVoice("bf_alice", Locale.UK), KokoroVoice("bf_emma", Locale.UK),
        KokoroVoice("bf_isabella", Locale.UK), KokoroVoice("bf_lily", Locale.UK),
        KokoroVoice("bm_daniel", Locale.UK), KokoroVoice("bm_fable", Locale.UK),
        KokoroVoice("bm_george", Locale.UK), KokoroVoice("bm_lewis", Locale.UK)
    )
    val default = entries.first()

    fun all(): List<KokoroVoice> = entries

    fun find(id: String?) = entries.firstOrNull { it.id == id } ?: default

    fun isValidId(id: String?): Boolean = entries.any { it.id == id }

    /** Returns a saved user choice, then the locale-specific factory default. */
    internal fun resolveDefaultId(savedId: String?, country: String?): String {
        if (isValidId(savedId)) return savedId!!
        return if (country.equals("GB", true) || country.equals("GBR", true)) "bf_emma" else default.id
    }

    fun preferredDefaultId(context: Context, country: String? = null): String {
        val saved = context.getSharedPreferences(SETTINGS_NAME, Context.MODE_PRIVATE)
            .getString(DEFAULT_VOICE_KEY, null)
        return resolveDefaultId(saved, country)
    }

    fun savePreferredDefault(context: Context, id: String) {
        require(isValidId(id)) { "Unknown Kokoro voice: $id" }
        context.getSharedPreferences(SETTINGS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(DEFAULT_VOICE_KEY, id)
            .apply()
    }

    fun displayName(entry: KokoroVoice): String {
        val shortName = entry.id.substringAfter('_')
        val name = shortName.take(1).uppercase(Locale.US) + shortName.drop(1)
        val region = if (entry.locale.country.equals("GB", true)) "UK" else "US"
        val gender = if (entry.id.getOrNull(1) == 'f') "female" else "male"
        return "$name - $region $gender (${entry.id})"
    }

    fun voices(): List<Voice> = entries.map { entry ->
        Voice(entry.id, entry.locale, Voice.QUALITY_HIGH, Voice.LATENCY_LOW, false, setOf("embeddedTts"))
    }
    fun supports(locale: Locale): Int = when {
        locale.language != "en" -> android.speech.tts.TextToSpeech.LANG_NOT_SUPPORTED
        locale.country.equals("GB", true) -> android.speech.tts.TextToSpeech.LANG_COUNTRY_AVAILABLE
        else -> android.speech.tts.TextToSpeech.LANG_COUNTRY_AVAILABLE
    }
}
