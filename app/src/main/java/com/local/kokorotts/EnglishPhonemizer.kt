package com.local.kokorotts

import android.content.Context
import java.io.File
import java.util.Locale

/** Native offline port of the official Misaki 0.9.4 English pipeline. */
internal class EnglishPhonemizer(private val context: Context) {
    fun phonemize(text: String, locale: Locale): String {
        NativeMisaki.ensureInitialized(context)
        return NativeMisaki.convert(text, locale.country.equals("GB", true))
    }
}

/** Compact binary lookup table emitted by scripts/build_misaki_lexicons.ps1. */
internal class MisakiLexicon private constructor(
    private val bytes: ByteArray,
    private val count: Int,
    private val payloadStart: Int,
    override val british: Boolean,
) : EnglishLexicon {
    override fun lookup(word: String): String? = findExact(word) ?: run {
        if (word != word.lowercase(Locale.US)) findExact(word.lowercase(Locale.US)) else null
    }

    private fun findExact(word: String): String? {
        if (word.any { it.code > 0x7f }) return null
        var low = 0
        var high = count - 1
        while (low <= high) {
            val midpoint = (low + high).ushr(1)
            val start = payloadStart + readLeInt(8 + midpoint * Int.SIZE_BYTES)
            when (val comparison = compareAsciiKey(start, word)) {
                in Int.MIN_VALUE..-1 -> low = midpoint + 1
                in 1..Int.MAX_VALUE -> high = midpoint - 1
                else -> return valueAt(midpoint)
            }
        }
        return null
    }

    internal fun forEachPronunciation(action: (String) -> Unit) {
        for (index in 0 until count) action(valueAt(index))
    }

    private fun valueAt(index: Int): String {
        var valueStart = payloadStart + readLeInt(8 + index * Int.SIZE_BYTES)
        while (bytes[valueStart] != '\t'.code.toByte()) valueStart++
        valueStart++
        var valueEnd = valueStart
        while (bytes[valueEnd] != '\n'.code.toByte()) valueEnd++
        return bytes.copyOfRange(valueStart, valueEnd).toString(Charsets.UTF_8)
    }

    /** Compares the indexed ASCII key at [start] with [word]. */
    private fun compareAsciiKey(start: Int, word: String): Int {
        var index = 0
        var position = start
        while (true) {
            val stored = bytes[position].toInt() and 0xff
            val requested = if (index < word.length) word[index].code else '\t'.code
            if (stored != requested) return stored - requested
            if (stored == '\t'.code) return 0
            index++
            position++
        }
    }

    private fun readLeInt(offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)

    companion object {
        fun fromAsset(context: Context, assetName: String, british: Boolean): MisakiLexicon =
            fromBytes(context.assets.open(assetName).use { it.readBytes() }, british)

        internal fun fromFile(file: File, british: Boolean): MisakiLexicon = fromBytes(file.readBytes(), british)

        internal fun fromBytes(bytes: ByteArray, british: Boolean): MisakiLexicon {
            require(bytes.size >= 8 && bytes.copyOfRange(0, 4).toString(Charsets.US_ASCII) == "MLEX") {
                "Invalid Misaki lexicon header"
            }
            val count = (bytes[4].toInt() and 0xff) or
                ((bytes[5].toInt() and 0xff) shl 8) or
                ((bytes[6].toInt() and 0xff) shl 16) or
                ((bytes[7].toInt() and 0xff) shl 24)
            val payloadStart = 8 + count * Int.SIZE_BYTES
            require(count > 0 && payloadStart in 8..bytes.size) { "Invalid Misaki lexicon index" }
            return MisakiLexicon(bytes, count, payloadStart, british)
        }
    }
}

internal interface EnglishLexicon {
    val british: Boolean
    fun lookup(word: String): String?
}

private class MapEnglishLexicon(private val values: Map<String, String>) : EnglishLexicon {
    override val british = false
    override fun lookup(word: String): String? = values[word] ?: values[word.uppercase(Locale.US)]
}

/** Pure Kotlin normalizer, lookup frontend, and conservative lexical fallback. */
internal object OfflineEnglishG2p {
    private const val MAX_TEXT_CHARS = 8_000
    private const val MAX_DERIVATION_DEPTH = 3
    private val softHyphenAtWrap = Regex("(?<=[A-Za-z])\\u00ad[\\t ]*(?:\\r\\n|[\\r\\n\\f])[\\t ]*(?=[a-z])")
    private val hyphenatedLineWrap = Regex("\\b([A-Za-z][A-Za-z']{1,})[-\\u2010\\u2011][\\t ]*(?:\\r\\n|[\\r\\n\\f])[\\t ]*([a-z][A-Za-z']*)")
    private val hyphenatedLostLineBreak = Regex("\\b([A-Za-z][A-Za-z']{1,})[-\\u2010\\u2011][\\t ]+([a-z][A-Za-z']*)")
    private val whitespace = Regex("[\\s\\u00a0\\u2007\\u202f]+")
    private val textTokens = Regex("[A-Za-z]+(?:'[A-Za-z]+)*(?:-[A-Za-z]+(?:'[A-Za-z]+)*)*|[^A-Za-z'-]+|[-']+")

    fun phonemize(text: String, dictionary: Map<String, String>): String = phonemize(text, MapEnglishLexicon(dictionary))

    fun phonemize(text: String, lexicon: EnglishLexicon): String {
        val clean = normalizeSourceText(text, lexicon).take(MAX_TEXT_CHARS)
        return textTokens.findAll(clean).joinToString("") { match ->
            val token = match.value
            if (!token.any(Char::isLetter)) token else pronounceToken(token, lexicon)
        }.replace(whitespace, " ").trim()
    }

    private fun pronounceToken(word: String, lexicon: EnglishLexicon): String {
        if ('-' !in word) return pronunciationOrNull(word, lexicon) ?: spell(word, lexicon)
        return word.split('-').filter(String::isNotBlank).joinToString(" ") { component ->
            pronunciationOrNull(component, lexicon) ?: spell(component, lexicon)
        }
    }

    fun normalizeSourceText(text: String, dictionary: Map<String, String>): String =
        normalizeSourceText(text, MapEnglishLexicon(dictionary))

    /** Repairs discretionary ebook/PDF wrapping before generic whitespace is collapsed. */
    fun normalizeSourceText(text: String, lexicon: EnglishLexicon): String {
        var normalized = canonicalizeTypography(text.take(MAX_TEXT_CHARS + 1_024))
        normalized = softHyphenAtWrap.replace(normalized, "")
        normalized = normalized.replace("\u00ad", "")
        normalized = repairVisibleWraps(normalized, hyphenatedLineWrap, lexicon)
        normalized = repairVisibleWraps(normalized, hyphenatedLostLineBreak, lexicon)
        return normalized.replace(whitespace, " ").trim()
    }

    private fun canonicalizeTypography(text: String): String = buildString(text.length) {
        text.forEach { character -> append(
            when (character) {
                '\u2018', '\u2019', '\u02bc', '\uff07' -> '\''
                '\u2010', '\u2011' -> '-'
                '\u200b', '\u2060' -> return@forEach
                else -> character
            },
        ) }
    }

    private fun repairVisibleWraps(text: String, pattern: Regex, lexicon: EnglishLexicon): String = pattern.replace(text) { match ->
        val left = match.groupValues[1]
        val right = match.groupValues[2]
        val joined = left + right
        if (pronunciationOrNull(joined, lexicon) != null) joined else "$left-$right"
    }

    private fun pronunciationOrNull(word: String, lexicon: EnglishLexicon, depth: Int = 0): String? {
        if (depth > MAX_DERIVATION_DEPTH) return null
        // Android provides no POS tags; retain the prior unambiguous source convention.
        if (word == "IT" || word == "US") return initialism(word, lexicon)
        if (word.equals("it", ignoreCase = true)) return "ɪt"
        if (word.equals("us", ignoreCase = true)) return "ʌs"
        lexicon.lookup(word)?.let { return kokoroV019Phones(it) }
        return deriveFromKnownStem(word, lexicon, depth)
    }

    private fun initialism(word: String, lexicon: EnglishLexicon): String = word.mapNotNull { letter ->
        lexicon.lookup(letter.toString())
    }.joinToString("").let(::kokoroV019Phones).ifBlank { spell(word, lexicon) }

    /** Small deterministic subset of Misaki Lexicon's productive inflections. */
    private fun deriveFromKnownStem(word: String, lexicon: EnglishLexicon, depth: Int): String? {
        val lower = word.lowercase(Locale.US)
        if (lower.endsWith("s'") && lower.length > 2) return pronunciationOrNull(word.dropLast(1), lexicon, depth + 1)
        if (lower.endsWith("'") && lower.length > 1) return pronunciationOrNull(word.dropLast(1), lexicon, depth + 1)
        if (lower.endsWith("s") && lower.length > 2) {
            val stem = when {
                !lower.endsWith("ss") -> word.dropLast(1)
                lower.endsWith("'s") || (lower.endsWith("es") && !lower.endsWith("ies")) -> word.dropLast(2)
                lower.endsWith("ies") -> word.dropLast(3) + "y"
                else -> ""
            }
            pronunciationOrNull(stem, lexicon, depth + 1)?.let { return it + pluralEnding(it, lexicon.british) }
        }
        if (lower.endsWith("d") && lower.length > 3) {
            val stem = when {
                !lower.endsWith("dd") -> word.dropLast(1)
                lower.endsWith("ed") && !lower.endsWith("eed") -> word.dropLast(2)
                else -> ""
            }
            pronunciationOrNull(stem, lexicon, depth + 1)?.let { return it + pastEnding(it, lexicon.british) }
        }
        if (lower.endsWith("ing") && lower.length > 4) {
            val base = word.dropLast(3)
            val candidates = buildList {
                if (base.length > 2) add(base)
                add(base + "e")
                if (base.length > 2 && base.takeLast(2).let { it[0] == it[1] }) add(base.dropLast(1))
                if (base.endsWith("ck")) add(base.dropLast(1))
            }
            candidates.firstNotNullOfOrNull { pronunciationOrNull(it, lexicon, depth + 1) }?.let { return it + "ɪŋ" }
        }
        return null
    }

    private fun pluralEnding(stem: String, british: Boolean): String = when (stem.lastOrNull()) {
        'p', 't', 'k', 'f', 'θ' -> "s"
        's', 'z', 'ʃ', 'ʒ', 'ʧ', 'ʤ' -> if (british) "ɪz" else "ᵻz"
        else -> "z"
    }

    private fun pastEnding(stem: String, british: Boolean): String = when (stem.lastOrNull()) {
        'p', 'k', 'f', 'θ', 'ʃ', 's', 'ʧ' -> "t"
        'd', 't' -> if (british) "ɪd" else "ᵻd"
        else -> "d"
    }

    /** Misaki's ɾ and ʔ map to Kokoro v0.19's T and t vocabulary entries. */
    internal fun kokoroV019Phones(phones: String): String = phones.replace('ɾ', 'T').replace('ʔ', 't')

    /** Safe rare-OOV fallback: known letter names, no exception or network. */
    private fun spell(word: String, lexicon: EnglishLexicon): String = word.lowercase(Locale.US)
        .mapNotNull { lexicon.lookup(it.toString()) ?: letterIpa[it] }
        .joinToString("").let(::kokoroV019Phones)

    private val letterIpa = mapOf(
        'a' to "ˈA", 'b' to "bˈi", 'c' to "sˈi", 'd' to "dˈi", 'e' to "ˈi",
        'f' to "ˈɛf", 'g' to "ʤˈi", 'h' to "ˈAʧ", 'i' to "ˈI", 'j' to "ʤˈA",
        'k' to "kˈA", 'l' to "ˈɛl", 'm' to "ˈɛm", 'n' to "ˈɛn", 'o' to "ˈO",
        'p' to "pˈi", 'q' to "kjˈu", 'r' to "ˈɑɹ", 's' to "ˈɛs", 't' to "tˈi",
        'u' to "jˈu", 'v' to "vˈi", 'w' to "dˈʌbəljˈu", 'x' to "ˈɛks",
        'y' to "wˈI", 'z' to "zˈi",
    )
}
