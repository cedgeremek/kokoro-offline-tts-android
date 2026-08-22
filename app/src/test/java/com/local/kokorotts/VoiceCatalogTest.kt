package com.local.kokorotts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceCatalogTest {
    @Test fun exposesEveryUniqueBundledVoice() {
        val voices = VoiceCatalog.all()
        assertEquals(28, voices.size)
        assertEquals(28, voices.map { it.id }.toSet().size)
        assertTrue(voices.all { it.locale.language == "en" })
    }

    @Test fun savedVoiceOverridesLocaleFactoryDefault() {
        assertEquals("bm_george", VoiceCatalog.resolveDefaultId("bm_george", "USA"))
        assertEquals("af_heart", VoiceCatalog.resolveDefaultId("af_heart", "GBR"))
    }

    @Test fun invalidSavedVoiceFallsBackByLocale() {
        assertEquals("af_alloy", VoiceCatalog.resolveDefaultId(null, "USA"))
        assertEquals("bf_emma", VoiceCatalog.resolveDefaultId("missing_voice", "GBR"))
    }

    @Test fun displayLabelsPreserveStableVoiceIds() {
        val label = VoiceCatalog.displayName(VoiceCatalog.find("bf_emma"))
        assertTrue(label.contains("Emma"))
        assertTrue(label.contains("UK female"))
        assertTrue(label.contains("bf_emma"))
    }
}
