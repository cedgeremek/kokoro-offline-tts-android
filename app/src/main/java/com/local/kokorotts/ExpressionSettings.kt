package com.local.kokorotts

import android.content.Context

/**
 * Deliberately narrow, persisted delivery control. Kokoro v1 has no public, safe
 * pitch or prosody-strength input: its sentence prosody is generated from text and
 * the selected voice style. This setting therefore changes only the existing speed
 * input, once per Android request, and never injects a synthetic pause or DSP pitch
 * shift.
 */
internal object ExpressionSettings {
    private const val PREFERENCES = "kokoro_expression"
    private const val DELIVERY_SPEED_KEY = "delivery_speed_multiplier"

    const val DEFAULT_DELIVERY_SPEED = 1.0f
    const val MIN_DELIVERY_SPEED = 0.85f
    const val MAX_DELIVERY_SPEED = 1.15f

    fun deliverySpeed(context: Context): Float =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getFloat(DELIVERY_SPEED_KEY, DEFAULT_DELIVERY_SPEED)
            .coerceIn(MIN_DELIVERY_SPEED, MAX_DELIVERY_SPEED)

    fun saveDeliverySpeed(context: Context, multiplier: Float) {
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit()
            .putFloat(DELIVERY_SPEED_KEY, multiplier.coerceIn(MIN_DELIVERY_SPEED, MAX_DELIVERY_SPEED))
            .apply()
    }
}
