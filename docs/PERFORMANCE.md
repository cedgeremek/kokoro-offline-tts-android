# Performance evidence

All public numbers below are from physical Samsung Galaxy S24 Ultra SM-S928U1
/ Snapdragon 8 Gen 3 (SM8650) runs. No emulator number is mixed into the table.

## v1.30 whole-article callback run

Configuration: open Bloomberg article, rewound to the start, `am_puck`, Android
rate 1.0, model speed 1.3, public Android `SynthesisCallback` path.

| Metric | Value |
|---|---:|
| Log interval | 2026-08-21 22:52:22.610 to 23:00:40.385 |
| Wall time | 497.775 s |
| Callback PCM | 22,102,560 bytes, PCM16 mono at 24 kHz |
| Audio duration | 22,102,560 / 48,000 = 460.470 s |
| Audio/wall | 460.470 / 497.775 = 0.9251 |
| Request plans/completions | 76 / 76 |
| Callback chunks | 379 |
| Generator windows | 379 QNN_HTP; 0 CPU |
| Generator RTF | min 0.26, p50 0.29, p95 0.32, p99 0.36, max 0.39, mean 0.2911 |
| First PCM | min 500.4 ms, p50 949.8, p95 1,372.8, p99 1,581.5, max 1,642.2, mean 967.2 |
| QNN run | min 316.3 ms, p50 600.1, p95 621.5, p99 634.3, max 639.3, mean 575.9 |

Mean generator throughput is `1 / 0.2911 = 3.44x` audio time. This is
generator-only throughput. Whole-session end-to-end wall/audio is
`497.775 / 460.470 = 1.081`, or 8.1% wall overhead above delivered play time.

Acceptance also recorded zero non-global plans, unavailable overlaps, rejected
callback chunks, QNN failures, preflight failures, QNN-disable events, and
clipped samples.

## Historical speedups

| Change | Baseline | Optimized | Calculation |
|---|---:|---:|---:|
| QNN vs q8 first PCM | 1,252 ms | 779 ms | `(1252-779)/1252 = 37.8%` lower; `1252/779 = 1.61x` |
| Duplicate context repair, first PCM | 2,333 ms | 1,130 ms | 51.6% lower; 2.06x |
| Duplicate context repair, total | 4,662 ms | 3,488 ms | 25.2% lower; 1.34x |
| Duplicate context repair, generator RTF | 0.7379 | 0.3104 | 57.9% lower; 2.38x throughput |
| Whole-window to progressive first PCM | 1,257 ms | 779 ms | 38.0% lower; 1.61x |
| Delivery model speed | 1.0 | 1.3 | `1 - 1/1.3 = 23.1%` shorter audio duration |

The last row is a playback-duration calculation and must not be described as
compute acceleration. Generator-only RTF is not interchangeable with Android
end-to-end callback latency.

## Audio/streaming gates

- First article sentence: byte-exact direct/callback PCM parity, zero projected
  starvation, zero clipped samples.
- Former problematic first join: 20,975-sample step reduced to 517 samples,
  below the nearby p95 of 2,420.
- Long `$16 billion auction` regression: 11 HTP windows (B128/B192), direct and
  callback parity, zero starvation, zero clipping.
- Former 10.28 s q8 continuation and 2,626.77 ms B640 / 1,362.21 ms cold B320
  starvation paths were absent from the accepted route.
- Focused duration-planner suite: 42/42.
- Full JVM suite: 53/54; the sole failure is a missing historical fixture, not
  a runtime assertion failure.
