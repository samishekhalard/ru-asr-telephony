# Results

200 utterances, 37.4 min of audio, 3752 reference words as written, 3827 after number expansion (the WER denominator). Measured 2026-10-24 on arm64 / Darwin.

## Absolute error rates

| engine | channel | WER | 95% CI | CER | RTF |
|---|---|---|---|---|---|
| whisper:whisper-large-v3-turbo | clean | 4.6% | 3.7% – 5.6% | 1.3% | 0.04 |
| whisper:whisper-large-v3-turbo | narrowband | 4.8% | 4.0% – 5.7% | 1.3% | 0.04 |
| whisper:whisper-large-v3-turbo | g711_ulaw | 4.8% | 3.9% – 5.8% | 1.3% | 0.04 |
| whisper:whisper-large-v3-turbo | g711_alaw | 4.9% | 4.0% – 5.8% | 1.3% | 0.04 |

RTF is the median real-time factor: below 1.0 is faster than real time, on one laptop.

## Each channel against the clean control

Paired bootstrap over the same utterances, so differences in utterance difficulty cancel. A positive delta means the channel costs accuracy.

| engine | channel | ΔWER vs clean | 95% CI | verdict |
|---|---|---|---|---|
| whisper:whisper-large-v3-turbo | narrowband | +0.21% | -0.31% – +0.73% | not distinguishable |
| whisper:whisper-large-v3-turbo | g711_ulaw | +0.21% | -0.32% – +0.80% | not distinguishable |
| whisper:whisper-large-v3-turbo | g711_alaw | +0.24% | -0.30% – +0.80% | not distinguishable |

Raw transcripts for every row are in `results/raw/`.
