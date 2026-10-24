# Russian ASR over a telephone line

*[Русская версия](README.ru.md)*

**Finding: the telephone channel itself costs almost nothing.** Band limiting
to 8 kHz and G.711 companding together move Whisper large-v3-turbo's word
error rate by at most 0.8 percentage points on read Russian speech, and the
measured effect cannot be distinguished from zero.

That is worth knowing because it rules something out. If a voice assistant on
a PBX transcribes badly, the codec and the bandwidth are not the reason.

## Why this measurement did not already exist

The public Russian ASR benchmark compares Whisper, GigaAM and Deepgram across
studio speech, audiobooks and podcasts, and its headline result is that the
ranking of engines flips between domains — no engine wins everywhere. It also
states plainly what it does not cover: **telephony, far-field and noisy
audio, and latency.**

Telephony is where voice assistants on phone lines actually live. Since
rankings are known to move between domains, the honest position was that
nobody could say what happens on a phone line without measuring it.

## What was measured

200 utterances from FLEURS Russian (read news sentences, studio recorded,
human transcripts, CC-BY): 37.4 minutes of audio, 3,752 reference words as
written and 3,827 after number expansion, which is the denominator every rate
below is computed over. The same utterances pass through four channels, and
one engine transcribes all four.

| channel | what it does |
|---|---|
| `clean` | untouched — the control |
| `narrowband` | 16 kHz → 8 kHz → 16 kHz, band limiting only, no codec |
| `g711_ulaw` | band limiting plus µ-law companding (North American PSTN) |
| `g711_alaw` | band limiting plus A-law companding (European PSTN) |

Band limiting and companding are kept apart on purpose. A single "telephony"
number cannot say which of the two a model is losing to, and that is the
question an engineer has to answer before deciding what to fix.

## Results

Whisper large-v3-turbo via MLX, on an M5 MacBook Pro.

| channel | WER | 95% CI | CER | RTF |
|---|---|---|---|---|
| clean | 4.6% | 3.7% – 5.6% | 1.3% | 0.04 |
| narrowband | 4.8% | 4.0% – 5.7% | 1.3% | 0.04 |
| g711_ulaw | 4.8% | 3.9% – 5.8% | 1.3% | 0.04 |
| g711_alaw | 4.9% | 4.0% – 5.8% | 1.3% | 0.04 |

Each channel against the control, paired over the same utterances:

| channel | ΔWER vs clean | 95% CI | verdict |
|---|---|---|---|
| narrowband | +0.21% | −0.31% – +0.73% | not distinguishable |
| g711_ulaw | +0.21% | −0.32% – +0.80% | not distinguishable |
| g711_alaw | +0.24% | −0.30% – +0.80% | not distinguishable |

The interval is the result here, not the point estimate. It says the channel
cost is **bounded above by roughly 0.8 percentage points**. "No measurable
effect" is a weaker and less useful claim than "whatever the effect is, it is
smaller than this".

RTF 0.04 means transcription ran about 25× faster than real time on a laptop.
The public benchmark does not report latency at all.

## The method checks out against an independent result

The clean-channel WER here is **4.6%**. The public benchmark reports **4.44%**
for the same model on the same corpus, over 775 utterances instead of 200.

Reproducing someone else's published number with different code, a different
normaliser and a different sample is the only cheap evidence that the rest of
this pipeline can be trusted. Without that line, these would be numbers from
a laptop.

## A hypothesis that did not survive

Before running anything, the expectation was that band limiting would
dominate: it removes everything above 4 kHz, which is where the energy
distinguishing с, ш, ф and ц lives, and by signal-to-noise ratio it destroys
far more of the waveform than G.711 quantisation does.

The recognition result does not follow the signal measurement. Whisper
appears to rely on what is below 4 kHz, and losing the top of the band costs
it very little. The hypothesis is recorded here because it was wrong, and a
benchmark that only reports confirmed guesses is not reporting much.

## What this does not show

- **FLEURS is read studio speech.** Clear articulation, no background noise,
  one speaker, no interruptions. A real call adds all of those on top of the
  channel. The finding is specifically that *the channel alone* is cheap — not
  that telephone audio is easy.
- **One engine.** GigaAM v3 leads the public benchmark on audiobooks and
  spontaneous speech while placing fourth on studio speech, so there is no
  reason to assume it responds to narrowband the way Whisper does. Adding it
  is the obvious next step; the `Engine` interface is three methods.
- **Simulated, not recorded.** Real call recordings are personal data and are
  not published. Resampling plus G.711 is a reasonable approximation of a
  PSTN path, but it has no handset, no room, no packet loss and no jitter.
- **200 utterances, taken in split order.** Not a random sample. If the start
  of the FLEURS test split is dominated by particular speakers, the sample
  inherits that.
- **One machine.** The RTF describes an M5 MacBook Pro and says nothing about
  a server.

## Method

**Normalisation.** Lowercase, ё → е, strip punctuation, collapse whitespace —
the same steps the public benchmark uses, so the clean numbers are comparable.
One step is added: **numbers are expanded into words on both sides of the
comparison**.

That step is not cosmetic. On the first sample this project ever transcribed,
Whisper heard «номер сорок семь» and wrote «номер 47» — a correct recognition
scored as two errors. Expanding digits changed that utterance's WER from 47.4%
to 26.3%. Telephony is full of order numbers, amounts and dates, so without
expansion a benchmark punishes engines for a formatting choice. It is applied
to the reference too, so neither spelling is privileged.

Its limits are stated in the source: `num2words` produces nominative
cardinals, so an inflected reference («второго марта») will not match an
expanded digit. This inflates WER slightly on dates and ordinals, identically
for every engine.

**Corpus-level rates.** Total errors over total reference words, not the mean
of per-utterance rates. Averaging per-utterance would let one short mangled
clip weigh as much as a long correct one.

**Paired comparison.** Channel differences use a paired bootstrap: utterance
indices are resampled once per draw and both conditions are scored on exactly
that set. Comparing two independent intervals for overlap is the wrong test
and a conservative one, because most of each interval's width comes from
utterances being unequally hard rather than from the channel. There is a test
that constructs a case where independent intervals overlap and the paired
interval still excludes zero.

**Raw transcripts are committed.** Every reference and hypothesis behind every
number is in `results/raw/`. A benchmark whose outputs cannot be inspected is
a claim, not a measurement — recompute the rates, or disagree with the
normalisation and apply your own.

## Reproducing

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m run --limit 200
```

Writes `results/summary.md` and `results/raw/*.json`. About ten minutes on an
M5; the corpus streams, so only the requested utterances are downloaded.

Wiring check without a model, in seconds:

```bash
PYTHONPATH=src python -m run --limit 3 --engines echo
```

`mlx-whisper` is Apple Silicon only. On other hardware, replace the engine —
it needs one method returning a string.

The audio column is decoded with `soundfile` rather than `torchcodec`, so no
torch build has to match the `datasets` version.

## Layout

```
src/
├── data.py      FLEURS streaming and decoding
├── degrade.py   the four channel profiles
├── engines.py   one interface, Whisper behind it
├── metrics.py   normalisation, WER/CER, bootstrap, paired comparison
└── run.py       the benchmark
tests/           62 tests, offline, no model and no network
results/
├── summary.md   the tables above, regenerated
└── raw/         every reference and hypothesis
```

## Tests

```bash
pytest
```

62 tests, none of which need a GPU, a model download or a network
connection. They assert the physics (a 6 kHz tone must not survive an 8 kHz
channel), the arithmetic (corpus-level rather than averaged), and the
reasoning (pairing detects what overlapping intervals miss).

The first test in `test_metrics.py` is the «сорок семь» / «47» failure from
the first transcription this project made, kept as a regression test.

## License

MIT — see [LICENSE](LICENSE).

Sources for the comparison numbers: [stt-ru-benchmark](https://github.com/vakovalskii/stt-ru-benchmark)
and its [write-up](https://habr.com/ru/articles/1042574/).
