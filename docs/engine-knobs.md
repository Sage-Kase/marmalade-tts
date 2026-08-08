# Engine knobs

Every per-engine parameter marmalade-tts exposes, plus how to pass it.

Three ways to set any knob:
- **CLI flag** — for per-utterance overrides (`--speed 1.4`, `--emotion Happy`)
- **Config file** — for persistent defaults (`engines.<name>.<key>` in
  `~/.config/marmalade-tts/config.yaml`)
- **`config set` shortcut** — `marmalade-tts config set engines.coqui.speaker p225`

The CLI flag always wins over the config value. A knob with no flag is
config-only (noted below).

---

## Universal knobs (all engines)

| Knob       | CLI flag             | Config key            | Notes                                |
|------------|----------------------|-----------------------|--------------------------------------|
| voice      | `--voice` / positional | `engines.<eng>.voice` / `.model` | Engine-specific format               |
| speed      | `--speed`            | `defaults.speed`      | 1.0 = normal, >1 = faster            |
| device     | —                    | `defaults.device` / `engines.<eng>.device` | `cpu` or `cuda`                       |
| daemon     | —                    | `engines.<eng>.daemon`| Keep model in RAM (faster, more memory) |

**`--speed` contract:** every engine honors `--speed`. Engines with a
native tempo/length-scale knob (most of them) pass it through; pocket has
no native knob and falls back to sox post-processing. See
[ENGINE-GUIDE.md § Honoring --speed](../ENGINE-GUIDE.md#honoring---speed-required)
for the rule and how new engines must implement it.

**`--lang` contract:** kokoro (misaki letter codes), kitten (espeak voice
names) and coqui (IETF codes, multilingual models) take a language
setting — engine classes declare this via `SUPPORTS_LANG`. For every
other engine the language is a property of the voice/model (piper encodes
it in the voice file; pocket/matcha/emojivoice are English-only), so
`--lang` there prints a stderr warning and is dropped rather than
silently ignored.

`--lang auto` works on kokoro and kitten (any other engine is an error).
It detects the language once per utterance, from that utterance's text,
and changes only the pronunciation language — never the voice. An
uncertain detection falls back to the engine's normal precedence.

- **kokoro** — detected English resolves to the voice's own English
  variant (British for `george`, American otherwise). Abstaining falls
  back to config lang, then the voice's natural language, then `a`.
- **kitten** — the model is trained on English, so a non-English
  utterance comes out accented. Max's 2026-08-08 call is that this beats
  reading Spanish through English letter rules; for real non-English
  quality use kokoro. English and abstention are byte-identical to the
  plain English path.

---

## kitten

Fast, lightweight, trained on English. 8 voices, 3 model sizes.

| Knob       | CLI       | Config key                   | Default | Notes                                  |
|------------|-----------|------------------------------|---------|----------------------------------------|
| voice      | positional / `--voice` | `engines.kitten.voice`       | `Kiki`  | One of: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo |
| lang       | `--lang`  | `engines.kitten.lang`        | `en-us` | espeak voice name — `en-us`, `es`, `fr-fr`, `it`, `pt-br`, `ja`, `hi`; or `auto` to detect per utterance |
| model_size | —         | `engines.kitten.model_size`  | `nano`  | `nano` (fp32 ~57MB, fastest + best quality), `micro` (int8 ~41MB), `mini` (int8 ~78MB) — upstream ships micro/mini only as dynamic-int8, which sounds worse and runs slower than fp32 nano |

Kitten phonemizes through espeak, so `--lang` really does change the
pronunciation — but the model only ever learned English IPA, so another
language comes out **accented**, not native. Max's 2026-08-08 call: that
beats reading Spanish through English letter rules. When the language
matters, use kokoro. The voice never changes: `--lang` moves the
phonemizer alone.

The detector's ISO codes are accepted and translated (`fr` → `fr-fr`,
`pt` → `pt-br`), and anything else goes to espeak verbatim, so its own
voice names work too — `--lang cmn` for Mandarin, `--lang en-gb` for
British letter rules. Non-Latin scripts (`ja`, `hi`, `cmn`) produce IPA
outside Kitten's token vocabulary; the unknown tokens are dropped rather
than rejected, so it degrades instead of failing.

```bash
marmalade-tts kitten Hugo "Hello"
marmalade-tts kitten Bella --lang auto "Hola, ¿cómo estás?"
marmalade-tts config set engines.kitten.model_size mini
```

---

## kokoro

High quality, multilingual (English, Japanese, Mandarin). 14 voices.

| Knob   | CLI         | Config key             | Default     | Notes                                          |
|--------|-------------|------------------------|-------------|------------------------------------------------|
| voice  | positional / `--voice` | `engines.kokoro.voice` | `af_heart`  | Bare name (`george`) or canonical ID (`bm_george`) |
| lang   | `--lang`    | `engines.kokoro.lang`  | voice's natural language | `a`/`b`/`j`/`z` (American/British/Japanese/Mandarin), or `auto` to detect per utterance |

Voice and language are orthogonal — `george` (British male) can speak
Japanese with `--lang j` for an accent effect. The kokoro daemon serves
every language from one loaded model (`KOKORO_LANG` only picks which G2P
front end is pre-warmed at startup), so no restart is needed to switch —
the single-loaded-model daemon caveat below does not apply here.

```bash
marmalade-tts kokoro george "Hello"
marmalade-tts kokoro --voice af_heart --lang j "konnichiwa"
```

---

## piper

Very fast ONNX engine. Thousands of community voices.

| Knob          | CLI         | Config key                       | Default        | Notes                                                  |
|---------------|-------------|----------------------------------|----------------|--------------------------------------------------------|
| model         | `--voice`   | `engines.piper.model`            | en_US-lessac-medium | Path to a `.onnx` voice model                          |
| speaker       | `--speaker` | —                                | none           | Integer — multi-speaker model only                     |
| noise_scale   | —           | `engines.piper.noise_scale`      | 0.667          | Timbre variation. Lower = more monotone but consistent; higher = livelier but more variable |
| noise_w_scale | —           | `engines.piper.noise_w_scale`    | 0.8            | Per-phoneme duration variation. Lower = robotic pacing; higher = more natural cadence variation |

`noise_scale` and `noise_w_scale` are the standard Piper expressivity
knobs — config-only by design (they're tuning knobs, not per-utterance
choices). Browse voices at <https://rhasspy.github.io/piper-samples/>.

**Installed voices:** `marmalade-tts install piper` fetches six voices
into `~/.local/share/piper/voices/` (all medium quality, 22.05 kHz;
`marmalade-tts piper --list` shows them with their locales):

| Voice                  | Language | Upstream dataset license |
|------------------------|----------|--------------------------|
| `en_US-lessac-medium`  | English (US) | Blizzard 2013 research licence (Lessac) |
| `de_DE-thorsten-medium`| German   | CC0 (Thorsten-Voice)     |
| `fr_FR-siwis-medium`   | French   | CC-BY 4.0 (SIWIS)        |
| `es_ES-davefx-medium`  | Spanish (Spain) | CC0 (OHF-Voice)   |
| `nl_NL-pim-medium`     | Dutch    | CC0 (OHF-Voice)          |
| `it_IT-paola-medium`   | Italian  | CC0-1.0                  |

Piper has no Japanese voice: upstream ships none, and piper's espeak-ng
G2P can't produce the per-mora pitch accent Japanese needs anyway — use
kokoro (`--lang j`) for Japanese. The language lives in the voice model,
not a flag: pick a voice, get its language.

```bash
marmalade-tts piper --voice ~/voices/en_US-amy-medium.onnx "Hello"
marmalade-tts config set engines.piper.noise_scale 0.85   # livelier
```

**Daemon caveat:** `--voice` switches the model on the subprocess path
only. The daemon loads one model at startup (set via `PIPER_MODEL` env
var); to switch models with the daemon enabled, restart it with the new
env var, or run that call with `engines.piper.daemon: false`. Same
caveat applies to matcha — they share the single-loaded-model daemon
shape. Fixing this across engines is on the [roadmap](ROADMAP.md).

---

## coqui

Research-grade. Many models, each honoring a different subset of knobs.

| Knob        | CLI            | Config key                   | Default | Notes                                                 |
|-------------|----------------|------------------------------|---------|-------------------------------------------------------|
| model       | `--voice`      | `engines.coqui.model`        | tts_models/en/ljspeech/tacotron2-DDC | Coqui model spec; `marmalade-tts coqui --list`        |
| speed       | `--speed`      | `defaults.speed`             | 1.0     | Honored by many but not all models                    |
| speaker     | `--speaker`    | `engines.coqui.speaker`      | —       | Speaker name — multi-speaker (e.g. VITS-VCTK `p225`) |
| speaker_idx | —              | `engines.coqui.speaker_idx`  | —       | Integer alternative to `speaker`                      |
| language    | `--lang`       | `engines.coqui.language`     | —       | IETF code (`en`, `es`, `fr`, …) — multilingual models |
| speaker_wav | `--speaker-wav`| `engines.coqui.speaker_wav`  | —       | Reference WAV for XTTS voice cloning                  |
| emotion     | `--emotion`    | `engines.coqui.emotion`      | —       | Emotion label — Tortoise + some VITS variants         |

**Per-model knob support:** Coqui passes any set knob straight through to
`TTS.tts_to_file`; the model ignores knobs it doesn't understand. The
rough mapping:

| Model family        | speed | speaker | language | speaker_wav | emotion |
|---------------------|-------|---------|----------|-------------|---------|
| Tacotron2 (ljspeech)| ✓     | —       | —        | —           | —       |
| VITS (single)       | ✓     | —       | —        | —           | —       |
| VITS-VCTK (multi)   | ✓     | ✓       | —        | —           | —       |
| YourTTS             | ✓     | ✓       | ✓        | ✓           | —       |
| XTTS v2             | ✓     | ✓       | ✓        | ✓           | —       |
| Tortoise            | —     | —       | —        | ✓           | ✓       |
| Capacitron / emo VITS | ✓   | ✓       | —        | —           | ✓       |

Rows reflect upstream Coqui model cards; marmalade's test suite only
exercises the models it ships in `presets`. If a knob you set is silently
ignored, the model just didn't honor it — Coqui doesn't error on
unrecognized kwargs.

XTTS v2 is the marquee model — voice cloning from a 6-second reference
WAV in 17 languages. See [voice-cloning.md](voice-cloning.md) for the
end-to-end setup including reference-clip quality tips.

```bash
# Multi-speaker VITS-VCTK
marmalade-tts coqui --voice tts_models/en/vctk/vits --speaker p225 "Hello"

# XTTS voice cloning, Spanish output
marmalade-tts coqui --voice tts_models/multilingual/multi-dataset/xtts_v2 \
                    --speaker-wav ~/refs/my-voice.wav \
                    --lang es \
                    "Hola, soy yo"

# Tortoise with emotion
marmalade-tts coqui --voice tts_models/en/ek1/tacotron2 --emotion Happy "Hello"
```

---

## pocket

CPU-only, ~200ms latency, voice cloning from any WAV. English only.

| Knob  | CLI         | Config key            | Default | Notes                                            |
|-------|-------------|-----------------------|---------|--------------------------------------------------|
| voice | positional / `--voice` | `engines.pocket.voice`| `alba`  | Built-in name OR path to a `.wav` / `.safetensors` |
| speed | `--speed`   | `defaults.speed`      | 1.0     | Via sox post-process (pocket-tts has no native knob) |

Voice cloning is positional: `marmalade-tts pocket my_voice.wav "Hello"`.
See [voice-cloning.md](voice-cloning.md) for the full how-to.

```bash
marmalade-tts pocket fantine "Hello"
marmalade-tts pocket ~/refs/me.wav "Cloned from my voice"
```

---

## matcha

Fast flow-matching TTS. Quality knobs control the ODE solver.

| Knob        | CLI         | Config key                   | Default            | Notes                                                              |
|-------------|-------------|------------------------------|--------------------|--------------------------------------------------------------------|
| model       | `--voice`   | `engines.matcha.model`       | `matcha_ljspeech`  | `matcha_ljspeech` (1F), `matcha_vctk` (multi), or `.ckpt` path     |
| speaker     | `--speaker` | `engines.matcha.spk`         | —                  | Integer 0–107 for `matcha_vctk`                                    |
| speed       | `--speed`   | `defaults.speed`             | 1.0                | Inverted into `length_scale` (matcha's higher = slower convention) |
| steps       | —           | `engines.matcha.steps`       | 10                 | Solver iterations. 10 = fast (default), 50 ≈ 5× slower but less robotic |
| temperature | —           | `engines.matcha.temperature` | 0.667              | Sampling stochasticity                                             |

```bash
marmalade-tts matcha --voice matcha_vctk --speaker 42 "Hello"
marmalade-tts config set engines.matcha.steps 50    # quality over speed
```

---

## emojivoice

Emoji-driven expressive TTS (Matcha-TTS + per-speaker emoji→style
checkpoints). Currently ships the `paige` speaker.

| Knob        | CLI         | Config key                       | Default | Notes                                       |
|-------------|-------------|----------------------------------|---------|---------------------------------------------|
| voice       | positional / `--voice` | `engines.emojivoice.voice`       | `paige` | Only `paige` ships today                    |
| speed       | `--speed`   | `defaults.speed`                 | 1.0     | Defaults to 0.8 (EmojiVoice's expressive scale) if not overridden |
| steps       | —           | `engines.emojivoice.steps`       | 10      | Same as matcha                              |
| temperature | —           | `engines.emojivoice.temperature` | 0.667   | Same as matcha                              |

Emotional style is set by **including an emoji in the text** —
`marmalade-tts emojivoice "I can't believe it 🤣"`. Supported emojis:
😍 😡 😎 😭 🙄 😁 🙂 🤣 😮 😅 🤔. The emoji is stripped before
synthesis; without one, the neutral style is used.

---

## api

Hosted OpenAI-compatible `/audio/speech` client (Venice by default —
`marmalade-tts api --list` shows the live model + voice lineup).

| Knob        | CLI         | Config key                | Default                          | Notes                                        |
|-------------|-------------|---------------------------|----------------------------------|----------------------------------------------|
| voice       | `--voice`   | `engines.api.voice`       | `af_heart`                       | Model-dependent (open set) — no positional voice |
| speed       | `--speed`   | `defaults.speed`          | 1.0                              | Sent natively (`speed`, 0.25–4.0 on most providers) |
| model       | —           | `engines.api.model`       | `tts-kokoro`                     | Provider model ID                            |
| base_url    | —           | `engines.api.base_url`    | `https://api.venice.ai/api/v1`   | Any OpenAI-compatible host                   |
| api_key_env | —           | `engines.api.api_key_env` | `VENICE_API_KEY`                 | Env var read for the key; `engines.api.api_key` inlines it |
| timeout     | —           | `engines.api.timeout`     | 120                              | Seconds per request                          |
| extra       | —           | `engines.api.extra`       | `{}`                             | Merged into the request payload verbatim (e.g. OpenAI's `instructions`) |

---

## See also

- [ENGINE-GUIDE.md](../ENGINE-GUIDE.md) — adding a new engine
- [config-default.yaml](../config-default.yaml) — annotated default config
- `marmalade-tts --list-effects` — audio post-processing (separate from engine knobs)
