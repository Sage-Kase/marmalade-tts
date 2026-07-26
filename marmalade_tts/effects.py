"""
Audio effects post-processing for marmalade-tts.

Effects are applied after synthesis using sox. Each effect is a named
transformation with optional parameters. Multiple effects are chained
in a single sox invocation — except `bitcrush`, whose bit-depth
quantization needs a 16-bit file to round through and so splits the
chain in two (see build_sox_stages).

Built-in presets combine effects for common use cases (cave, telephone, etc.).
Custom presets can be defined in config under effects.presets.

Dependencies:
  sox — required for any effect processing
       apt install sox   /   brew install sox

Usage (CLI):
  marmalade-tts "Hello" --effect reverb=50
  marmalade-tts "Hello" --effect pitch=200 --effect reverb=30
  marmalade-tts "Hello" --effect cave          # named preset
  marmalade-tts --list-effects                 # show all effects and presets
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

# ── Effect definitions ────────────────────────────────────────────────────────
# Each entry: (sox_args_template, description, param_description)
# {value} is replaced by the user-supplied parameter.

EFFECTS = {
    # name          sox args (list)                                         description                           param hint
    "reverb":   (lambda p: ["reverb", str(p or 50)],
                 "Add room reverb",                                         "amount 0-100 (default 50)"),
    "pitch":    (lambda p: ["pitch", str(p or 100)],
                 "Shift pitch in cents (100 cents = 1 semitone)",           "cents, e.g. 200 (up) or -300 (down)"),
    "tempo":    (lambda p: ["tempo", str(p or 1.2)],
                 "Change speed without shifting pitch",                     "factor, e.g. 1.2 (faster) or 0.8 (slower)"),
    "echo":     (lambda p: _parse_echo(p),
                 "Add echo/delay",                                          "gain-in:gain-out:delay-ms:decay, e.g. 0.8:0.88:60:0.4"),
    "overdrive":(lambda p: ["overdrive", str(p or 20)],
                 "Add overdrive/distortion (robotic quality)",              "gain 1-100 (default 20)"),
    "flanger":  (lambda p: ["flanger"],
                 "Add flanger modulation (sci-fi wobble)",                  "no parameter needed"),
    "chorus":   (lambda p: _parse_chorus(p),
                 "Add chorus (doubled-voice effect)",                       "optional: gain-in:gain-out:delay:decay:speed:depth"),
    "treble":   (lambda p: ["treble", str(p or 6)],
                 "Boost or cut high frequencies (EQ)",                     "dB, e.g. 6 (boost) or -6 (cut)"),
    "bass":     (lambda p: ["bass", str(p or 6)],
                 "Boost or cut low frequencies (EQ)",                      "dB, e.g. 6 (boost) or -6 (cut)"),
    "bandpass": (lambda p: _parse_bandpass(p),
                 "Bandpass filter — keep only a frequency range",           "low-hz:high-hz, e.g. 300:3400 (telephone)"),
    "speed":    (lambda p: ["speed", str(p or 1.2)],
                 "Change speed AND pitch together",                         "factor, e.g. 1.2 (faster+higher)"),
    "vol":      (lambda p: ["vol", str(p or 2.0)],
                 "Adjust volume",                                           "factor, e.g. 2.0 (double) or 0.5 (half)"),
    "normalize":(lambda p: ["norm"],
                 "Normalize audio to peak level",                           "no parameter needed"),
    "fade":     (lambda p: _parse_fade(p),
                 "Add fade in/out",                                         "in-seconds:out-seconds, e.g. 0.1:0.5"),
    "lowpass":  (lambda p: ["lowpass", str(p or 3000)],
                 "Low-pass filter — roll off highs",                        "cutoff Hz (default 3000)"),
    "highpass": (lambda p: ["highpass", str(p or 300)],
                 "High-pass filter — roll off lows",                        "cutoff Hz (default 300)"),
    "mid":      (lambda p: _parse_mid(p),
                 "Peaking (mid-band) EQ",                                   "freq:gain, e.g. 1000:6"),
    "tremolo":  (lambda p: _parse_tremolo(p),
                 "Amplitude tremolo (volume LFO)",                          "speed:depth, e.g. 5:0.5 (depth 0-1)"),
    "phaser":   (lambda p: _parse_phaser(p),
                 "Phaser — sweeping notches (sci-fi)",                      "speed:decay, e.g. 0.5:0.4"),
    "compressor":(lambda p: _parse_compressor(p),
                 "Downward compressor (tame dynamics)",                     "threshold_dB:ratio, e.g. -20:4"),
    "ringmod":  (lambda p: _parse_ringmod(p),
                 "Ring modulator (Dalek/cyborg timbre)",                    "freq:mix, e.g. 60:0.7 (mix 0-1)"),
    "bitcrush": (lambda p: _parse_bitcrush(p),
                 "Lo-fi crush — bit-depth quantize + sample-rate crush",    "bits:factor, e.g. 6:6 (factor 1 = bit crush only)"),
}


# ── Built-in presets ──────────────────────────────────────────────────────────

BUILTIN_PRESETS = {
    "cave":        ["reverb=80", "echo=0.6:0.6:120:0.3"],
    "chipmunk":    ["pitch=900"],
    "deep":        ["pitch=-400", "bass=6"],
    "telephone":   ["bandpass=300:3400", "overdrive=5", "vol=1.3"],
    "stadium":     ["reverb=90", "echo=0.8:0.7:80:0.25"],
    "megaphone":   ["bandpass=500:4000", "overdrive=30", "vol=1.1"],
    # Curated voice stackups — pro vocal chains + character voices.
    # Order follows the convention: filters/EQ → compression → drive →
    # modulation → reverb last.
    "broadcaster": ["highpass=90", "mid=300:-3", "compressor=-18:3",
                    "mid=3000:3", "treble=3", "bass=2"],
    "podcast":     ["highpass=80", "bass=3", "compressor=-20:2.5",
                    "mid=250:-2", "treble=2"],
    # Signed off on device 2026-07-26 — leave as is; don't retune without asking.
    "trailer":     ["pitch=-250", "bass=5", "compressor=-18:4",
                    "mid=2500:2", "reverb=22"],
    "audiobook":   ["highpass=85", "compressor=-22:3", "mid=2500:2", "reverb=10"],
    "walkie_talkie": ["highpass=400", "lowpass=5000", "overdrive=25",
                      "bitcrush=11:1", "compressor=-32:6", "vol=1.5"],
    "vintage_radio": ["highpass=400", "lowpass=4000", "mid=1000:12",
                      "overdrive=8", "compressor=-26:3", "tremolo=4:0.15",
                      "reverb=8", "vol=1.3"],
    "intercom":    ["bandpass=450:2500", "overdrive=18", "mid=1500:4",
                    "reverb=30", "vol=1.2"],
    "underwater":  ["lowpass=700", "chorus", "pitch=-80", "tremolo=1.5:0.2",
                    "vol=1.35"],
    "ai":          ["pitch=150", "phaser=0.4:0.5", "flanger", "reverb=30"],
    "ethereal":    ["highpass=250", "pitch=120", "reverb=70",
                    "tremolo=3:0.25", "treble=3"],
    # Reverb first: the whole wet signal gets pitched down and dragged to
    # 0.85×, so the tail reads as a huge slow throat rather than a room.
    "dragon":      ["reverb=45", "pitch=-649", "mid=1058:-2",
                    "overdrive=7", "chorus", "tempo=0.85"],
    # Ports of marmalade-tts-android's Android-only stackups (BuiltinEffects
    # E-L), block-for-block in the app's order.
    "cyborg":      ["ringmod=60:0.7", "bandpass=300:3400", "overdrive=6"],
    "eight_bit":   ["lowpass=3446", "bitcrush=7:8"],
    "glitch":      ["bitcrush=8:3", "ringmod=120:0.4", "bandpass=400:3000",
                    "vol=1.35"],
}


# ── Parameter parsers ─────────────────────────────────────────────────────────

def _parse_echo(p) -> list:
    """echo=0.8:0.88:60:0.4  →  ['echo', '0.8', '0.88', '60', '0.4']"""
    if not p:
        return ["echo", "0.8", "0.88", "60", "0.4"]
    parts = str(p).split(":")
    if len(parts) == 4:
        return ["echo"] + parts
    raise ValueError(f"echo expects gain-in:gain-out:delay-ms:decay, got: {p!r}")


def _parse_bandpass(p) -> list:
    """bandpass=300:3400  →  sinc filter low-pass + high-pass"""
    if not p:
        return ["sinc", "300-3400"]
    parts = str(p).split(":")
    if len(parts) == 2:
        return ["sinc", f"{parts[0]}-{parts[1]}"]
    raise ValueError(f"bandpass expects low-hz:high-hz, got: {p!r}")


def _parse_chorus(p) -> list:
    """chorus with sensible defaults."""
    if not p:
        return ["chorus", "0.8", "0.9", "55", "0.4", "0.25", "2", "-s"]
    parts = str(p).split(":")
    if len(parts) == 6:
        return ["chorus"] + parts + ["-s"]
    raise ValueError(f"chorus expects gain-in:gain-out:delay:decay:speed:depth, got: {p!r}")


def _parse_fade(p) -> list:
    """fade=0.1:0.5  →  fade in 0.1s, fade out 0.5s"""
    if not p:
        return ["fade", "0.05", "0", "0.3"]
    parts = str(p).split(":")
    if len(parts) == 2:
        # sox fade format: fade [type] fade-in-length [stop-position] fade-out-length
        return ["fade", parts[0], "0", parts[1]]
    raise ValueError(f"fade expects in-seconds:out-seconds, got: {p!r}")


def _parse_mid(p) -> list:
    """mid=1000:6  →  ['equalizer', '1000', '1.0q', '6']  (peaking EQ, fixed Q=1)"""
    freq, gain = "1000", "0"
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"mid expects freq:gain, got: {p!r}")
        freq, gain = parts
    return ["equalizer", freq, "1.0q", gain]


def _parse_tremolo(p) -> list:
    """tremolo=5:0.5  →  ['tremolo', '5', '50']  (depth 0-1 → sox percent)"""
    speed, depth = "5", "0.4"
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"tremolo expects speed:depth, got: {p!r}")
        speed, depth = parts
    return ["tremolo", speed, str(float(depth) * 100)]


def _parse_phaser(p) -> list:
    """phaser=0.5:0.4  →  ['phaser', '0.7', '0.7', '3.0', '0.4', '0.5', '-s'] (speed:decay)"""
    speed, decay = "0.5", "0.4"
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"phaser expects speed:decay, got: {p!r}")
        speed, decay = parts
    # sox phaser: gain-in gain-out delay decay speed shape
    return ["phaser", "0.7", "0.7", "3.0", decay, speed, "-s"]


def _parse_compressor(p) -> list:
    """compressor=-20:4  →  a sox `compand` with a two-segment downward curve.

    Maps threshold (dBFS) + ratio to a compand transfer function: unity below
    threshold, then `ratio:1` reduction from threshold up to 0 dBFS.
    """
    threshold, ratio = -20.0, 4.0
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"compressor expects threshold_dB:ratio, got: {p!r}")
        threshold, ratio = float(parts[0]), float(parts[1])
    # Output level at 0 dBFS input after compression above the threshold.
    out_at_zero = threshold + (0.0 - threshold) / max(ratio, 1.0)
    # compand attack,decay  soft-knee:in1,out1,in2,out2
    transfer = f"6:-90,-90,{threshold:g},{threshold:g},0,{out_at_zero:g}"
    return ["compand", "0.005,0.1", transfer]


def _parse_ringmod(p) -> list:
    """ringmod=60:0.7  →  ['tremolo', '60', '70']

    Port of the Android app's RingMod block (multiply by a freq-Hz carrier
    sine, blended dry/wet by mix). sox has no true ring modulator; tremolo at
    audio rate is amplitude modulation — same sidebands plus the dry carrier,
    which perceptually matches a mix-blended ring mod. depth = mix * 100.
    """
    freq, mix = "60", "0.7"
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"ringmod expects freq:mix, got: {p!r}")
        freq, mix = parts
    return ["tremolo", freq, str(float(mix) * 100)]


def _parse_bitcrush(p) -> tuple[list, list]:
    """bitcrush=6:6  →  (args closing this sox stage, args opening the next)

    Port of the Android app's BitcrushProcessor: quantize to `bits` bit depth,
    then sample-and-hold every `factor` samples.

    sox has no quantizer effect — but writing a WAV *is* one, since samples
    land on 16-bit steps. So quantizing to `bits` is: scale down by
    2^(bits-15), let the intermediate file round, scale back up. That gives
    exactly the app's round(x · 2^bits) / 2^bits, and it's why this returns a
    pair — the file boundary splits the chain (see build_sox_stages).

    The sample-rate crush then runs in the next stage. sox's `upsample`
    zero-stuffs rather than sample-and-holds, so its images are brighter than
    the app's hold — same family of artifact — and the limited gain stage
    (`gain -l`) makes up the 1/factor level the zeros cost without
    hard-clipping the peaks they preserve.
    """
    bits, factor = 6.0, 6
    if p:
        parts = str(p).split(":")
        if len(parts) != 2:
            raise ValueError(f"bitcrush expects bits:factor, got: {p!r}")
        bits, factor = float(parts[0]), int(parts[1])
    crush = []
    if factor > 1:
        makeup = 20 * math.log10(factor)
        crush = ["downsample", str(factor), "upsample", str(factor),
                 "gain", "-l", f"{makeup:.2f}"]
    if bits >= 16:
        # The output file is 16-bit anyway — nothing to quantize, no boundary.
        return [], crush
    step = 2.0 ** (bits - 15)
    return ["vol", f"{step:.10g}"], ["vol", f"{1 / step:.10g}"] + crush


# ── Public API ────────────────────────────────────────────────────────────────

def sox_available() -> bool:
    """Check if sox is installed."""
    return shutil.which("sox") is not None


def resolve_effect_list(effect_specs: list[str], config: dict) -> list[str]:
    """Resolve a list of effect specs, expanding preset names.

    A spec is either:
      - A preset name: "cave"
      - An effect=value: "reverb=50"
      - An effect name (no value): "flanger"

    Returns a flat list of effect specs (no presets, all resolved).
    """
    # Merge builtin presets with user-defined presets from config
    user_presets = config.get("effects", {}).get("presets", {})
    all_presets = {**BUILTIN_PRESETS, **user_presets}

    resolved = []
    for spec in effect_specs:
        if spec in all_presets:
            # Expand preset — presets can contain other presets (one level deep)
            for sub_spec in all_presets[spec]:
                if sub_spec in all_presets:
                    resolved.extend(all_presets[sub_spec])
                else:
                    resolved.append(sub_spec)
        else:
            resolved.append(spec)
    return resolved


def _parse_spec(spec: str) -> tuple[str, object]:
    """Parse 'reverb=50' → ('reverb', '50'), 'flanger' → ('flanger', None)."""
    if "=" in spec:
        name, _, value = spec.partition("=")
        return name.strip(), value.strip()
    return spec.strip(), None


def build_sox_stages(effect_specs: list[str]) -> list[list[str]]:
    """Build the sox effect chain from a list of resolved specs.

    Returns one list of sox args per sox invocation, e.g.:
      [['reverb', '50', 'pitch', '200', 'norm']]

    Almost always a single stage. `bitcrush` is the exception: its bit-depth
    quantization needs a 16-bit file to round through, so it splits the chain
    at that point and the caller runs the stages back to back through temp
    files (see _parse_bitcrush and apply_effects).
    """
    stages = []
    current = []
    for spec in effect_specs:
        name, value = _parse_spec(spec)
        if name not in EFFECTS:
            raise ValueError(f"Unknown effect: {name!r}. Run --list-effects to see available effects.")
        builder, _desc, _hint = EFFECTS[name]
        if name == "bitcrush":
            # The only effect that returns a stage boundary rather than args.
            closing, opening = builder(value)
            if closing:
                stages.append(current + closing)
                current = list(opening)
            else:
                current.extend(opening)
        else:
            current.extend(builder(value))
    stages.append(current)
    return stages


def apply_effects(in_path: str, out_path: str, effect_specs: list[str], config: dict = None):
    """Apply audio effects to a WAV file using sox.

    Args:
        in_path:      Input WAV file.
        out_path:     Output WAV file (can be the same as in_path — uses temp file).
        effect_specs: List of effect specs, e.g. ["reverb=50", "pitch=200", "cave"].
        config:       Full config dict (for user-defined presets).

    Raises:
        RuntimeError: If sox is not installed or the sox command fails.
    """
    if config is None:
        config = {}

    resolved = resolve_effect_list(effect_specs, config)
    if not resolved:
        return

    if not sox_available():
        raise RuntimeError(
            "sox is required for audio effects but was not found.\n"
            "Install it: apt install sox   or   brew install sox"
        )

    stages = build_sox_stages(resolved)

    # One sox invocation per stage, threaded through temp files. A chain
    # without bitcrush is a single stage straight from in_path to out_path;
    # in_path == out_path also needs a temp file to land in.
    same_file = os.path.realpath(in_path) == os.path.realpath(out_path)
    temps = []
    try:
        src = in_path
        for i, stage in enumerate(stages):
            final = i == len(stages) - 1
            if final and not same_file:
                dst = out_path
            else:
                fd, dst = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                temps.append(dst)
            # -D: never auto-dither. Intermediates are forced to 16-bit
            # because bitcrush's quantization is done BY that rounding.
            cmd = ["sox", "-D", src] + ([] if final else ["-b", "16"]) + [dst] + stage
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                err = proc.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"sox failed:\n{err}")
            src = dst
        if same_file:
            # shutil.move (not os.replace) — handles cross-filesystem moves
            # (e.g. tmpfs /tmp → ext4 home dir, which os.replace can't do).
            temps.remove(src)
            shutil.move(src, out_path)
            # tempfile.mkstemp creates with 0600; restore the user's default umask
            # so the final output is readable like any other file they create.
            try:
                umask = os.umask(0)
                os.umask(umask)
                os.chmod(out_path, 0o666 & ~umask)
            except OSError:
                pass
    finally:
        for path in temps:
            try:
                os.unlink(path)
            except OSError:
                pass


def list_effects(user_presets: dict = None):
    """Print all available effects and presets."""
    print("Available effects:")
    for name, (_, desc, hint) in EFFECTS.items():
        print(f"  {name:<12} {desc}")
        if hint:
            print(f"               param: {hint}")

    print()
    print("Built-in presets:")
    for name, specs in BUILTIN_PRESETS.items():
        print(f"  {name:<12} {' + '.join(specs)}")

    if user_presets:
        print()
        print("User presets (from config):")
        for name, specs in user_presets.items():
            print(f"  {name:<12} {' + '.join(specs)}")

    print()
    print("Usage:")
    print("  marmalade-tts \"Hello\" --effect reverb=50")
    print("  marmalade-tts \"Hello\" --effect pitch=200 --effect reverb=30")
    print("  marmalade-tts \"Hello\" --effect cave         # named preset")
