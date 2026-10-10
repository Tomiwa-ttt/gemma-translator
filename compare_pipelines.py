"""
Phase 2 comparison: two ways to turn spoken English into French text.

  A (current app):  Moonshine STT  ->  English text  ->  Gemma 4 E2B  ->  French text
  B (Gemma only):   audio  ->  Gemma 4 E2B  ->  French text   (one model)

Both use the same recordings, the same Gemma model and the same CPU backend,
so the timings and outputs are directly comparable.

Usage (from ~/gemma-translator, with the app stopped):
    source venv/bin/activate
    python compare_pipelines.py --record      # record the test phrases, then compare
    python compare_pipelines.py --record --device 2   # use a specific mic number
    python compare_pipelines.py               # re-run on the saved recordings

Results are printed and saved to comparison_results.md and comparison_results.csv.
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
AUDIO_DIR = Path("test_audio")
MODEL_PATH = Path.home() / ".litert-lm" / "models" / "gemma4-e2b" / "model.litertlm"

# Short to long, plus an idiom and a number, so both pipelines get a fair mix.
PHRASES = [
    "Where is the train station?",
    "My name is Alex. Nice to meet you.",
    "Can I have the bill, please?",
    "I'm running late, so I'll catch the next bus.",
    "How much does a ticket to Montreal cost for two adults this Saturday?",
]

TEXT_PROMPT = (
    "Translate the following English text into natural French. "
    "Output ONLY the French translation, with no quotes, notes or explanations.\n\n{text}"
)
AUDIO_PROMPT = (
    "Translate the English speech in this audio into natural French. "
    "Output ONLY the French translation, with no quotes, notes or explanations."
)


# ---------------------------------------------------------------- recording
SILENT_PEAK = 0.003  # average (RMS) loudness below this is effectively silence


def peak_level(audio):
    """Average loudness (RMS). Unlike the peak, a single click can't fake speech."""
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def record_phrases(device=None):
    import sounddevice as sd

    AUDIO_DIR.mkdir(exist_ok=True)
    mic = sd.query_devices(device, kind="input")["name"]
    print(f"\nUsing microphone: {mic}")
    print("Recording test phrases. For each one: press Enter, speak, press Enter.\n")
    for i, phrase in enumerate(PHRASES, 1):
        input(f'[{i}/{len(PHRASES)}] Say: "{phrase}"\n   Press Enter to START...')
        chunks = []

        def cb(indata, frames, t, status):
            chunks.append(indata.copy())

        # Record at the mic's native rate, then resample to 16 kHz for both models.
        rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
        with sd.InputStream(
            samplerate=rate, channels=1, dtype="float32", callback=cb, device=device
        ):
            input("   Recording... press Enter to STOP.")
        audio = np.concatenate(chunks)[:, 0] if chunks else np.zeros(rate, np.float32)
        if rate != SAMPLE_RATE:
            n = int(len(audio) * SAMPLE_RATE / rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
            audio = audio.astype(np.float32)
        path = AUDIO_DIR / f"phrase_{i:02d}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        peak = peak_level(audio)
        print(f"   Saved {path} ({len(audio) / SAMPLE_RATE:.1f}s, loudness {peak:.4f})")
        if peak < SILENT_PEAK:
            sys.exit(
                "\n   That recording is SILENT. macOS is probably blocking Terminal from the mic.\n"
                "   Fix: System Settings > Privacy & Security > Microphone > turn on Terminal,\n"
                "   then quit Terminal fully (Cmd+Q), reopen it, and run this again."
            )
        print()


# ---------------------------------------------------------------- models
def reply_text(resp):
    """Pull the text out of a LiteRT-LM reply, whatever shape it comes in."""
    content = resp.get("content") if isinstance(resp, dict) else resp
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        text = "".join(parts).strip()
        if text:
            return text
    print(f"   [debug] unexpected reply format: {resp!r}")
    return ""


def load_gemma():
    import litert_lm
    from litert_lm import Backend, Engine

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}. Run ./download_model.sh first.")
    t = time.perf_counter()
    engine = Engine(str(MODEL_PATH), backend=Backend.CPU(), audio_backend=Backend.CPU())
    print(f"Gemma 4 E2B loaded in {time.perf_counter() - t:.1f}s")
    return engine


def load_moonshine():
    from moonshine_voice import Transcriber, get_model_for_language

    t = time.perf_counter()
    model_path, model_arch = get_model_for_language("en")
    stt = Transcriber(model_path=model_path, model_arch=model_arch)
    print(f"Moonshine STT loaded in {time.perf_counter() - t:.1f}s")
    return stt


def ask_gemma(engine, contents):
    from litert_lm import SamplerConfig

    conv = engine.create_conversation(sampler_config=SamplerConfig(temperature=0.0))
    try:
        return reply_text(conv.send_message(contents))
    finally:
        conv.close()


# ---------------------------------------------------------------- pipelines
def pipeline_a(stt, engine, wav_path):
    """Moonshine STT, then Gemma text translation."""
    from litert_lm import Contents

    audio, sr = sf.read(wav_path, dtype="float32")
    t0 = time.perf_counter()
    transcript = stt.transcribe_without_streaming(audio, sr)
    english = " ".join(line.text for line in transcript.lines).strip()
    t1 = time.perf_counter()
    french = ask_gemma(engine, Contents.of(TEXT_PROMPT.format(text=english))) if english else ""
    t2 = time.perf_counter()
    return english, french, t1 - t0, t2 - t1


def pipeline_b(engine, wav_path):
    """Audio straight into Gemma, French text out."""
    from litert_lm import Content, Contents

    t0 = time.perf_counter()
    french = ask_gemma(
        engine,
        Contents.of(Content.AudioFile(str(Path(wav_path).resolve())), Content.Text(AUDIO_PROMPT)),
    )
    return french, time.perf_counter() - t0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="record the test phrases first")
    ap.add_argument("--device", type=int, default=None, help="input device number to record from")
    args = ap.parse_args()

    if args.record:
        record_phrases(args.device)

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        sys.exit("No recordings found. Run with --record first.")

    silent = [w.name for w in wavs if peak_level(sf.read(w, dtype="float32")[0]) < SILENT_PEAK]
    if silent:
        sys.exit(
            f"These recordings are silent: {', '.join(silent)}\n"
            "Allow Terminal in System Settings > Privacy & Security > Microphone, "
            "restart Terminal, then run with --record again."
        )

    engine = load_gemma()
    stt = load_moonshine()

    # Warm-up: the first call to each path includes one-off setup cost.
    print("Warming up both pipelines (not timed)...")
    pipeline_a(stt, engine, wavs[0])
    pipeline_b(engine, wavs[0])

    rows = []
    for i, wav in enumerate(wavs):
        dur = sf.info(wav).duration
        said = PHRASES[i] if i < len(PHRASES) else "(your own recording)"
        print(f"\n[{wav.name}] {dur:.1f}s  said: {said}")
        eng, fr_a, t_stt, t_mt = pipeline_a(stt, engine, wav)
        print(f"  A  heard : {eng}")
        print(f"  A  French: {fr_a}   ({t_stt:.2f}s STT + {t_mt:.2f}s Gemma = {t_stt + t_mt:.2f}s)")
        fr_b, t_b = pipeline_b(engine, wav)
        print(f"  B  French: {fr_b}   ({t_b:.2f}s)")
        rows.append(
            dict(file=wav.name, audio_s=round(dur, 2), said=said, a_heard=eng, a_french=fr_a,
                 a_stt_s=round(t_stt, 2), a_gemma_s=round(t_mt, 2),
                 a_total_s=round(t_stt + t_mt, 2), b_french=fr_b, b_total_s=round(t_b, 2))
        )

    a_med = statistics.median(r["a_total_s"] for r in rows)
    b_med = statistics.median(r["b_total_s"] for r in rows)
    faster = "B (Gemma only)" if b_med < a_med else "A (Moonshine + Gemma)"

    with open("comparison_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Pipeline comparison: Moonshine + Gemma (A) vs Gemma only (B)",
        "",
        f"Machine: {os.uname().machine}, CPU backend. Times exclude model loading and playback.",
        "",
        "| Phrase | Audio | A: heard | A: French | A time | B: French | B time |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['said']} | {r['audio_s']}s | {r['a_heard']} | {r['a_french']} | "
            f"{r['a_total_s']}s | {r['b_french']} | {r['b_total_s']}s |"
        )
    lines += [
        "",
        f"**Median time:** A = {a_med:.2f}s, B = {b_med:.2f}s. Faster: {faster}.",
        "",
        "Judge accuracy by comparing each French line against what was said.",
    ]
    Path("comparison_results.md").write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 60)
    print(f"Median time  A (Moonshine + Gemma): {a_med:.2f}s")
    print(f"Median time  B (Gemma only):        {b_med:.2f}s")
    print(f"Faster: {faster}")
    print("Saved comparison_results.md and comparison_results.csv")


if __name__ == "__main__":
    main()
