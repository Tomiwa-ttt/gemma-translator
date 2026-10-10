# Gemma Edge Translator: English to French

An offline voice translator that turns spoken English into spoken French. Every model runs on the device, with no internet connection, API keys or cloud costs after setup.

This is the software prototype for **The Arc**, a pocket translator built around a Raspberry Pi 5. It currently runs on a Mac while the hardware is being put together.

Built on [google-gemma/gemma-translator](https://github.com/google-gemma/gemma-translator) by Google Creative Lab (Apache 2.0).

## How it works

```
Microphone -> Moonshine STT -> English text -> Gemma 4 E2B -> French text -> Piper TTS (fr_FR-siwis) -> Speaker
```

| Stage | Model | Runtime |
|---|---|---|
| Speech to text | Moonshine (English) | moonshine-voice |
| Translation | Gemma 4 E2B (instruction tuned) | LiteRT-LM, CPU |
| Text to speech | Piper `fr_FR-siwis-medium` | moonshine-voice |

## What I changed

- **French support.** Added French to the language selector and mapped it to the `fr-fr` voice in the backend.
- **Lower mic latency.** The original opened the microphone on every key press, which added 1 to 5 seconds before audio was captured. The mic now opens once and stays ready, and a 0.5 second pre-roll buffer keeps the first word from being clipped.
- **Audio playback fix on macOS.** With the mic held open, Chrome's echo cancellation put macOS into voice-call mode and silenced the translated speech. It is now disabled, with noise suppression and auto gain kept on.
- **Long-text layout.** Long sentences no longer overlap between the input and output panels. Each panel stays in its own half and the font size scales down with text length.
- **Desktop cursor.** The kiosk build hid the mouse cursor on narrow windows. It now stays visible.
- **Pipeline benchmark.** Added `compare_pipelines.py` to test whether Gemma's built-in audio input can replace the separate speech-to-text model.

## Experiment: does Gemma need a separate speech model?

Gemma 4 E2B accepts audio directly, so one model could in principle go from English speech straight to French text. I tested this against the current two-model pipeline on five recorded English phrases, ranging from a short question to a longer sentence with a number and a day of the week. Both pipelines used the same recordings, the same Gemma model and the same CPU backend.

- **A:** Moonshine STT, then Gemma translates the text
- **B:** Gemma receives the audio and outputs French directly

| | A: Moonshine + Gemma | B: Gemma only |
|---|---|---|
| Median time | 2.15s | 8.20s |
| Time range | 2.1s to 3.0s | 1.9s to 9.4s |
| Fully correct translations | 2 of 5 | 2 of 5 |
| Shows the English it heard | Yes | No |
| Invented content | No | Yes, on 1 of 5 |

Measured on an Apple Silicon laptop, CPU only, model loading excluded.

**Findings**

- **Accuracy was even, but the errors differed.** A's mistakes came from speech recognition, such as hearing "bill" as "video", after which Gemma translated the wrong word correctly. B recognised some words better but invented a phrase when a clip had silence before the speech.
- **A was faster and more consistent.** A stayed between 2 and 3 seconds. B was quick on short clips but slowed sharply on clips longer than about 4.5 seconds.
- **A is easier to verify.** It shows the English it heard, so a user can catch a recognition error. B gives no way to check.
- **Proper names were misheard by both.**

**Decision:** keep the two-model pipeline. On a Raspberry Pi 5, which is several times slower than a laptop, B's delays would be too long for conversation. Accuracy work will focus on a stronger speech-to-text model.

Run it yourself with `python compare_pipelines.py --record`. Results are saved locally to `comparison_results.md` and `comparison_results.csv`.

## Run it on macOS

Requirements: Homebrew, Python 3.12, Node 18+, about 6 GB of free disk space.

```bash
brew install python@3.12 node bash
git clone https://github.com/Tomiwa-ttt/gemma-translator.git
cd gemma-translator

python3.12 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

./download_model.sh
"$(brew --prefix)/bin/bash" ./start.sh
```

Open http://localhost:5173 in Chrome and allow microphone access.

- Use the arrow keys to set Lane 1 to **English**, press Space, set Lane 2 to **French**, then press Space again.
- Hold **Z** while speaking and release to translate.

**macOS notes**

- The upstream `setup.sh` fails because it requires package hashes that `requirements.txt` does not contain. The `pip install` line above replaces it.
- `start.sh` needs Bash 4.3 or newer, and macOS ships Bash 3.2. That is why it runs through Homebrew's Bash.
- The first French translation pauses while the French voice downloads once. Later ones are immediate.
- If `download_model.sh` stalls, download the model file directly and import it with `litert-lm import gemma-4-E2B-it.litertlm gemma4-e2b`.

## Known limitations

- **One-way only.** Moonshine has no French model, so the French lane can speak but not listen.
- **Proper names** are often misheard.
- **Speaker feedback.** On laptop speakers, pressing Z while the translation is still playing lets the mic pick up the French audio.

## Credits and license

Based on [Gemma Translator](https://github.com/google-gemma/gemma-translator) by Alan Yam, Shashwath Santosh and Dan Motzenbecker at Google Creative Lab. Uses [Gemma 4](https://ai.google.dev/gemma), [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Moonshine](https://github.com/moonshine-ai/moonshine).

Licensed under Apache 2.0. See [LICENSE](LICENSE).

Modifications by Tomiwa Ajibola ([@Tomiwa-ttt](https://github.com/Tomiwa-ttt)).
