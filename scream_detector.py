import csv
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow_hub as hub

SAMPLE_RATE = 16000
HOP_SAMPLES = 8000              # Check every 0.5 seconds
WINDOW_SAMPLES = 16000          # Analyze the latest 1 second
THRESHOLD = 0.35
REQUIRED_HITS = 2
COOLDOWN_SECONDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model():
    print("Loading audio model...", flush=True)
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as file:
        classes = [row["display_name"] for row in csv.DictReader(file)]

    return model, classes.index("Screaming")


def score_window(model, scream_index, waveform):
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, WINDOW_SAMPLES - len(waveform)))

    scores, _, _ = model(waveform)
    return float(np.max(scores.numpy()[:, scream_index]))


def run_microphone(model, scream_index):
    print("\nAvailable microphone inputs:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(f"  {index}: {device['name']}")

    selection = input(
        "\nMicrophone number (Enter for Windows default): "
    ).strip()
    device_id = int(selection) if selection else None
    device_info = sd.query_devices(device_id, "input")
    print(f"Using: {device_info['name']}")

    log_path = SCRIPT_DIR / "scream_alerts_live.csv"
    new_file = not log_path.exists() or log_path.stat().st_size == 0

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["timestamp_utc", "screaming_score"])

    print(f"Alert log: {log_path}")
    print("Listening. Press Ctrl+C to stop.\n")

    chunks = deque(maxlen=2)
    hits = 0
    last_alert = float("-inf")

    try:
        with sd.InputStream(
            device=device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SAMPLES,
        ) as stream:
            while True:
                audio, overflowed = stream.read(HOP_SAMPLES)
                if overflowed:
                    print("Warning: microphone input overflowed.")

                chunks.append(audio[:, 0].copy())
                if len(chunks) < 2:
                    continue

                waveform = np.concatenate(chunks)
                mic_level = float(np.sqrt(np.mean(waveform ** 2)))
                score = score_window(model, scream_index, waveform)

                print(
                    f"Mic level: {mic_level:.6f} | "
                    f"Screaming score: {score:.3f}",
                    flush=True,
                )

                hits = hits + 1 if score >= THRESHOLD else 0
                now = time.monotonic()

                if (
                    hits >= REQUIRED_HITS
                    and now - last_alert >= COOLDOWN_SECONDS
                ):
                    timestamp = datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    print(
                        f"SCREAM DETECTED — {timestamp} — "
                        f"score {score:.3f}",
                        flush=True,
                    )

                    with log_path.open(
                        "a", newline="", encoding="utf-8"
                    ) as file:
                        csv.writer(file).writerow(
                            [timestamp, f"{score:.3f}"]
                        )

                    last_alert = now
                    hits = 0

    except KeyboardInterrupt:
        print("\nStopped listening.")


def decode_recording(path):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "pipe:1",
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, check=True
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg was not found. Install it and reopen PowerShell."
        ) from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(
            f"FFmpeg could not decode the file: {details}"
        ) from error

    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(audio) == 0:
        raise RuntimeError("The file contains no decodable audio.")
    return audio


def run_recording(model, scream_index):
    entered_path = input(
        "\nPath to WAV, MP3, or M4A recording: "
    ).strip().strip('"').strip("'")
    recording = Path(entered_path).expanduser().resolve()

    if not recording.is_file():
        print(f"File not found: {recording}")
        return

    print(f"Decoding: {recording}", flush=True)
    audio = decode_recording(recording)
    duration = len(audio) / SAMPLE_RATE

    if len(audio) <= WINDOW_SAMPLES:
        starts = [0]
    else:
        starts = list(
            range(
                0,
                len(audio) - WINDOW_SAMPLES + 1,
                HOP_SAMPLES,
            )
        )
        final_start = len(audio) - WINDOW_SAMPLES
        if starts[-1] != final_start:
            starts.append(final_start)

    events = []
    hits = 0
    event_start = None
    event_end = None
    event_max_score = 0.0
    highest_score = 0.0

    for start in starts:
        window = audio[start:start + WINDOW_SAMPLES]
        score = score_window(model, scream_index, window)
        highest_score = max(highest_score, score)

        start_seconds = start / SAMPLE_RATE
        end_seconds = min(
            (start + WINDOW_SAMPLES) / SAMPLE_RATE, duration
        )

        if score >= THRESHOLD:
            if hits == 0:
                event_start = start_seconds
                event_max_score = score
            hits += 1
            event_end = end_seconds
            event_max_score = max(event_max_score, score)
        else:
            if hits >= REQUIRED_HITS:
                events.append(
                    (event_start, event_end, event_max_score)
                )
            hits = 0
            event_start = None
            event_max_score = 0.0

    if hits >= REQUIRED_HITS:
        events.append((event_start, event_end, event_max_score))

    log_path = recording.with_name(
        f"{recording.stem}_scream_alerts.csv"
    )
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "start_seconds",
            "end_seconds",
            "max_screaming_score",
        ])
        for start, end, score in events:
            writer.writerow([
                f"{start:.2f}",
                f"{end:.2f}",
                f"{score:.3f}",
            ])

    print(f"\nAnalyzed {duration:.1f} seconds.")
    for start, end, score in events:
        print(
            f"SCREAM DETECTED: {start:.2f}s–{end:.2f}s "
            f"(score {score:.3f})"
        )

    print(f"Detected intervals: {len(events)}")
    print(f"Highest screaming score: {highest_score:.3f}")
    print(f"CSV saved: {log_path}")


def main():
    print("Human scream detector")
    print("  1. Live microphone")
    print("  2. WAV / MP3 / M4A recording")
    choice = input("Choose 1 or 2: ").strip()

    if choice not in ("1", "2"):
        print("Please restart and choose 1 or 2.")
        return

    model, scream_index = load_model()

    if choice == "1":
        run_microphone(model, scream_index)
    else:
        run_recording(model, scream_index)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, sd.PortAudioError) as error:
        print(f"Error: {error}")
