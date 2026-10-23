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
WINDOW_SAMPLES = 16000       # Analyze 1 second of audio
HOP_SAMPLES = 8000           # Advance 0.5 seconds per check
COOLDOWN_SECONDS = 5

# Starting values, not calibrated probabilities.
# Breaking glass is brief, so it requires only one positive window.
EVENT_RULES = {
    "scream": {"threshold": 0.35, "required_hits": 2},
    "siren": {"threshold": 0.35, "required_hits": 2},
    "alarm": {"threshold": 0.35, "required_hits": 2},
    "breaking_glass": {"threshold": 0.50, "required_hits": 1},
}

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model_and_classes():
    print("Loading audio model...", flush=True)
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as file:
        class_names = [
            row["display_name"] for row in csv.DictReader(file)
        ]

    # Resolve against the class names in the installed YAMNet model.
    # Do not treat generic speech or generic glass sounds as alerts.
    indices = {
        "scream": [
            i for i, name in enumerate(class_names)
            if name.lower() == "screaming"
        ],
        "siren": [
            i for i, name in enumerate(class_names)
            if "siren" in name.lower()
        ],
        "alarm": [
            i for i, name in enumerate(class_names)
            if "alarm" in name.lower()
            and "clock" not in name.lower()
        ],
        "breaking_glass": [
            i for i, name in enumerate(class_names)
            if name.strip().lower() == "shatter"
        ],

    }

    missing = [event for event, matches in indices.items() if not matches]
    if missing:
        raise RuntimeError(
            "Could not find YAMNet classes for: "
            + ", ".join(missing)
            + ". Check the installed model's class map."
        )

    print("Classes used for alerts:")
    for event, matches in indices.items():
        labels = ", ".join(class_names[i] for i in matches)
        print(f"  {event}: {labels}")

    return model, indices


def score_events(model, indices, waveform):
    waveform = np.asarray(waveform, dtype=np.float32)

    if len(waveform) < WINDOW_SAMPLES:
        waveform = np.pad(
            waveform,
            (0, WINDOW_SAMPLES - len(waveform)),
        )

    scores, _, _ = model(waveform)
    scores = scores.numpy()

    return {
        event: float(np.max(scores[:, class_indices]))
        for event, class_indices in indices.items()
    }


def write_header_if_needed(path, header):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(header)


def choose_microphone():
    print("\nAvailable microphone inputs:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(f"  {index}: {device['name']}")

    choice = input(
        "\nMicrophone number (Enter for Windows default): "
    ).strip()

    device_id = int(choice) if choice else None
    device_info = sd.query_devices(device_id, "input")
    print(f"Using: {device_info['name']}", flush=True)
    return device_id


def run_microphone(model, indices):
    device_id = choose_microphone()
    log_path = SCRIPT_DIR / "emergency_alerts_live.csv"

    write_header_if_needed(
        log_path,
        ["timestamp_utc", "event_type", "score"],
    )

    print(f"Alert log: {log_path}")
    print("Listening. Press Ctrl+C to stop.\n", flush=True)

    chunks = deque(maxlen=2)
    hits = {event: 0 for event in EVENT_RULES}
    last_alert = {
        event: float("-inf") for event in EVENT_RULES
    }

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
                mic_level = float(
                    np.sqrt(np.mean(waveform ** 2))
                )
                event_scores = score_events(
                    model, indices, waveform
                )

                score_text = " | ".join(
                    f"{event}: {score:.3f}"
                    for event, score in event_scores.items()
                )
                print(
                    f"Mic level: {mic_level:.6f} | {score_text}",
                    flush=True,
                )

                now = time.monotonic()

                for event, rule in EVENT_RULES.items():
                    score = event_scores[event]

                    if score >= rule["threshold"]:
                        hits[event] += 1
                    else:
                        hits[event] = 0

                    if (
                        hits[event] >= rule["required_hits"]
                        and now - last_alert[event]
                        >= COOLDOWN_SECONDS
                    ):
                        timestamp = datetime.now(
                            timezone.utc
                        ).isoformat(timespec="seconds")

                        print(
                            f"ALERT: {event} — {timestamp} — "
                            f"score {score:.3f}",
                            flush=True,
                        )

                        with log_path.open(
                            "a", newline="", encoding="utf-8"
                        ) as file:
                            csv.writer(file).writerow(
                                [timestamp, event, f"{score:.3f}"]
                            )

                        last_alert[event] = now
                        hits[event] = 0

    except KeyboardInterrupt:
        print("\nStopped listening.")


def decode_recording(path):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg was not found. Run 'ffmpeg -version' "
            "in a fresh PowerShell window."
        ) from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(
            f"FFmpeg could not decode the file: {details}"
        ) from error

    audio = np.frombuffer(
        result.stdout, dtype="<f4"
    ).copy()

    if len(audio) == 0:
        raise RuntimeError(
            "The file contains no decodable audio."
        )

    return audio


def recording_window_starts(sample_count):
    if sample_count <= WINDOW_SAMPLES:
        return [0]

    starts = list(
        range(
            0,
            sample_count - WINDOW_SAMPLES + 1,
            HOP_SAMPLES,
        )
    )

    # Ensure the end of the file is analyzed.
    final_start = sample_count - WINDOW_SAMPLES
    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def run_recording(model, indices):
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

    events = []
    highest_scores = {
        event: 0.0 for event in EVENT_RULES
    }
    state = {
        event: {
            "hits": 0,
            "start": None,
            "end": None,
            "max_score": 0.0,
        }
        for event in EVENT_RULES
    }

    for start in recording_window_starts(len(audio)):
        window = audio[start:start + WINDOW_SAMPLES]
        event_scores = score_events(model, indices, window)

        start_seconds = start / SAMPLE_RATE
        end_seconds = min(
            (start + WINDOW_SAMPLES) / SAMPLE_RATE,
            duration,
        )

        for event, rule in EVENT_RULES.items():
            score = event_scores[event]
            highest_scores[event] = max(
                highest_scores[event], score
            )
            current = state[event]

            if score >= rule["threshold"]:
                if current["hits"] == 0:
                    current["start"] = start_seconds
                    current["max_score"] = score

                current["hits"] += 1
                current["end"] = end_seconds
                current["max_score"] = max(
                    current["max_score"], score
                )
            else:
                if (
                    current["hits"]
                    >= rule["required_hits"]
                ):
                    events.append((
                        current["start"],
                        current["end"],
                        event,
                        current["max_score"],
                    ))

                state[event] = {
                    "hits": 0,
                    "start": None,
                    "end": None,
                    "max_score": 0.0,
                }

    # Include events that continue to the end of the file.
    for event, rule in EVENT_RULES.items():
        current = state[event]
        if current["hits"] >= rule["required_hits"]:
            events.append((
                current["start"],
                current["end"],
                event,
                current["max_score"],
            ))

    events.sort(key=lambda item: (item[0], item[2]))

    log_path = recording.with_name(
        f"{recording.stem}_emergency_alerts.csv"
    )

    with log_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "start_seconds",
            "end_seconds",
            "event_type",
            "max_score",
        ])

        for start, end, event, score in events:
            writer.writerow([
                f"{start:.2f}",
                f"{end:.2f}",
                event,
                f"{score:.3f}",
            ])

    print(f"\nAnalyzed {duration:.1f} seconds.")

    for start, end, event, score in events:
        print(
            f"ALERT: {event} at {start:.2f}s–"
            f"{end:.2f}s (score {score:.3f})"
        )

    print(f"Detected intervals: {len(events)}")
    print("Highest scores:")
    for event, score in highest_scores.items():
        print(f"  {event}: {score:.3f}")

    print(f"CSV saved: {log_path}")


def main():
    print("Emergency sound detector")
    print("  1. Live microphone")
    print("  2. WAV / MP3 / M4A recording")
    choice = input("Choose 1 or 2: ").strip()

    if choice not in ("1", "2"):
        print("Please restart and choose 1 or 2.")
        return

    model, indices = load_model_and_classes()

    if choice == "1":
        run_microphone(model, indices)
    else:
        run_recording(model, indices)


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        ValueError,
        sd.PortAudioError,
    ) as error:
        print(f"Error: {error}")
