# Human Scream Detector

Detect possible human screams from either a live microphone or a WAV, MP3, or M4A recording. The program prints alerts and saves results to CSV files.

This is a prototype, not a safety-critical alarm. Test it with screams and similar sounds, such as cheering, before relying on its results.

## Files

Place these files in the same folder:

- `scream_detector.py` — the detector script
- `requirements.txt` — Python dependencies
- `README.md` — these instructions

## 1. Install Python 3.11

Install Python 3.11 if it is not already installed. In PowerShell, verify:

```powershell
py -3.11 --version
```

Use Python 3.11 for this project, rather than the Python 3.14 environment that previously produced incompatible TensorFlow packages.

## 2. Install FFmpeg

FFmpeg is needed for recording mode (WAV, MP3, and M4A). It is not needed for live microphone mode.

In PowerShell:

```powershell
winget install --id Gyan.FFmpeg -e
```

If it is already installed, you can skip that command. After installing, close and reopen PowerShell—or restart your IDE if using its terminal—then verify:

```powershell
ffmpeg -version
```

The command should print FFmpeg version information.

## 3. Create the Python environment

Open PowerShell and move to the project folder:

```powershell
Set-Location "D:\ml\new attempt"
```

Create a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv ".venv-scream"
```

If `.venv-scream` already exists, keep it and skip the creation command.

Install the dependencies into that environment:

```powershell
& ".\.venv-scream\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-scream\Scripts\python.exe" -m pip install -r ".\requirements.txt"
& ".\.venv-scream\Scripts\python.exe" -m pip check
```

Use `.venv-scream` when running the detector. Do not use the older `.venv` environment.

## 4. Run the detector

From `D:\ml\new attempt`:

```powershell
& ".\.venv-scream\Scripts\python.exe" ".\scream_detector.py"
```

The program displays:

```text
1. Live microphone
2. WAV / MP3 / M4A recording
Choose 1 or 2:
```

The audio model may download the first time you run the program, so the first startup can take longer. TensorFlow may also print informational warnings.

### Option 1: Live microphone

Enter `1`, then choose the number of a microphone **input** from the displayed device list. Press Enter instead to use the Windows default input.

On the system used to develop this script, the realme headset's Windows WASAPI input appeared as device `18`. **Check the list each time**: device numbers can change after restarting or reconnecting hardware.

The terminal prints microphone levels and screaming scores. When the alert rule is met, it prints `SCREAM DETECTED` and appends a row to:

```text
D:\ml\new attempt\scream_alerts_live.csv
```

Live-mode timestamps are in **UTC**. Press `Ctrl+C` to stop.

If `Mic level` stays near zero while you speak, check the selected input and Windows microphone settings. Changing the scream threshold will not fix missing microphone audio.

### Option 2: Recording

Enter `2`, then paste the full path to a WAV, MP3, or M4A file, for example:

```text
D:\ml\fydp\new-event detector\NIGENS\maleScream\31818__malexmedia__malexmedia-man-scream.wav
```

The program prints detected start and end times and saves a CSV **beside the recording**. For the example above, the output would be:

```text
D:\ml\fydp\new-event detector\NIGENS\maleScream\31818__malexmedia__malexmedia-man-scream_scream_alerts.csv
```

`start_seconds` and `end_seconds` are positions **within the recording**, not clock times. A CSV with just its header means that no interval passed the current detection rule.

## Scores and detection settings

The starting settings in `scream_detector.py` are:

```python
THRESHOLD = 0.35
REQUIRED_HITS = 2
```

An alert requires two consecutive analyzed windows to score at least `0.35` for YAMNet's **Screaming** class. The CSV's `max_screaming_score` is the highest score in the detected interval. For example, `0.758` is a model score—not **75.8% detector accuracy**.

If a known scream is missed, inspect the printed `Highest screaming score` before adjusting the threshold. Lowering the threshold may also increase false alerts. Test recordings without screams, especially loud speech and cheering.

## Troubleshooting

- **`ffmpeg` is not recognized:** Reopen PowerShell or restart your IDE after installing FFmpeg, then run `ffmpeg -version`.
- **TensorFlow import errors:** Confirm that the command uses `.venv-scream`, not `.venv`, and check the installed packages with `pip check` as shown above.
- **No live alerts and mic level near zero:** Test the selected recording device in Windows or with `mic_test.py` before changing model settings.
- **Recording CSV has only a header:** The file decoded, but no interval met the threshold and consecutive-window rule. Check the terminal's highest score and compare with what you hear in the recording.
