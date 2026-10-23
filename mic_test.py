import numpy as np
import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())

index = int(input("\nEnter the number of the microphone to test: "))
device = sd.query_devices(index, "input")
rate = int(device["default_samplerate"])
print(f"Testing: {device['name']} at {rate} Hz. Press Ctrl+C to stop.")

try:
    with sd.InputStream(
        device=index,
        samplerate=rate,
        channels=1,
        dtype="float32",
    ) as stream:
        while True:
            audio, overflowed = stream.read(rate // 2)
            samples = audio[:, 0]
            rms = float(np.sqrt(np.mean(samples**2)))
            peak = float(np.max(np.abs(samples)))
            print(
                f"RMS: {rms:.8f} | Peak: {peak:.8f}"
                + (" | OVERFLOW" if overflowed else ""),
                flush=True,
            )
except KeyboardInterrupt:
    print("\nStopped.")
