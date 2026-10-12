import deltaRice
import h5py
import matplotlib.pyplot as plt
import nabPy
import NoiseHistogramsPerPixel as PixelNoise
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys


file_path = (
    Path("~/nab-nearline/SampleData")
    .expanduser()
    / "Runs10100"
    / "Run10102_0.h5"
)

data = nabPy.File(str(file_path))

pixel_noise_data = PixelNoise.pixel_noise(data, bin_width=3)


from NoiseHistogramReport import plot_all_noise_histograms

plot_all_noise_histograms(
    pixel_noise_data,
    output_pdf="run10102_histograms.pdf",
    suspicious_csv="run10102_suspicious_histograms.csv"
)
sys.exit()

def robust_sigma(values):
    """
    Robust estimate of the standard deviation.

    Unlike np.std(), this is much less sensitive to occasional large spikes.
    """
    values = np.asarray(values)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    return 1.4826 * mad


def measure_baseline_waveform(wave, chunk_size=256):
    """
    Measure fast noise and slow baseline motion in one baseline waveform.
    """
    chunk_medians = []
    chunk_noise = []

    for start in range(0, len(wave) - chunk_size + 1, chunk_size):
        chunk = wave[start:start + chunk_size]

        chunk_medians.append(np.median(chunk))
        chunk_noise.append(robust_sigma(chunk))

    chunk_medians = np.asarray(chunk_medians)
    chunk_noise = np.asarray(chunk_noise)

    return {
        "baseline_level": np.median(chunk_medians),
        "fast_noise": np.median(chunk_noise),
        "baseline_wander": robust_sigma(chunk_medians),
    }


baselines = data.noiseWaves()
headers = baselines.headers().reset_index(drop=True)

rows = []

for i in range(baselines.numWaves):
    wave = baselines.wave(i, prep=True)
    header = headers.iloc[i]

    measurements = measure_baseline_waveform(wave)

    rows.append({
        "waveform": i,
        "pixel": int(header["pixel"]),
        "board": int(header["board"]),
        "channel": int(header["channel"]),
        **measurements,
    })


waveform_results = pd.DataFrame(rows)

pixel_results = (
    waveform_results
    .groupby(["pixel", "board", "channel"])
    .agg(
        n_waveforms=("waveform", "count"),
        baseline_level=("baseline_level", "median"),
        fast_noise=("fast_noise", "median"),
        baseline_wander=("baseline_wander", "median"),
    )
    .reset_index()
)


print("\nPER-PIXEL BASELINE RESULTS")
print(pixel_results.to_string(index=False))


output_dir = Path("~/nab-nearline/tests/output").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "baseline_noise_by_pixel.csv"
pixel_results.to_csv(output_file, index=False)

print(f"\nSaved results to: {output_file}")