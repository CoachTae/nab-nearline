import deltaRice
import h5py
import matplotlib.pyplot as plt
import nabPy
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def pixel_noise(File, bin_width=3):
    '''
    Extracts noise information for each pixel.

    Input: Pre-loaded HDF5 file
           bin_width: Bin size for histograms (default: 5)
    '''
    # Extract noise waveforms
    noise = File.noiseWaves()

    # Extract noise headers (table)
    noise_headers = noise.headers()

    # Convert pixel column to numpy array
    pixels = noise_headers['pixel'].to_numpy()

    # Get all noise waveforms
    noise_waves = noise.waves()
    noise_waves = noise_waves.compute()

    # Center waves for statistical analysis
    wave_centers = np.median(noise_waves, axis=1)
    centered_waves = noise_waves - wave_centers[:, None]


    # Calculate RMS and STD of each noise waveform
    noise_rms = np.sqrt(np.mean(centered_waves**2, axis=1))
    noise_std = np.std(centered_waves, axis=1)

    # Calculate mean RMS and STD for each unique pixel
    unique_pixels, inverse, counts = np.unique(pixels, return_inverse=True, return_counts=True)
    mean_rms = np.bincount(inverse, weights=noise_rms) / counts
    mean_std = np.bincount(inverse, weights=noise_std) / counts
    baselines = np.bincount(inverse, weights=wave_centers) / counts

    # Histogram start
    # First, calculate half-range based on all waveforms
    max_abs = max(abs(np.min(noise_waves)), abs(np.max(noise_waves)))
    half_range = max(np.ceil(max_abs/bin_width) * bin_width, bin_width)
    start = -half_range
    nbins = int((2 * half_range) / bin_width)

    # Bin index for every sample
    bin_indices = np.floor((noise_waves - start) / bin_width).astype(np.int32)
    np.clip(bin_indices, 0, nbins - 1, out=bin_indices)

    # Combine pixel index and bin index into one flat index
    flat = (inverse.astype(np.int32)[:, None] * nbins + bin_indices).ravel()
    hist = np.bincount(flat, minlength=nbins * len(unique_pixels)).reshape(len(unique_pixels), nbins)

    # Create dictionary for each pixel's noise data
    pixel_noise_data = {
            int(pix): {'rms': rms, 'std': std, 'hist': hist[i], 'baseline': baseline, "n_waveforms": int(count)}
            for i, (pix, rms, std, baseline, count) in enumerate(zip(unique_pixels, mean_rms, mean_std, baselines, counts))
        }

    # Trim trailing and leading bins
    for pix in unique_pixels:
        hist_data = pixel_noise_data[pix]['hist']
        # Find first and last non-zero bins
        nonzero = np.nonzero(hist_data)[0]
        if len(nonzero) > 0:
            first, last = nonzero[0], nonzero[-1]
            pixel_noise_data[pix]['hist'] = hist_data[first:last+1]
            # Update binning information for this pixel
            pixel_noise_data[pix]['binning'] = {
                'start': start + first * bin_width,
                'bin_width': bin_width,
                'nbins': last - first + 1
            }


    return pixel_noise_data
    
    