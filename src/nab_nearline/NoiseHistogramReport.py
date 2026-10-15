from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


"""
NOT A PRIMARY NEARLINE ANALYSIS TOOL (yet)!!!

This module was a one-off test. It is possible that it may be implemented into the nearline analysis pipeline at some point.
Until that decision is made, it should not be deleted.
"""

def find_histogram_peaks(
    counts,
    smooth_window=3,
    min_height_fraction=0.15,
    min_separation_bins=3
):
    """
    Very simple peak finder for histogram counts.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    smooth : np.ndarray
        Smoothed histogram counts.
    """
    counts = np.asarray(counts, dtype=float)

    if counts.size == 0 or np.max(counts) <= 0:
        return np.array([], dtype=int), counts

    # Small moving-average smoothing
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    smooth = np.convolve(counts, kernel, mode="same")

    threshold = min_height_fraction * np.max(smooth)

    candidate_peaks = []

    for i in range(1, len(smooth) - 1):
        if (
            smooth[i] >= smooth[i - 1]
            and smooth[i] > smooth[i + 1]
            and smooth[i] >= threshold
        ):
            candidate_peaks.append(i)

    # Merge peaks that are too close together
    filtered_peaks = []

    for idx in candidate_peaks:
        if not filtered_peaks:
            filtered_peaks.append(idx)
        else:
            if idx - filtered_peaks[-1] < min_separation_bins:
                # keep whichever is taller
                if smooth[idx] > smooth[filtered_peaks[-1]]:
                    filtered_peaks[-1] = idx
            else:
                filtered_peaks.append(idx)

    return np.asarray(filtered_peaks, dtype=int), smooth


def plot_all_noise_histograms(
    pixel_noise_data,
    output_pdf="all_noise_histograms.pdf",
    suspicious_csv="suspicious_histograms.csv",
    plots_per_page=6
):
    """
    Make a PDF containing all pixel histograms and save a CSV listing
    suspicious histograms (e.g. multi-peak shapes).

    Expected pixel_noise_data format:
    {
        pixel: {
            "rms": ...,
            "std": ...,
            "baseline": ...,
            "n_waveforms": ...,
            "hist": np.array([...]),
            "binning": {
                "start": ...,
                "bin_width": ...,
                "nbins": ...
            }
        }
    }
    """

    output_pdf = Path(output_pdf)
    suspicious_csv = Path(suspicious_csv)

    pixel_keys = sorted(
        key for key in pixel_noise_data.keys()
        if isinstance(key, (int, np.integer))
    )

    suspicious_rows = []

    # ---------- Build PDF ----------
    with PdfPages(output_pdf) as pdf:

        for page_start in range(0, len(pixel_keys), plots_per_page):

            page_pixels = pixel_keys[page_start:page_start + plots_per_page]

            ncols = 3
            nrows = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9))
            axes = axes.flatten()

            for ax, pixel in zip(axes, page_pixels):

                result = pixel_noise_data[pixel]

                counts = np.asarray(result["hist"])
                baseline = float(result["baseline"])
                rms = float(result["rms"])
                std = float(result["std"])
                n_waveforms = int(result["n_waveforms"])

                binning = result["binning"]
                start = float(binning["start"])
                bin_width = float(binning["bin_width"])
                nbins = int(binning["nbins"])

                edges = start + np.arange(nbins + 1) * bin_width
                centers = (edges[:-1] + edges[1:]) / 2

                peaks, smooth = find_histogram_peaks(counts)
                peak_positions = centers[peaks] if len(peaks) > 0 else np.array([])

                suspicious = len(peaks) >= 2

                # Save suspicious entries
                if suspicious:
                    suspicious_rows.append({
                        "pixel": int(pixel),
                        "n_waveforms": n_waveforms,
                        "baseline": baseline,
                        "rms": rms,
                        "std": std,
                        "n_peaks": int(len(peaks)),
                        "peak_positions_adc": ";".join(
                            f"{p:.2f}" for p in peak_positions
                        )
                    })

                # Raw histogram
                ax.bar(
                    centers,
                    counts,
                    width=bin_width,
                    align="center",
                    alpha=0.8
                )

                # Smoothed curve for easier eyeballing
                ax.plot(centers, smooth, linewidth=1.5)

                # Mark baseline
                ax.axvline(
                    baseline,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Baseline = {baseline:.1f}"
                )

                # Mark detected peaks
                if len(peaks) > 0:
                    ax.scatter(
                        peak_positions,
                        counts[peaks],
                        marker="o",
                        s=20,
                        zorder=3,
                        label=f"Peaks = {len(peaks)}"
                    )

                title = (
                    f"Pixel {pixel}\n"
                    f"RMS={rms:.2f}, STD={std:.2f}, N={n_waveforms}"
                )
                if suspicious:
                    title = "[FLAGGED] " + title

                ax.set_title(title)
                ax.set_xlabel("ADC value")
                ax.set_ylabel("Counts")
                ax.legend(fontsize=8)

            # Hide unused axes
            for ax in axes[len(page_pixels):]:
                ax.set_visible(False)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ---------- Save suspicious CSV ----------
    with open(suspicious_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pixel",
                "n_waveforms",
                "baseline",
                "rms",
                "std",
                "n_peaks",
                "peak_positions_adc",
            ]
        )
        writer.writeheader()
        writer.writerows(suspicious_rows)

    print(f"Saved histogram PDF to: {output_pdf}")
    print(f"Saved suspicious histogram list to: {suspicious_csv}")
    print(f"Flagged {len(suspicious_rows)} suspicious histograms.")