#!/usr/bin/env python3
"""Rapid Nab baseline-waveform stability report for several HDF5 subruns.

Usage:
    python quick_noise_report.py /path/RunXXXX_0.h5 ... /path/RunXXXX_4.h5
    python quick_noise_report.py --all /path/to/RunXXXX_*.h5

Output files go to results/today_noise/. This analysis only uses dedicated
baseline waveforms; it does not apply pass/fail thresholds.
"""
import argparse
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def utc_reference(path):
    """Return matching (UTC seconds, DAQ ticks) acquisition timestamp pair."""
    with h5py.File(path, 'r') as f:
        for dataset, utc_field, daq_field in (
            ('Parameters/RunTime/SubStartTime', 'SubStartTimeUTC', 'SubStartTimeDAQ'),
            ('Parameters/RunTime/StartTime', 'StartTimeUTC', 'StartTimeDAQ'),
        ):
            if dataset in f:
                record = f[dataset][()]
                if record.dtype.names and {utc_field, daq_field}.issubset(record.dtype.names):
                    return int(record[utc_field]), int(record[daq_field])
    raise ValueError(f'No paired UTC/DAQ start time in {path}')


def read_subrun(path):
    import nabPy

    match = re.search(r'Run(\d+)_(\d+)\.h5$', path.name)
    run = int(match.group(1)) if match else -1
    subrun = int(match.group(2)) if match else -1

    data = nabPy.File(str(path))
    noise = data.noiseWaves()
    headers = noise.headers().reset_index(drop=True)
    if headers is None or headers.empty:
        raise ValueError(f'No baseline headers in {path}')

    # Prepared ADC units, as in the current noise histogram analysis.
    waveforms = noise.waves(prep=True).compute()
    if waveforms.ndim != 2 or waveforms.shape[0] != len(headers):
        raise ValueError(f'Header/waveform shape mismatch in {path}: {waveforms.shape}')

    required = {'timestamp', 'pixel', 'board', 'channel', 'bc'}
    missing = required.difference(headers.columns)
    if missing:
        raise ValueError(f'Missing header fields: {sorted(missing)}')

    ref_utc, ref_daq = utc_reference(path)
    raw_ticks = headers['timestamp'].to_numpy(dtype=np.int64)
    # Cast before subtraction to avoid unsigned-integer underflow.
    unix_seconds = ref_utc + (raw_ticks - ref_daq) * 4e-9

    result = pd.DataFrame({
        'file': path.name,
        'run': run,
        'subrun': subrun,
        'pixel': headers['pixel'].to_numpy(dtype=int),
        'board': headers['board'].to_numpy(dtype=int),
        'channel': headers['channel'].to_numpy(dtype=int),
        'bc': headers['bc'].to_numpy(dtype=int),
        'daq_timestamp_ticks': raw_ticks,
        'unix_seconds': unix_seconds,
        'mean_adc': np.mean(waveforms, axis=1),
        'std_adc': np.std(waveforms, axis=1, ddof=0),
        'n_samples': waveforms.shape[1],
    })
    return result


def make_timeseries_pdf(df, bcs, outfile):
    """Each row: one readout channel, showing baseline mean and noise STD."""
    with PdfPages(outfile) as pdf:
        for offset in range(0, len(bcs), 4):
            chosen = bcs[offset:offset + 4]
            fig, axes = plt.subplots(len(chosen), 2, figsize=(11.7, 8.3),
                                     squeeze=False, sharex=True)
            for row, bc in enumerate(chosen):
                part = df.loc[df['bc'] == bc].sort_values('unix_seconds')
                pxs = sorted(part['pixel'].unique().tolist())
                pixel_label = ','.join(map(str, pxs))
                description = (f'Pixel {pixel_label}  |  Board {int(part.board.iloc[0])}, '
                               f'channel {int(part.channel.iloc[0])}  |  N={len(part)}')
                for ax, column, label in (
                    (axes[row, 0], 'mean_adc', 'Baseline mean (ADC)'),
                    (axes[row, 1], 'std_adc', 'Noise STD (ADC)'),
                ):
                    # Join consecutive observations within a subrun only;
                    # avoid drawing misleading slopes across acquisition gaps.
                    for _, segment in part.groupby('file', sort=False):
                        ax.plot(segment['elapsed_min'], segment[column], 'o-',
                                markersize=2.8, linewidth=0.8)
                    ax.set_ylabel(label, fontsize=8)
                    ax.grid(alpha=0.3)
                    ax.tick_params(labelsize=8)
                axes[row, 0].set_title(description, loc='left', fontsize=9)
            axes[-1, 0].set_xlabel('Minutes since first waveform')
            axes[-1, 1].set_xlabel('Minutes since first waveform')
            fig.suptitle('Nab baseline stability | waveform-level data', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path, help='Five HDF5 subrun paths')
    parser.add_argument('--all', action='store_true', help='Also plot every readout channel')
    parser.add_argument('--output', type=Path,
                        default=Path('results/today_noise'), help='Output folder')
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for p in args.files]
    for path in paths:
        if not path.is_file():
            parser.error(f'No such file: {path}')
    args.output.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in paths:
        print(f'Reading {path.name}...', flush=True)
        frames.append(read_subrun(path))
    data = pd.concat(frames, ignore_index=True).sort_values('unix_seconds')
    data['utc_time'] = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
    data['elapsed_min'] = (data['unix_seconds'] - data['unix_seconds'].min()) / 60
    data.to_csv(args.output / 'waveform_noise_measurements.csv', index=False)

    # Group by physical readout channel, NOT pixel ID: pixel mappings can collide.
    channel = data.groupby('bc', sort=True).agg(
        board=('board', 'first'), channel=('channel', 'first'),
        pixel=('pixel', 'first'), n_pixels_mapped=('pixel', 'nunique'),
        n_waveforms=('std_adc', 'size'),
        baseline_average=('mean_adc', 'mean'),
        std_average=('std_adc', 'mean'),
        baseline_min=('mean_adc', 'min'), baseline_max=('mean_adc', 'max'),
        std_min=('std_adc', 'min'), std_max=('std_adc', 'max'),
    ).reset_index()
    channel['baseline_span_adc'] = channel['baseline_max'] - channel['baseline_min']
    channel['std_span_adc'] = channel['std_max'] - channel['std_min']
    channel.to_csv(args.output / 'channel_summary.csv', index=False)

    # Report repeated pixel IDs *within each subrun* rather than assuming
    # that a pixel identifier is a one-to-one hardware mapping.
    mapping = data[['file', 'run', 'subrun', 'pixel', 'bc', 'board', 'channel']].drop_duplicates()
    pixel_aliases = (mapping.groupby(['file', 'pixel'], as_index=False)
                     .agg(n_readout_channels=('bc', 'nunique')))
    pixel_aliases = pixel_aliases.loc[pixel_aliases.n_readout_channels > 1]
    pixel_aliases.to_csv(args.output / 'shared_pixel_ids.csv', index=False)
    channel_mapping_changes = channel.loc[channel.n_pixels_mapped > 1]
    channel_mapping_changes.to_csv(args.output / 'channel_mapping_changes.csv', index=False)

    coverage = data.groupby('file').agg(
        n_waveforms=('bc', 'size'), n_channels=('bc', 'nunique'),
        n_pixel_ids=('pixel', 'nunique'),
        first_utc=('utc_time', 'min'), last_utc=('utc_time', 'max'),
    )
    coverage.to_csv(args.output / 'subrun_coverage.csv')

    eligible = channel.loc[channel.n_waveforms >= 3]
    selected = list(dict.fromkeys(
        eligible.nlargest(6, 'baseline_span_adc').bc.tolist()
        + eligible.nlargest(6, 'std_span_adc').bc.tolist()
    ))
    if selected:
        make_timeseries_pdf(data, selected, args.output / 'priority_channel_plots.pdf')
    if args.all:
        make_timeseries_pdf(data, channel.bc.tolist(), args.output / 'all_channel_plots.pdf')

    first, last = data['utc_time'].min(), data['utc_time'].max()
    top_mean = channel.nlargest(10, 'baseline_span_adc')
    top_std = channel.nlargest(10, 'std_span_adc')
    lines = [
        '# Same-day Nab baseline-noise review', '',
        f'- Files analyzed: {len(paths)}',
        f'- UTC coverage: {first} to {last}',
        f'- Total baseline waveforms: {len(data)}',
        f'- Distinct readout channels (BC): {data.bc.nunique()}',
        f'- Distinct pixel IDs: {data.pixel.nunique()}',
        f'- Subrun/pixel combinations mapping to multiple channels: {len(pixel_aliases)}',
        f'- Channels whose pixel mapping differs among supplied files: {len(channel_mapping_changes)}',
        '', '## Largest observed baseline-mean spans (not pass/fail)',
        '', '| BC | Pixel | N waveforms | Max minus min (ADC) |',
        '|---:|---:|---:|---:|',
    ]
    for r in top_mean.itertuples(index=False):
        lines.append(f'| {r.bc} | {r.pixel} | {r.n_waveforms} | {r.baseline_span_adc:.2f} |')
    lines += ['', '## Largest observed noise-STD spans (not pass/fail)', '',
              '| BC | Pixel | N waveforms | Max minus min (ADC) |',
              '|---:|---:|---:|---:|']
    for r in top_std.itertuples(index=False):
        lines.append(f'| {r.bc} | {r.pixel} | {r.n_waveforms} | {r.std_span_adc:.2f} |')
    lines += ['', '## Interpretation limits', '',
              '- These are relative changes, not evidence that a channel is faulty.',
              '- Noise statistics alone cannot establish physics-data quality or required additional exposure.',
              '- Check shared_pixel_ids.csv before presenting anything grouped solely by pixel ID.',
              '- Absolute timestamps use the paired UTC/DAQ start references; check that the listed UTC window matches the acquisition log.',
              '- A few observations per channel can suggest trends but do not establish periodicity.',
              '']
    (args.output / 'review_summary.md').write_text('\n'.join(lines))
    print('\nDone. Results:', args.output.resolve())
    print('UTC window:', first, 'to', last)
    print('Waveforms:', len(data), '| channels:', data.bc.nunique())
    print('Shared pixel IDs:', len(pixel_aliases), '| Mapping changes:', len(channel_mapping_changes))
    print('Start with review_summary.md and priority_channel_plots.pdf')


if __name__ == '__main__':
    main()
