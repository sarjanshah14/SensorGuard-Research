#!/usr/bin/env python
import os
import sys
import argparse
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for high-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

SKAB_ROOT = 'SKAB'

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def find_all_csv_files(root_dir):
    """Recursively collect every CSV file path under root_dir."""
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found. "
              f"Make sure you run this script in the project root.")
        sys.exit(1)

    paths = []
    for root, _dirs, files in os.walk(root_dir):
        for f in sorted(files):
            if f.endswith('.csv'):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def list_files(paths):
    """Print discovered CSV files with sizes."""
    print(f"\n=== Discovered SKAB CSV Files ({len(paths)} total) ===")
    for p in paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p}  ({size_kb:.1f} KB)")
    print()


# ---------------------------------------------------------------------------
# Loading & stacking
# ---------------------------------------------------------------------------

def load_single(file_path):
    """Load one semicolon-delimited CSV file."""
    df = pd.read_csv(file_path, sep=';', parse_dates=['datetime'],
                     index_col='datetime')
    return df


def load_all_combined(paths):
    """
    Load every CSV, tag each row with 'category' and 'source_file'
    metadata columns, then stack everything into one DataFrame sorted
    by the datetime index.
    """
    frames = []
    print(f"\nLoading {len(paths)} CSV files from SKAB/ ...")
    for p in paths:
        relative = os.path.relpath(p, SKAB_ROOT)
        category = os.path.dirname(relative)        # e.g. 'valve1'
        source   = os.path.splitext(os.path.basename(p))[0]  # e.g. '0'

        try:
            df = load_single(p)
        except Exception as e:
            print(f"  [WARN] Skipping {p}: {e}")
            continue

        df['category']    = category
        df['source_file'] = source
        frames.append(df)
        print(f"  ✓  {p}  →  {len(df):,} rows")

    if not frames:
        print("No files could be loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames).sort_index()
    print(f"\nCombined dataset: {len(combined):,} rows  ×  {combined.shape[1]} columns")
    return combined


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary_statistics(df):
    """Summary for the full combined dataset."""
    print("\n=== Combined Dataset Summary ===")
    print(f"Total rows (timestamps) : {len(df):,}")
    print(f"Total columns           : {df.shape[1]}")
    print(f"Datetime range          : {df.index.min()}  →  {df.index.max()}")

    # Per-category breakdown
    print("\nRows per category:")
    cat_counts = df.groupby('category').size().sort_index()
    for cat, n in cat_counts.items():
        print(f"  {cat:<20} {n:>8,} rows")

    # Anomaly stats (only files that have the column)
    if 'anomaly' in df.columns:
        total_anom   = df['anomaly'].sum()
        total_pct    = total_anom / len(df) * 100
        changepoints = df['changepoint'].sum() if 'changepoint' in df.columns else 'N/A'
        print(f"\nAnomaly rows (total)  : {int(total_anom):,} / {len(df):,}  ({total_pct:.2f}%)")
        print(f"Changepoints (total)  : {changepoints}")

        print("\nAnomaly % per category:")
        for cat, g in df.groupby('category'):
            if 'anomaly' in g.columns:
                pct = g['anomaly'].sum() / len(g) * 100
                print(f"  {cat:<20} {pct:>6.2f}%")
    else:
        print("\nNo anomaly column found in combined data.")

    # Sensor summary
    meta_cols  = {'anomaly', 'changepoint', 'category', 'source_file'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    if sensor_cols:
        print("\nSensor channel statistics (across all files):")
        summary = df[sensor_cols].describe().T[['mean', 'std', 'min', 'max']]
        print(summary.to_string())


# ---------------------------------------------------------------------------
# Plotting – full combined dataset
# ---------------------------------------------------------------------------

def plot_anomaly_distribution(df, output_dir=None):
    """
    Bar chart: anomaly row counts per (category, source_file) pair.
    Only drawn when the 'anomaly' column is present.
    """
    if 'anomaly' not in df.columns:
        print("Skipping anomaly distribution plot (no 'anomaly' column).")
        return

    agg = (df.groupby(['category', 'source_file'])['anomaly']
             .agg(['sum', 'count'])
             .rename(columns={'sum': 'anomaly_rows', 'count': 'total_rows'}))
    agg['anomaly_pct'] = agg['anomaly_rows'] / agg['total_rows'] * 100
    agg = agg.reset_index()
    agg['label'] = agg['category'] + '/' + agg['source_file']

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(agg)))
    bars = ax.bar(agg['label'], agg['anomaly_pct'], color=colors)
    ax.set_xlabel("Dataset file")
    ax.set_ylabel("Anomaly %")
    ax.set_title("Anomaly Percentage per SKAB File (Combined Dataset)", weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    _save_or_show(fig, output_dir, "combined_anomaly_distribution.png")


def plot_correlation_heatmap(df, output_dir=None):
    """Correlation heatmap of all sensor channels over the combined dataset."""
    meta_cols   = {'anomaly', 'changepoint', 'category', 'source_file'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    corr = df[sensor_cols].corr()

    fig = plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5,
                cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    plt.title("Sensor Correlation Matrix (All SKAB Files Combined)", weight='bold', pad=15)
    plt.tight_layout()

    _save_or_show(fig, output_dir, "combined_correlation_heatmap.png")


def plot_sensor_distributions(df, output_dir=None):
    """
    Box-plot: distribution of each sensor column, broken down by category.
    Gives a quick cross-category comparison for the paper.
    """
    meta_cols   = {'anomaly', 'changepoint', 'category', 'source_file'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    n = len(sensor_cols)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Sensor Value Distributions per Category (All SKAB Files)", weight='bold')

    categories = sorted(df['category'].unique())
    palette    = sns.color_palette("tab10", len(categories))

    for ax, col in zip(axes, sensor_cols):
        data_by_cat = [df.loc[df['category'] == cat, col].dropna().values
                       for cat in categories]
        bp = ax.boxplot(data_by_cat, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(categories, rotation=20, ha='right')
        ax.set_ylabel(col, fontsize=9, weight='medium')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "combined_sensor_distributions.png")


def plot_mean_signals_per_category(df, output_dir=None):
    """
    Line plots of per-minute mean sensor values, one subplot per sensor,
    coloured by category — shows how fault types differ over time.
    """
    meta_cols   = {'anomaly', 'changepoint', 'category', 'source_file'}
    sensor_cols = [c for c in df.columns if c not in meta_cols]
    n = len(sensor_cols)

    categories = sorted(df['category'].unique())
    palette    = dict(zip(categories, sns.color_palette("tab10", len(categories))))

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Mean Sensor Signals per Category (1-min resample)", weight='bold')

    for ax, col in zip(axes, sensor_cols):
        for cat in categories:
            sub = df.loc[df['category'] == cat, col].dropna()
            # Resample to 1-min mean to keep plot readable
            resampled = sub.resample('1min').mean()
            ax.plot(resampled.index, resampled.values,
                    label=cat, color=palette[cat], linewidth=1.2, alpha=0.85)
        ax.set_ylabel(col, fontsize=9, weight='medium')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True, facecolor='white',
                  framealpha=0.8, fontsize=8)

    plt.xlabel("Datetime", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, output_dir, "combined_mean_signals_per_category.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, output_dir, filename):
    """Either save the figure to output_dir or display it."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {out_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SKAB Combined Dataset Visualizer — loads ALL CSVs at once"
    )
    parser.add_argument(
        '-l', '--list', action='store_true',
        help="List all available CSV files in SKAB/ and exit"
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='plots',
        help="Directory to save generated plots (default: 'plots')"
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help="Display plots on screen instead of saving them"
    )

    args = parser.parse_args()

    # Discover files
    all_paths = find_all_csv_files(SKAB_ROOT)

    if args.list:
        list_files(all_paths)
        return

    # Load and stack all CSVs
    df = load_all_combined(all_paths)

    # Print summary
    print_summary_statistics(df)

    # Generate plots
    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        print(f"\nGenerating plots → saving to '{output_dir}/' ...")
    else:
        print("\nGenerating plots (displaying on screen) ...")

    plot_anomaly_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_sensor_distributions(df, output_dir)
    plot_mean_signals_per_category(df, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
