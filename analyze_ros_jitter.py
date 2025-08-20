#!/usr/bin/env python3
"""
ROS Jitter Analysis Script

This script analyzes the jitter for CAN messages with ID 101 to evaluate
system performance and timing consistency.

The script measures the time intervals between consecutive CAN messages with ID 101
and calculates jitter statistics.

PREREQUISITES:
- Candump log file containing CAN messages with ID 101
- Messages should be in chronological order
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

def load_candump_data(log_file):
    """Load and prepare the data from the candump log file."""
    print(f"Loading data from: {log_file}")

    messages = []

    with open(log_file, 'r') as f:
        for line in f:
            # Parse candump format: (timestamp) can0 101#data T/R
            match = re.match(r'\(([0-9.]+)\) can0 101#([0-9A-F]+) ([TR])', line.strip())
            if match:
                timestamp = float(match.group(1))
                data = match.group(2)
                direction = match.group(3)
                messages.append({
                    'timestamp': timestamp,
                    'data': data,
                    'direction': direction
                })

    df = pd.DataFrame(messages)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"CAN messages with ID 101: {len(df)}")
    return df

def calculate_jitter(can_df):
    """
    Calculate jitter between consecutive CAN messages with ID 101.

    Args:
        can_df: DataFrame with CAN messages

    Returns:
        DataFrame with jitter analysis results
    """
    print("Calculating jitter for CAN messages with ID 101...")

    if len(can_df) < 2:
        print("Warning: Need at least 2 messages to calculate jitter")
        return pd.DataFrame()

    # Calculate intervals between consecutive messages
    intervals = []
    timestamps = []
    data_values = []
    directions = []

    for i in range(1, len(can_df)):
        prev_row = can_df.iloc[i-1]
        curr_row = can_df.iloc[i]

        interval = curr_row['timestamp'] - prev_row['timestamp']

        intervals.append(interval)
        timestamps.append(curr_row['timestamp'])
        data_values.append(curr_row['data'])
        directions.append(curr_row['direction'])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'interval': intervals,
        'data': data_values,
        'direction': directions
    })

    # Calculate jitter (variation in intervals)
    if len(results_df) > 1:
        results_df['jitter'] = results_df['interval'].diff().abs()

    print(f"Analyzed {len(results_df)} intervals")
    return results_df

def create_plots(results_df, output_dir="plots"):
    """Create comprehensive plots for jitter analysis."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Blues")

    # Message interval histogram
    plt.figure(figsize=(12, 8))
    plt.hist(results_df['interval'] * 1000, bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.xlabel('Interval (ms)')
    plt.ylabel('Frequency')
    plt.title('Controller Node Transmission Interval Distribution')
    plt.grid(True, alpha=0.3)
    interval_hist_file = os.path.join(plots_dir, 'interval_histogram.png')
    plt.savefig(interval_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Interval histogram saved to: {interval_hist_file}")

    # Combined interval and jitter time series with dual y-axis
    if 'jitter' in results_df.columns:
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Calculate elapsed time from the first timestamp
        elapsed_time = results_df['timestamp'] - results_df['timestamp'].min()
        elapsed_time_jitter = results_df['timestamp'][1:] - results_df['timestamp'].min()

        # Plot intervals on primary y-axis (left)
        color1 = 'blue'
        ax1.plot(elapsed_time, results_df['interval'] * 1000,
                color=color1, alpha=0.7, linewidth=1, label='Interval')
        ax1.set_xlabel('Elapsed Time (s)')
        ax1.set_ylabel('Interval (ms)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(bottom=7)  # Set interval y-axis to start from 7 ms

        # Create secondary y-axis for jitter (right)
        ax2 = ax1.twinx()
        color2 = 'red'

        ax2.plot(elapsed_time_jitter, results_df['jitter'][1:] * 1000,
                color=color2, alpha=0.7, linewidth=1, label='Jitter')
        ax2.set_ylabel('Jitter (ms)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 5)  # Set jitter y-axis to go from 0 to 5 ms

        # Add grid and title
        ax1.grid(True, alpha=0.3)
        plt.title('Controller Node Transmission Interval and Jitter Time Series')

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        combined_timeseries_file = os.path.join(plots_dir, 'interval_jitter_combined.png')
        plt.savefig(combined_timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined interval and jitter time series plot saved to: {combined_timeseries_file}")

    # Jitter histogram
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        plt.hist(results_df['jitter'][1:] * 1000, bins=50, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Jitter (ms)')
        plt.ylabel('Frequency')
        plt.title('Controller Node Transmission Jitter Distribution')
        plt.grid(True, alpha=0.3)
        jitter_hist_file = os.path.join(plots_dir, 'jitter_histogram.png')
        plt.savefig(jitter_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter histogram saved to: {jitter_hist_file}")


def print_statistics(results_df, output_dir="plots"):
    """Print comprehensive statistics about the jitter analysis."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the statistics text
    stats_text = []
    stats_text.append("="*60)
    stats_text.append("Controller Node Transmission JITTER ANALYSIS RESULTS")
    stats_text.append("="*60)

    interval_ms = results_df['interval'] * 1000

    stats_text.append("\nInterval Statistics (ms):")
    stats_text.append(f"  Count: {len(results_df)}")
    stats_text.append(f"  Mean: {interval_ms.mean():.3f}")
    stats_text.append(f"  Median: {interval_ms.median():.3f}")
    stats_text.append(f"  Std Dev: {interval_ms.std():.3f}")
    stats_text.append(f"  Min: {interval_ms.min():.3f}")
    stats_text.append(f"  Max: {interval_ms.max():.3f}")
    stats_text.append(f"  95th percentile: {interval_ms.quantile(0.95):.3f}")
    stats_text.append(f"  99th percentile: {interval_ms.quantile(0.99):.3f}")

    if 'jitter' in results_df.columns:
        jitter_ms = results_df['jitter'][1:] * 1000
        stats_text.append("\nJitter Statistics (ms):")
        stats_text.append(f"  Count: {len(jitter_ms)}")
        stats_text.append(f"  Mean: {jitter_ms.mean():.3f}")
        stats_text.append(f"  Median: {jitter_ms.median():.3f}")
        stats_text.append(f"  Std Dev: {jitter_ms.std():.3f}")
        stats_text.append(f"  Min: {jitter_ms.min():.3f}")
        stats_text.append(f"  Max: {jitter_ms.max():.3f}")
        stats_text.append(f"  95th percentile: {jitter_ms.quantile(0.95):.3f}")
        stats_text.append(f"  99th percentile: {jitter_ms.quantile(0.99):.3f}")

    # Calculate time span
    time_span = results_df['timestamp'].max() - results_df['timestamp'].min()
    stats_text.append("\nTime Analysis:")
    stats_text.append(f"  Time span: {time_span:.2f} seconds")
    stats_text.append(f"  Messages per second: {len(results_df) / time_span:.2f}")

    # Interval categories
    low_interval = len(interval_ms[interval_ms <= 10])  # ≤10ms
    medium_interval = len(interval_ms[(interval_ms > 10) & (interval_ms <= 20)])  # 10-20ms
    high_interval = len(interval_ms[interval_ms > 20])  # >20ms

    stats_text.append("\nInterval Categories:")
    stats_text.append(f"  ≤10ms: {low_interval} ({low_interval/len(results_df)*100:.2f}%)")
    stats_text.append(f"  10-20ms: {medium_interval} ({medium_interval/len(results_df)*100:.2f}%)")
    stats_text.append(f"  >20ms: {high_interval} ({high_interval/len(results_df)*100:.2f}%)")

    # Direction statistics (if applicable)
    if 'direction' in results_df.columns:
        stats_text.append("\nDirection Statistics:")
        direction_counts = results_df['direction'].value_counts()
        for direction, count in direction_counts.items():
            percentage = count / len(results_df) * 100
            stats_text.append(f"  {direction}: {count} ({percentage:.2f}%)")

    # Print to stdout
    for line in stats_text:
        print(line)

    # Save to file
    results_file = os.path.join(output_dir, 'jitter_results.txt')
    with open(results_file, 'w') as f:
        f.write('\n'.join(stats_text))

    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze jitter for CAN messages with ID 101')
    parser.add_argument('--log-file', required=True, help='Path to the candump log file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    can_df = load_candump_data(args.log_file)

    if len(can_df) == 0:
        print("No CAN messages with ID 101 found in the log file.")
        return

    # Calculate jitter
    results_df = calculate_jitter(can_df)

    if len(results_df) == 0:
        print("Insufficient data to calculate jitter.")
        return

    # Print statistics
    print_statistics(results_df, args.output_dir)

    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in directory: {args.output_dir}/plots")
        create_plots(results_df, args.output_dir)

    # Save results to CSV
    output_csv = os.path.join(args.output_dir, 'jitter_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
