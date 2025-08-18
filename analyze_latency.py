#!/usr/bin/env python3
"""
Latency and Jitter Analysis Script

This script analyzes the latency and jitter between CAN throttle messages
and ROS cmd_vel topic messages to evaluate system performance.

The script measures latency from CAN message sent to ROS message received (CAN to ROS).

PREREQUISITES:
- CSV file must be preprocessed to contain only relevant messages in chronological order
- Messages should alternate between CAN and ROS messages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_data(csv_file):
    """Load and prepare the data from the combined CSV file."""
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_numeric(df['timestamp'])

    # Separate CAN and ROS messages
    can_df = df[df['message'] == 'can_throttle'].copy()
    ros_df = df[df['message'] == '/rover/radio/cmd_vel'].copy()

    print(f"CAN messages: {len(can_df)}")
    print(f"ROS messages: {len(ros_df)}")

    # Sort by timestamp
    can_df = can_df.sort_values('timestamp').reset_index(drop=True)
    ros_df = ros_df.sort_values('timestamp').reset_index(drop=True)

    return can_df, ros_df

def calculate_latency_and_jitter(can_df, ros_df):
    """
    Calculate latency and jitter between CAN and ROS messages.

    Args:
        can_df: DataFrame with CAN messages
        ros_df: DataFrame with ROS messages

    Returns:
        DataFrame with latency analysis results
    """
    print("Calculating latency and jitter (CAN to ROS)...")

    # Find the minimum number of messages to pair
    min_messages = min(len(can_df), len(ros_df))
    print(f"Using {min_messages} message pairs for analysis")

    latencies = []
    can_timestamps = []
    ros_timestamps = []

    # Pair messages by index (assuming they are in chronological order)
    for idx in range(min_messages):
        can_row = can_df.iloc[idx]
        ros_row = ros_df.iloc[idx]

        can_time = can_row['timestamp']
        ros_time = ros_row['timestamp']

        # Measure time from CAN message to ROS message
        latency = ros_time - can_time

        latencies.append(latency)
        can_timestamps.append(can_time)
        ros_timestamps.append(ros_time)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'can_timestamp': can_timestamps,
        'ros_timestamp': ros_timestamps,
        'latency': latencies
    })

    # Calculate jitter (variation in latency)
    if len(results_df) > 1:
        results_df['jitter'] = results_df['latency'].diff().abs()

    print(f"Analyzed {len(results_df)} message pairs")
    return results_df

def create_plots(results_df, output_dir="plots"):
    """Create comprehensive plots for latency and jitter analysis."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Blues")

    # Direction labels for plots
    source_label = 'CAN'
    target_label = 'ROS'
    source_timestamp = 'can_timestamp'

    # Latency time series
    plt.figure(figsize=(12, 8))
    # Calculate elapsed time from the first timestamp
    elapsed_time = results_df[source_timestamp] - results_df[source_timestamp].min()
    plt.plot(elapsed_time, results_df['latency'] * 1000000,
             alpha=0.5, label='Raw Latency', color='blue')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Latency (μs)')
    plt.title('Latency Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    time_series_file = os.path.join(plots_dir, 'latency_timeseries.png')
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency time series plot saved to: {time_series_file}")

    # Latency histogram
    plt.figure(figsize=(12, 8))
    plt.hist(results_df['latency'] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Frequency')
    plt.title(f'{source_label} to {target_label} Latency Distribution')
    plt.grid(True, alpha=0.3)
    latency_hist_file = os.path.join(plots_dir, 'latency_histogram.png')
    plt.savefig(latency_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency histogram saved to: {latency_hist_file}")

    # Jitter time series
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        # Calculate elapsed time from the first timestamp
        elapsed_time = results_df[source_timestamp][1:] - results_df[source_timestamp].min()
        plt.plot(elapsed_time, results_df['jitter'][1:] * 1000000,
                alpha=0.6, linewidth=1, color='blue')
        plt.xlabel('Elapsed Time (s)')
        plt.ylabel('Jitter (μs)')
        plt.title('Jitter Time Series')
        plt.grid(True, alpha=0.3)
        jitter_timeseries_file = os.path.join(plots_dir, 'jitter_timeseries.png')
        plt.savefig(jitter_timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter time series plot saved to: {jitter_timeseries_file}")

    # Jitter histogram
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        plt.hist(results_df['jitter'][1:] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Jitter (μs)')
        plt.ylabel('Frequency')
        plt.title(f'{source_label} to {target_label} Jitter Distribution')
        plt.grid(True, alpha=0.3)
        jitter_hist_file = os.path.join(plots_dir, 'jitter_histogram.png')
        plt.savefig(jitter_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter histogram saved to: {jitter_hist_file}")

def print_statistics(results_df, output_dir="plots"):
    """Print comprehensive statistics about the latency and jitter."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Direction labels (CAN to ROS)
    source_label = 'CAN'
    target_label = 'ROS'
    source_timestamp = 'can_timestamp'

    # Prepare the statistics text
    stats_text = []
    stats_text.append("="*60)
    stats_text.append(f"LATENCY AND JITTER ANALYSIS RESULTS ({source_label} to {target_label})")
    stats_text.append("="*60)

    latency_us = results_df['latency'] * 1000000

    stats_text.append("\nLatency Statistics (μs):")
    stats_text.append(f"  Count: {len(results_df)}")
    stats_text.append(f"  Mean: {latency_us.mean():.3f}")
    stats_text.append(f"  Median: {latency_us.median():.3f}")
    stats_text.append(f"  Std Dev: {latency_us.std():.3f}")
    stats_text.append(f"  Min: {latency_us.min():.3f}")
    stats_text.append(f"  Max: {latency_us.max():.3f}")
    stats_text.append(f"  95th percentile: {latency_us.quantile(0.95):.3f}")
    stats_text.append(f"  99th percentile: {latency_us.quantile(0.99):.3f}")

    if 'jitter' in results_df.columns:
        jitter_us = results_df['jitter'][1:] * 1000000
        stats_text.append("\nJitter Statistics (μs):")
        stats_text.append(f"  Count: {len(jitter_us)}")
        stats_text.append(f"  Mean: {jitter_us.mean():.3f}")
        stats_text.append(f"  Median: {jitter_us.median():.3f}")
        stats_text.append(f"  Std Dev: {jitter_us.std():.3f}")
        stats_text.append(f"  Min: {jitter_us.min():.3f}")
        stats_text.append(f"  Max: {jitter_us.max():.3f}")
        stats_text.append(f"  95th percentile: {jitter_us.quantile(0.95):.3f}")
        stats_text.append(f"  99th percentile: {jitter_us.quantile(0.99):.3f}")

    # Calculate time span
    time_span = results_df[source_timestamp].max() - results_df[source_timestamp].min()
    stats_text.append("\nTime Analysis:")
    stats_text.append(f"  Time span: {time_span:.2f} seconds")
    stats_text.append(f"  Messages per second: {len(results_df) / time_span:.2f}")

    # Latency categories
    low_latency = len(latency_us[latency_us <= 500])  # ≤500μs
    medium_latency = len(latency_us[(latency_us > 500) & (latency_us <= 1000)])  # 500-1000μs
    high_latency = len(latency_us[latency_us > 1000])  # >1000μs

    stats_text.append("\nLatency Categories:")
    stats_text.append(f"  ≤500μs: {low_latency} ({low_latency/len(results_df)*100:.2f}%)")
    stats_text.append(f"  500-1000μs: {medium_latency} ({medium_latency/len(results_df)*100:.2f}%)")
    stats_text.append(f"  >1000μs: {high_latency} ({high_latency/len(results_df)*100:.2f}%)")

    # Print to stdout
    for line in stats_text:
        print(line)

    # Save to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('\n'.join(stats_text))

    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze latency and jitter between CAN and ROS messages')
    parser.add_argument('--csv-file', required=True, help='Path to the combined CSV file containing CAN and ROS messages')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    can_df, ros_df = load_data(args.csv_file)

    # Calculate latency and jitter
    results_df = calculate_latency_and_jitter(can_df, ros_df)

    # Print statistics
    print_statistics(results_df, args.output_dir)

    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in directory: {args.output_dir}/plots")
        create_plots(results_df, args.output_dir)

    # Save results to CSV
    output_csv = os.path.join(args.output_dir, 'latency_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
