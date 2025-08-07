#!/usr/bin/env python3
"""
Latency and Jitter Analysis Script

This script analyzes the latency and jitter between CAN steering messages
and radio steering topic messages to evaluate system performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_data(can_file, ros_file):
    """Load and prepare the data from both CSV files."""
    print(f"Loading CAN data from: {can_file}")
    can_df = pd.read_csv(can_file)
    can_df['timestamp'] = pd.to_numeric(can_df['timestamp'])

    print(f"Loading ROS data from: {ros_file}")
    ros_df = pd.read_csv(ros_file)
    ros_df['timestamp'] = pd.to_numeric(ros_df['timestamp'])

    print(f"CAN messages: {len(can_df)}")
    print(f"ROS messages: {len(ros_df)}")

    return can_df, ros_df

def calculate_latency_and_jitter(can_df, ros_df, max_latency_threshold=1.0):
    """
    Calculate latency and jitter between CAN and ROS messages.

    Args:
        can_df: DataFrame with CAN messages
        ros_df: DataFrame with ROS messages
        max_latency_threshold: Maximum acceptable latency in seconds

    Returns:
        DataFrame with latency analysis results
    """
    print("Calculating latency and jitter...")

    # Sort both dataframes by timestamp
    can_df = can_df.sort_values('timestamp').reset_index(drop=True)
    ros_df = ros_df.sort_values('timestamp').reset_index(drop=True)

    # Find the closest ROS message for each CAN message
    latencies = []
    can_timestamps = []
    ros_timestamps = []
    can_values = []
    ros_values = []

    for idx, can_row in can_df.iterrows():
        can_time = can_row['timestamp']
        can_value = can_row['value']

        # Find the closest radio message in time
        time_diffs = np.abs(ros_df['timestamp'] - can_time)
        closest_idx = time_diffs.idxmin()
        closest_ros = ros_df.iloc[closest_idx]

        latency = closest_ros['timestamp'] - can_time

        # Only include if latency is within reasonable bounds
        if abs(latency) <= max_latency_threshold:
            latencies.append(latency)
            can_timestamps.append(can_time)
            ros_timestamps.append(closest_ros['timestamp'])
            can_values.append(can_value)
            ros_values.append(closest_ros['value'])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'can_timestamp': can_timestamps,
        'ros_timestamp': ros_timestamps,
        'latency': latencies,
        'can_value': can_values,
        'ros_value': ros_values
    })

    # Calculate jitter (variation in latency)
    if len(results_df) > 1:
        results_df['jitter'] = results_df['latency'].diff().abs()

    print(f"Analyzed {len(results_df)} message pairs")
    return results_df

def create_plots(results_df, output_dir="plots"):
    """Create comprehensive plots for latency and jitter analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Blues")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))  # noqa: F841

    # 1. Latency over time
    plt.subplot(3, 2, 1)
    plt.scatter(results_df['can_timestamp'], results_df['latency'] * 1000000,
                alpha=0.6, s=20, color='blue')
    plt.xlabel('CAN Timestamp')
    plt.ylabel('Latency (μs)')
    plt.title('Latency Over Time')
    plt.grid(True, alpha=0.3)

    # 2. Jitter over time
    plt.subplot(3, 2, 2)
    if 'jitter' in results_df.columns:
        plt.scatter(results_df['can_timestamp'][1:], results_df['jitter'][1:] * 1000000,
                    alpha=0.6, s=20, color='blue')
        plt.xlabel('CAN Timestamp')
        plt.ylabel('Jitter (μs)')
        plt.title('Jitter Over Time')
        plt.grid(True, alpha=0.3)

    # 3. Latency histogram
    plt.subplot(3, 2, 3)
    plt.hist(results_df['latency'] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution')
    plt.grid(True, alpha=0.3)

    # 4. Jitter histogram
    plt.subplot(3, 2, 4)
    if 'jitter' in results_df.columns:
        plt.hist(results_df['jitter'][1:] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Jitter (μs)')
        plt.ylabel('Frequency')
        plt.title('Jitter Distribution')
        plt.grid(True, alpha=0.3)

    # 5. Latency vs Value correlation
    plt.subplot(3, 2, 5)
    plt.scatter(results_df['can_value'], results_df['latency'] * 1000000,
                alpha=0.6, s=20, color='blue')
    plt.xlabel('CAN Value')
    plt.ylabel('Latency (μs)')
    plt.title('Latency vs CAN Value')
    plt.grid(True, alpha=0.3)

    # 6. Rolling statistics
    plt.subplot(3, 2, 6)
    window_size = min(100, len(results_df) // 10)
    if window_size > 1:
        rolling_mean = results_df['latency'].rolling(window=window_size).mean() * 1000000
        rolling_std = results_df['latency'].rolling(window=window_size).std() * 1000000
        plt.plot(results_df['can_timestamp'], rolling_mean, label='Rolling Mean', linewidth=2, color='blue')
        plt.fill_between(results_df['can_timestamp'],
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.3, label='±1σ', color='lightblue')
        plt.xlabel('CAN Timestamp')
        plt.ylabel('Latency (μs)')
        plt.title(f'Rolling Statistics (window={window_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create additional detailed plots
    create_detailed_plots(results_df, output_dir)

def create_detailed_plots(results_df, output_dir):
    """Create additional detailed analysis plots."""

    # 1. Box plot of latency statistics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.boxplot(results_df['latency'] * 1000000, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Latency (μs)')
    plt.title('Latency Box Plot')
    plt.grid(True, alpha=0.3)

    # 2. Cumulative distribution of latency
    plt.subplot(2, 2, 2)
    sorted_latency = np.sort(results_df['latency'] * 1000000)
    cumulative_prob = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
    plt.plot(sorted_latency, cumulative_prob, linewidth=2, color='blue')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Latency')
    plt.grid(True, alpha=0.3)

    # 3. Time series of latency with moving average
    plt.subplot(2, 2, 3)
    plt.plot(results_df['can_timestamp'], results_df['latency'] * 1000000,
             alpha=0.5, label='Raw Latency', color='lightblue')
    window_size = min(50, len(results_df) // 20)
    if window_size > 1:
        moving_avg = results_df['latency'].rolling(window=window_size).mean() * 1000000
        plt.plot(results_df['can_timestamp'], moving_avg,
                linewidth=2, label=f'Moving Average (window={window_size})', color='blue')
    plt.xlabel('CAN Timestamp')
    plt.ylabel('Latency (μs)')
    plt.title('Latency Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Jitter analysis
    plt.subplot(2, 2, 4)
    if 'jitter' in results_df.columns:
        plt.plot(results_df['can_timestamp'][1:], results_df['jitter'][1:] * 1000000,
                alpha=0.6, linewidth=1, color='blue')
        plt.xlabel('CAN Timestamp')
        plt.ylabel('Jitter (μs)')
        plt.title('Jitter Time Series')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(results_df, output_dir="plots"):
    """Print comprehensive statistics about the latency and jitter."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the statistics text
    stats_text = []
    stats_text.append("="*60)
    stats_text.append("LATENCY AND JITTER ANALYSIS RESULTS")
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
    time_span = results_df['can_timestamp'].max() - results_df['can_timestamp'].min()
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
    parser.add_argument('--can-file', required=True, help='Path to CAN CSV file')
    parser.add_argument('--ros-file', required=True, help='Path to ROS CSV file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--max-latency', type=float, default=1.0,
                       help='Maximum acceptable latency threshold in seconds')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    can_df, ros_df = load_data(args.can_file, args.ros_file)

    # Calculate latency and jitter
    results_df = calculate_latency_and_jitter(can_df, ros_df, args.max_latency)

    if len(results_df) == 0:
        print("No valid message pairs found within the latency threshold.")
        return

    # Print statistics
    print_statistics(results_df, args.output_dir)

    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in directory: {args.output_dir}")
        create_plots(results_df, args.output_dir)

    # Save results to CSV
    output_csv = os.path.join(args.output_dir, 'latency_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
