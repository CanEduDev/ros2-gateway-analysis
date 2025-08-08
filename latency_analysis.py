#!/usr/bin/env python3
"""
Latency and Jitter Analysis Script

This script analyzes the latency and jitter between CAN steering messages
and radio steering topic messages to evaluate system performance.

The script can measure latency in two directions:
- CAN → ROS: Measures time from CAN message sent to ROS message received
- ROS → CAN: Measures time from ROS message sent to CAN message received

Use the --direction argument to specify the measurement direction.

PREREQUISITES:
- Both CSV files must be preprocessed to contain only relevant messages
- Both files must have the same number of messages
- The script will error out if these conditions are not met
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

def calculate_latency_and_jitter(can_df, ros_df, direction='can_to_ros'):
    """
    Calculate latency and jitter between CAN and ROS messages.

    Args:
        can_df: DataFrame with CAN messages
        ros_df: DataFrame with ROS messages
        direction: 'can_to_ros' or 'ros_to_can' to specify measurement direction

    Returns:
        DataFrame with latency analysis results
    """
    print(f"Calculating latency and jitter ({direction})...")

    # Validate that both files have the same length
    if len(can_df) != len(ros_df):
        print("ERROR: File length mismatch!")
        print(f"CAN file has {len(can_df)} messages")
        print(f"ROS file has {len(ros_df)} messages")
        print("Please ensure both log files have been preprocessed to contain only relevant messages")
        print("and that both files have the same number of messages.")
        raise ValueError("File length mismatch - both files must have the same number of messages")

    # Sort both dataframes by timestamp
    can_df = can_df.sort_values('timestamp').reset_index(drop=True)
    ros_df = ros_df.sort_values('timestamp').reset_index(drop=True)

    latencies = []
    can_timestamps = []
    ros_timestamps = []
    can_values = []
    ros_values = []

    # Use paired messages (same index position) since files are preprocessed
    for idx in range(len(can_df)):
        can_row = can_df.iloc[idx]
        ros_row = ros_df.iloc[idx]

        can_time = can_row['timestamp']
        ros_time = ros_row['timestamp']
        can_value = can_row['value']
        ros_value = ros_row['value']

        if direction == 'can_to_ros':
            # Measure time from CAN message to ROS message
            latency = ros_time - can_time
        elif direction == 'ros_to_can':
            # Measure time from ROS message to CAN message
            latency = can_time - ros_time

        latencies.append(latency)
        can_timestamps.append(can_time)
        ros_timestamps.append(ros_time)
        can_values.append(can_value)
        ros_values.append(ros_value)

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

def create_plots(results_df, output_dir="plots", direction='can_to_ros'):
    """Create comprehensive plots for latency and jitter analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Blues")

    # Determine direction labels for plots
    if direction == 'can_to_ros':
        source_label = 'CAN'
        target_label = 'ROS'
        source_timestamp = 'can_timestamp'
    else:  # ros_to_can
        source_label = 'ROS'
        target_label = 'CAN'
        source_timestamp = 'ros_timestamp'

    # 1. Latency over time
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df[source_timestamp], results_df['latency'] * 1000000,
                alpha=0.6, s=20, color='blue')
    plt.xlabel(f'{source_label} Timestamp')
    plt.ylabel('Latency (μs)')
    plt.title(f'{source_label} → {target_label} Latency Over Time')
    plt.grid(True, alpha=0.3)
    latency_time_file = os.path.join(output_dir, f'latency_over_time_{direction}.png')
    plt.savefig(latency_time_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency over time plot saved to: {latency_time_file}")

    # 2. Jitter over time
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        plt.scatter(results_df[source_timestamp][1:], results_df['jitter'][1:] * 1000000,
                    alpha=0.6, s=20, color='blue')
        plt.xlabel(f'{source_label} Timestamp')
        plt.ylabel('Jitter (μs)')
        plt.title(f'{source_label} → {target_label} Jitter Over Time')
        plt.grid(True, alpha=0.3)
        jitter_time_file = os.path.join(output_dir, f'jitter_over_time_{direction}.png')
        plt.savefig(jitter_time_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter over time plot saved to: {jitter_time_file}")

    # 3. Latency histogram
    plt.figure(figsize=(12, 8))
    plt.hist(results_df['latency'] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Frequency')
    plt.title(f'{source_label} → {target_label} Latency Distribution')
    plt.grid(True, alpha=0.3)
    latency_hist_file = os.path.join(output_dir, f'latency_histogram_{direction}.png')
    plt.savefig(latency_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency histogram saved to: {latency_hist_file}")

    # 4. Jitter histogram
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        plt.hist(results_df['jitter'][1:] * 1000000, bins=50, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Jitter (μs)')
        plt.ylabel('Frequency')
        plt.title(f'{source_label} → {target_label} Jitter Distribution')
        plt.grid(True, alpha=0.3)
        jitter_hist_file = os.path.join(output_dir, f'jitter_histogram_{direction}.png')
        plt.savefig(jitter_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter histogram saved to: {jitter_hist_file}")

    # 5. Latency vs Value correlation
    plt.figure(figsize=(12, 8))
    source_value = 'can_value' if direction == 'can_to_ros' else 'ros_value'
    plt.scatter(results_df[source_value], results_df['latency'] * 1000000,
                alpha=0.6, s=20, color='blue')
    plt.xlabel(f'{source_label} Value')
    plt.ylabel('Latency (μs)')
    plt.title(f'Latency vs {source_label} Value')
    plt.grid(True, alpha=0.3)
    latency_vs_value_file = os.path.join(output_dir, f'latency_vs_value_{direction}.png')
    plt.savefig(latency_vs_value_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency vs value plot saved to: {latency_vs_value_file}")

    # 6. Rolling statistics
    window_size = min(100, len(results_df) // 10)
    if window_size > 1:
        plt.figure(figsize=(12, 8))
        rolling_mean = results_df['latency'].rolling(window=window_size).mean() * 1000000
        rolling_std = results_df['latency'].rolling(window=window_size).std() * 1000000
        plt.plot(results_df[source_timestamp], rolling_mean, label='Rolling Mean', linewidth=2, color='blue')
        plt.fill_between(results_df[source_timestamp],
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.3, label='±1σ', color='lightblue')
        plt.xlabel(f'{source_label} Timestamp')
        plt.ylabel('Latency (μs)')
        plt.title(f'{source_label} → {target_label} Rolling Statistics (window={window_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        rolling_stats_file = os.path.join(output_dir, f'rolling_statistics_{direction}.png')
        plt.savefig(rolling_stats_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rolling statistics plot saved to: {rolling_stats_file}")

    # Create additional detailed plots
    create_detailed_plots(results_df, output_dir, direction)

def create_detailed_plots(results_df, output_dir, direction):
    """Create additional detailed analysis plots."""

    # Determine direction labels for plots
    if direction == 'can_to_ros':
        source_label = 'CAN'
        source_timestamp = 'can_timestamp'
    else:  # ros_to_can
        source_label = 'ROS'
        source_timestamp = 'ros_timestamp'

    # 1. Box plot of latency statistics
    plt.figure(figsize=(12, 8))
    plt.boxplot(results_df['latency'] * 1000000, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Latency (μs)')
    plt.title('Latency Box Plot')
    plt.grid(True, alpha=0.3)
    box_plot_file = os.path.join(output_dir, f'latency_boxplot_{direction}.png')
    plt.savefig(box_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency box plot saved to: {box_plot_file}")

    # 2. Cumulative distribution of latency
    plt.figure(figsize=(12, 8))
    sorted_latency = np.sort(results_df['latency'] * 1000000)
    cumulative_prob = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
    plt.plot(sorted_latency, cumulative_prob, linewidth=2, color='blue')
    plt.xlabel('Latency (μs)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Latency')
    plt.grid(True, alpha=0.3)
    cumulative_dist_file = os.path.join(output_dir, f'cumulative_distribution_{direction}.png')
    plt.savefig(cumulative_dist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cumulative distribution plot saved to: {cumulative_dist_file}")

    # 3. Time series of latency with moving average
    plt.figure(figsize=(12, 8))
    plt.plot(results_df[source_timestamp], results_df['latency'] * 1000000,
             alpha=0.5, label='Raw Latency', color='lightblue')
    window_size = min(50, len(results_df) // 20)
    if window_size > 1:
        moving_avg = results_df['latency'].rolling(window=window_size).mean() * 1000000
        plt.plot(results_df[source_timestamp], moving_avg,
                linewidth=2, label=f'Moving Average (window={window_size})', color='blue')
    plt.xlabel(f'{source_label} Timestamp')
    plt.ylabel('Latency (μs)')
    plt.title('Latency Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    time_series_file = os.path.join(output_dir, f'latency_timeseries_{direction}.png')
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency time series plot saved to: {time_series_file}")

    # 4. Jitter analysis
    if 'jitter' in results_df.columns:
        plt.figure(figsize=(12, 8))
        plt.plot(results_df[source_timestamp][1:], results_df['jitter'][1:] * 1000000,
                alpha=0.6, linewidth=1, color='blue')
        plt.xlabel(f'{source_label} Timestamp')
        plt.ylabel('Jitter (μs)')
        plt.title('Jitter Time Series')
        plt.grid(True, alpha=0.3)
        jitter_timeseries_file = os.path.join(output_dir, f'jitter_timeseries_{direction}.png')
        plt.savefig(jitter_timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Jitter time series plot saved to: {jitter_timeseries_file}")

def print_statistics(results_df, output_dir="plots", direction='can_to_ros'):
    """Print comprehensive statistics about the latency and jitter."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine direction labels
    if direction == 'can_to_ros':
        source_label = 'CAN'
        target_label = 'ROS'
        source_timestamp = 'can_timestamp'
    else:  # ros_to_can
        source_label = 'ROS'
        target_label = 'CAN'
        source_timestamp = 'ros_timestamp'

    # Prepare the statistics text
    stats_text = []
    stats_text.append("="*60)
    stats_text.append(f"LATENCY AND JITTER ANALYSIS RESULTS ({source_label} → {target_label})")
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
    results_file = os.path.join(output_dir, f'results_{direction}.txt')
    with open(results_file, 'w') as f:
        f.write('\n'.join(stats_text))

    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze latency and jitter between CAN and ROS messages')
    parser.add_argument('--can-file', required=True, help='Path to CAN CSV file')
    parser.add_argument('--ros-file', required=True, help='Path to ROS CSV file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--direction', choices=['can_to_ros', 'ros_to_can'], default='can_to_ros',
                       help='Direction of latency measurement: can_to_ros (CAN→ROS) or ros_to_can (ROS→CAN)')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    can_df, ros_df = load_data(args.can_file, args.ros_file)

    # Calculate latency and jitter
    results_df = calculate_latency_and_jitter(can_df, ros_df, args.direction)

    # Print statistics
    print_statistics(results_df, args.output_dir, args.direction)

    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in directory: {args.output_dir}")
        create_plots(results_df, args.output_dir, args.direction)

    # Save results to CSV
    output_csv = os.path.join(args.output_dir, f'latency_results_{args.direction}.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
