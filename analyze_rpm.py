#!/usr/bin/env python3
"""
RPM Analysis Script

This script analyzes the RPM values from CAN wheel RPM messages and ROS wheel status messages
to evaluate system performance and calculate delays between the two message types.

The script plots both RPM values on the same graph and calculates delays from CAN to ROS messages.

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
    can_df = df[df['message'] == 'can_wheel_rpm'].copy()
    ros_df = df[df['message'] == '/rover/wheel_front_left/wheel_status'].copy()

    print(f"CAN messages: {len(can_df)}")
    print(f"ROS messages: {len(ros_df)}")

    # Sort by timestamp
    can_df = can_df.sort_values('timestamp').reset_index(drop=True)
    ros_df = ros_df.sort_values('timestamp').reset_index(drop=True)

    return can_df, ros_df

def calculate_delays(can_df, ros_df):
    """
    Calculate delays between CAN and ROS messages.

    Args:
        can_df: DataFrame with CAN messages
        ros_df: DataFrame with ROS messages

    Returns:
        DataFrame with delay analysis results
    """
    print("Calculating delays (CAN to ROS)...")

    # Find the minimum number of messages to pair
    min_messages = min(len(can_df), len(ros_df))
    print(f"Using {min_messages} message pairs for analysis")

    delays = []
    can_timestamps = []
    ros_timestamps = []
    can_rpm_values = []
    ros_rpm_values = []

    # Pair messages by index (assuming they are in chronological order)
    for idx in range(min_messages):
        can_row = can_df.iloc[idx]
        ros_row = ros_df.iloc[idx]

        can_time = can_row['timestamp']
        ros_time = ros_row['timestamp']
        can_rpm = can_row['rpm']
        ros_rpm = ros_row['rpm']

        # Measure time from CAN message to ROS message
        delay = ros_time - can_time

        delays.append(delay)
        can_timestamps.append(can_time)
        ros_timestamps.append(ros_time)
        can_rpm_values.append(can_rpm)
        ros_rpm_values.append(ros_rpm)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'can_timestamp': can_timestamps,
        'ros_timestamp': ros_timestamps,
        'can_rpm': can_rpm_values,
        'ros_rpm': ros_rpm_values,
        'delay': delays
    })

    print(f"Analyzed {len(results_df)} message pairs")
    return results_df

def create_plots(results_df, output_dir="plots"):
    """Create comprehensive plots for RPM analysis."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_context("notebook", font_scale=1)
    sns.set_palette("Blues")

    # RPM comparison plot
    plt.figure(figsize=(7, 3.5))
    # Calculate elapsed time from the first timestamp
    elapsed_time = results_df['can_timestamp'] - results_df['can_timestamp'].min()

    plt.plot(elapsed_time, results_df['can_rpm'],
             alpha=0.7, linewidth=2, label='CAN Wheel RPM', color='darkblue', linestyle='-')
    plt.plot(elapsed_time, results_df['ros_rpm'],
             alpha=0.7, linewidth=2, label='ROS Wheel Status', color='orange', linestyle='--')

    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('RPM')
    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2)
    plt.grid(True, alpha=0.3)
    rpm_comparison_file = os.path.join(plots_dir, 'rpm_comparison.pdf')
    plt.savefig(rpm_comparison_file, bbox_inches='tight')
    plt.close()
    print(f"RPM comparison plot saved to: {rpm_comparison_file}")

def print_statistics(results_df, output_dir="plots"):
    """Print comprehensive statistics about the RPM data and delays."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the statistics text
    stats_text = []
    stats_text.append("="*60)
    stats_text.append("RPM ANALYSIS RESULTS (CAN to ROS)")
    stats_text.append("="*60)

    delay_us = results_df['delay'] * 1000000
    rpm_diff = results_df['ros_rpm'] - results_df['can_rpm']

    stats_text.append("\nDelay Statistics (μs):")
    stats_text.append(f"  Count: {len(results_df)}")
    stats_text.append(f"  Mean: {delay_us.mean():.3f}")
    stats_text.append(f"  Median: {delay_us.median():.3f}")
    stats_text.append(f"  Std Dev: {delay_us.std():.3f}")
    stats_text.append(f"  Min: {delay_us.min():.3f}")
    stats_text.append(f"  Max: {delay_us.max():.3f}")
    stats_text.append(f"  95th percentile: {delay_us.quantile(0.95):.3f}")
    stats_text.append(f"  99th percentile: {delay_us.quantile(0.99):.3f}")

    stats_text.append("\nRPM Statistics:")
    stats_text.append(f"  CAN RPM - Mean: {results_df['can_rpm'].mean():.3f}")
    stats_text.append(f"  CAN RPM - Min: {results_df['can_rpm'].min():.3f}")
    stats_text.append(f"  CAN RPM - Max: {results_df['can_rpm'].max():.3f}")
    stats_text.append(f"  ROS RPM - Mean: {results_df['ros_rpm'].mean():.3f}")
    stats_text.append(f"  ROS RPM - Min: {results_df['ros_rpm'].min():.3f}")
    stats_text.append(f"  ROS RPM - Max: {results_df['ros_rpm'].max():.3f}")

    stats_text.append("\nRPM Difference Statistics:")
    stats_text.append(f"  Mean difference: {rpm_diff.mean():.3f}")
    stats_text.append(f"  Std Dev difference: {rpm_diff.std():.3f}")
    stats_text.append(f"  Min difference: {rpm_diff.min():.3f}")
    stats_text.append(f"  Max difference: {rpm_diff.max():.3f}")

    # Calculate time span
    time_span = results_df['can_timestamp'].max() - results_df['can_timestamp'].min()
    stats_text.append("\nTime Analysis:")
    stats_text.append(f"  Time span: {time_span:.2f} seconds")
    stats_text.append(f"  Messages per second: {len(results_df) / time_span:.2f}")

    # Delay categories
    low_delay = len(delay_us[delay_us <= 500])  # ≤500μs
    medium_delay = len(delay_us[(delay_us > 500) & (delay_us <= 1000)])  # 500-1000μs
    high_delay = len(delay_us[delay_us > 1000])  # >1000μs

    stats_text.append("\nDelay Categories:")
    stats_text.append(f"  ≤500μs: {low_delay} ({low_delay/len(results_df)*100:.2f}%)")
    stats_text.append(f"  500-1000μs: {medium_delay} ({medium_delay/len(results_df)*100:.2f}%)")
    stats_text.append(f"  >1000μs: {high_delay} ({high_delay/len(results_df)*100:.2f}%)")

    # Print to stdout
    for line in stats_text:
        print(line)

    # Save to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('\n'.join(stats_text))

    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze RPM values and delays between CAN and ROS messages')
    parser.add_argument('--csv-file', required=True, help='Path to the combined CSV file containing CAN and ROS messages')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Load data
    can_df, ros_df = load_data(args.csv_file)

    # Calculate delays
    results_df = calculate_delays(can_df, ros_df)

    # Print statistics
    print_statistics(results_df, args.output_dir)

    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in directory: {args.output_dir}/plots")
        create_plots(results_df, args.output_dir)

    # Save results to CSV
    output_csv = os.path.join(args.output_dir, 'rpm_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
