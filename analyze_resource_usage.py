#!/usr/bin/env python3
"""
Resource Usage Analyzer

This script analyzes resource usage CSV data and generates plots and statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys


def create_plots(csv_file: str, output_dir: str):
    """
    Create plots from the collected Docker stats data.

    Args:
        csv_file: Path to the CSV file with stats data
        output_dir: Output directory for plots
    """
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert relative time to seconds from start
    df['relative_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Blues")

    # 1. CPU usage over time
    plt.figure(figsize=(12, 8))
    # Extract CPU percentage (remove % sign and convert to float)
    cpu_percent = df['CPUPerc'].str.rstrip('%').astype(float)
    plt.plot(df['relative_time'], cpu_percent, linewidth=2, color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.grid(True, alpha=0.3)
    cpu_file = os.path.join(plots_dir, 'cpu_usage_over_time.png')
    plt.savefig(cpu_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CPU usage plot saved to: {cpu_file}")

    # 2. Memory usage over time
    plt.figure(figsize=(12, 8))
    # Extract memory usage (convert from format like "1.983GiB / 15.51GiB")
    memory_usage = []
    for mem_str in df['MemUsage']:
        try:
            # Extract the first part (used memory) and convert to MiB
            used_mem = mem_str.split('/')[0].strip()
            if 'GiB' in used_mem:
                mem_mb = float(used_mem.replace('GiB', '')) * 1024
            elif 'MiB' in used_mem:
                mem_mb = float(used_mem.replace('MiB', ''))
            elif 'KiB' in used_mem:
                mem_mb = float(used_mem.replace('KiB', '')) / 1024
            else:
                mem_mb = 0
            memory_usage.append(mem_mb)
        except:
            memory_usage.append(0)

    plt.plot(df['relative_time'], memory_usage, linewidth=2, color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True, alpha=0.3)
    memory_file = os.path.join(plots_dir, 'memory_usage_over_time.png')
    plt.savefig(memory_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Memory usage plot saved to: {memory_file}")

    # 3. Memory percentage over time
    plt.figure(figsize=(12, 8))
    # Extract memory percentage (remove % sign and convert to float)
    mem_percent = df['MemPerc'].str.rstrip('%').astype(float)
    plt.plot(df['relative_time'], mem_percent, linewidth=2, color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (%)')
    plt.title('Memory Usage Percentage Over Time')
    plt.grid(True, alpha=0.3)
    mem_percent_file = os.path.join(plots_dir, 'memory_percentage_over_time.png')
    plt.savefig(mem_percent_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Memory percentage plot saved to: {mem_percent_file}")

    # 4. Combined CPU and Memory plot
    plt.figure(figsize=(14, 10))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # CPU subplot
    ax1.plot(df['relative_time'], cpu_percent, linewidth=2, color='blue')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('CPU Usage Over Time')
    ax1.grid(True, alpha=0.3)

    # Memory subplot
    ax2.plot(df['relative_time'], memory_usage, linewidth=2, color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Memory Usage (MiB)')
    ax2.set_title('Memory Usage Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_file = os.path.join(plots_dir, 'cpu_memory_combined.png')
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined CPU and Memory plot saved to: {combined_file}")


def print_statistics(csv_file: str, output_dir: str):
    """
    Print comprehensive statistics about the resource usage.

    Args:
        csv_file: Path to the CSV file with stats data
        output_dir: Output directory for results.txt file
    """
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract CPU percentage
    cpu_percent = df['CPUPerc'].str.rstrip('%').astype(float)

    # Extract memory usage
    memory_usage = []
    for mem_str in df['MemUsage']:
        try:
            used_mem = mem_str.split('/')[0].strip()
            if 'GiB' in used_mem:
                mem_mb = float(used_mem.replace('GiB', '')) * 1024
            elif 'MiB' in used_mem:
                mem_mb = float(used_mem.replace('MiB', ''))
            elif 'KiB' in used_mem:
                mem_mb = float(used_mem.replace('KiB', '')) / 1024
            else:
                mem_mb = 0
            memory_usage.append(mem_mb)
        except:
            memory_usage.append(0)

    # Extract memory percentage
    mem_percent = df['MemPerc'].str.rstrip('%').astype(float)

    # Prepare the statistics output
    output_lines = []
    output_lines.append("="*60)
    output_lines.append("DOCKER STATS MONITORING RESULTS")
    output_lines.append("="*60)

    output_lines.append(f"\nData Summary:")
    output_lines.append(f"  Total data points: {len(df)}")
    output_lines.append(f"  Time span: {df['timestamp'].max() - df['timestamp'].min()}")
    output_lines.append(f"  Container: {df['Name'].iloc[0] if 'Name' in df.columns else 'Unknown'}")

    output_lines.append(f"\nCPU Usage Statistics (%):")
    output_lines.append(f"  Mean: {cpu_percent.mean():.3f}")
    output_lines.append(f"  Median: {cpu_percent.median():.3f}")
    output_lines.append(f"  Std Dev: {cpu_percent.std():.3f}")
    output_lines.append(f"  Min: {cpu_percent.min():.3f}")
    output_lines.append(f"  Max: {cpu_percent.max():.3f}")
    output_lines.append(f"  95th percentile: {cpu_percent.quantile(0.95):.3f}")
    output_lines.append(f"  99th percentile: {cpu_percent.quantile(0.99):.3f}")

    output_lines.append(f"\nMemory Usage Statistics (MiB):")
    output_lines.append(f"  Mean: {pd.Series(memory_usage).mean():.3f}")
    output_lines.append(f"  Median: {pd.Series(memory_usage).median():.3f}")
    output_lines.append(f"  Std Dev: {pd.Series(memory_usage).std():.3f}")
    output_lines.append(f"  Min: {pd.Series(memory_usage).min():.3f}")
    output_lines.append(f"  Max: {pd.Series(memory_usage).max():.3f}")
    output_lines.append(f"  95th percentile: {pd.Series(memory_usage).quantile(0.95):.3f}")
    output_lines.append(f"  99th percentile: {pd.Series(memory_usage).quantile(0.99):.3f}")

    output_lines.append(f"\nMemory Percentage Statistics (%):")
    output_lines.append(f"  Mean: {mem_percent.mean():.3f}")
    output_lines.append(f"  Median: {mem_percent.median():.3f}")
    output_lines.append(f"  Std Dev: {mem_percent.std():.3f}")
    output_lines.append(f"  Min: {mem_percent.min():.3f}")
    output_lines.append(f"  Max: {mem_percent.max():.3f}")
    output_lines.append(f"  95th percentile: {mem_percent.quantile(0.95):.3f}")
    output_lines.append(f"  99th percentile: {mem_percent.quantile(0.99):.3f}")

    # Output to stdout
    for line in output_lines:
        print(line)

    # Output to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f"\nStatistics also saved to: {results_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze resource usage CSV data and generate plots and statistics'
    )
    parser.add_argument(
        '--input-csv', '-i',
        type=str,
        required=True,
        help='Input CSV file with resource usage data (required)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for plots and results (required)'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file not found: {args.input_csv}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Print statistics
        print_statistics(args.input_csv, args.output_dir)

        # Create plots
        create_plots(args.input_csv, args.output_dir)

        print(f"\nAnalysis completed. Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
