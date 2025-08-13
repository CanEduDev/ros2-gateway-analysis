#!/usr/bin/env python3
"""
Docker Stats to CSV Converter

This script runs 'docker stats --format table json' and captures the output,
converting it to CSV format with a configurable timeout.
"""

import json
import csv
import subprocess
import signal
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Optional, Dict, Any
import argparse


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def parse_docker_stats_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single line of docker stats JSON output.
    
    Args:
        line: JSON string from docker stats output
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        # Remove any leading/trailing whitespace
        line = line.strip()
        if not line:
            return None
        
        # Remove ANSI escape sequences that Docker stats uses for terminal formatting
        import re
        # Remove common ANSI escape sequences: \x1b[H, \x1b[K, \x1b[J, etc.
        line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
        
        # Try to find JSON content in the line
        # Look for content that starts with { and ends with }
        json_match = re.search(r'\{.*\}', line)
        if json_match:
            json_content = json_match.group(0)
            # Parse JSON
            data = json.loads(json_content)
            return data
        else:
            return None
            
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON line: {e}", file=sys.stderr)
        return None


def run_docker_stats(timeout_seconds: int) -> list:
    """
    Run docker stats command and capture output with timeout.
    
    Args:
        timeout_seconds: Maximum time to run the command in seconds
        
    Returns:
        List of parsed stats dictionaries
        
    Raises:
        TimeoutError: If the command times out
        subprocess.CalledProcessError: If the docker command fails
    """
    stats_data = []
    
    # Set up timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Start docker stats process
        process = subprocess.Popen(
            ['docker', 'stats', '--format', 'json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"Starting docker stats collection for {timeout_seconds} seconds...")
        
        # Read output line by line
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            parsed_data = parse_docker_stats_line(line)
            if parsed_data:
                # Add timestamp
                parsed_data['timestamp'] = datetime.now().isoformat()
                stats_data.append(parsed_data)
                
                # Print progress every 10 records
                if len(stats_data) % 10 == 0:
                    print(f"Collected {len(stats_data)} records...")
        
        # Wait for process to finish
        process.wait()
        
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            raise subprocess.CalledProcessError(
                process.returncode, 
                ['docker', 'stats'], 
                stderr_output
            )
            
    except TimeoutError:
        print(f"\nTimeout reached after {timeout_seconds} seconds")
        # Terminate the process if it's still running
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            process.wait()
        # Don't raise the exception, just return the data collected so far
        return stats_data
    finally:
        # Cancel the alarm
        signal.alarm(0)
    
    return stats_data


def write_to_csv(stats_data: list, output_file: str):
    """
    Write stats data to CSV file.
    
    Args:
        stats_data: List of stats dictionaries
        output_file: Output CSV file path
    """
    if not stats_data:
        print("No data to write to CSV")
        return
    
    # Get fieldnames from the first record
    fieldnames = list(stats_data[0].keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_data)
    
    print(f"Successfully wrote {len(stats_data)} records to {output_file}")


def create_plots(csv_file: str, output_dir: str = "plots"):
    """
    Create plots from the collected Docker stats data.
    
    Args:
        csv_file: Path to the CSV file with stats data
        output_dir: Output directory for plots
    """
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    cpu_file = os.path.join(output_dir, 'cpu_usage_over_time.png')
    plt.savefig(cpu_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CPU usage plot saved to: {cpu_file}")
    
    # 2. Memory usage over time
    plt.figure(figsize=(12, 8))
    # Extract memory usage (convert from format like "1.983GiB / 15.51GiB")
    memory_usage = []
    for mem_str in df['MemUsage']:
        try:
            # Extract the first part (used memory) and convert to MB
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
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True, alpha=0.3)
    memory_file = os.path.join(output_dir, 'memory_usage_over_time.png')
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
    mem_percent_file = os.path.join(output_dir, 'memory_percentage_over_time.png')
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
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_file = os.path.join(output_dir, 'cpu_memory_combined.png')
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined CPU and Memory plot saved to: {combined_file}")


def print_statistics(csv_file: str):
    """
    Print comprehensive statistics about the resource usage.
    
    Args:
        csv_file: Path to the CSV file with stats data
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
    
    print("="*60)
    print("DOCKER STATS MONITORING RESULTS")
    print("="*60)
    
    print(f"\nData Summary:")
    print(f"  Total data points: {len(df)}")
    print(f"  Time span: {df['timestamp'].max() - df['timestamp'].min()}")
    print(f"  Container: {df['Name'].iloc[0] if 'Name' in df.columns else 'Unknown'}")
    
    print(f"\nCPU Usage Statistics (%):")
    print(f"  Mean: {cpu_percent.mean():.3f}")
    print(f"  Median: {cpu_percent.median():.3f}")
    print(f"  Std Dev: {cpu_percent.std():.3f}")
    print(f"  Min: {cpu_percent.min():.3f}")
    print(f"  Max: {cpu_percent.max():.3f}")
    print(f"  95th percentile: {cpu_percent.quantile(0.95):.3f}")
    print(f"  99th percentile: {cpu_percent.quantile(0.99):.3f}")
    
    print(f"\nMemory Usage Statistics (MB):")
    print(f"  Mean: {pd.Series(memory_usage).mean():.3f}")
    print(f"  Median: {pd.Series(memory_usage).median():.3f}")
    print(f"  Std Dev: {pd.Series(memory_usage).std():.3f}")
    print(f"  Min: {pd.Series(memory_usage).min():.3f}")
    print(f"  Max: {pd.Series(memory_usage).max():.3f}")
    print(f"  95th percentile: {pd.Series(memory_usage).quantile(0.95):.3f}")
    print(f"  99th percentile: {pd.Series(memory_usage).quantile(0.99):.3f}")
    
    print(f"\nMemory Percentage Statistics (%):")
    print(f"  Mean: {mem_percent.mean():.3f}")
    print(f"  Median: {mem_percent.median():.3f}")
    print(f"  Std Dev: {mem_percent.std():.3f}")
    print(f"  Min: {mem_percent.min():.3f}")
    print(f"  Max: {mem_percent.max():.3f}")
    print(f"  95th percentile: {mem_percent.quantile(0.95):.3f}")
    print(f"  99th percentile: {mem_percent.quantile(0.99):.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Capture Docker stats and save to CSV with timeout'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=60,
        help='Timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Output directory for CSV and plots (default: output)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = os.path.join(args.output_dir, f"docker_stats_{timestamp}.csv")
    
    try:
        # Run docker stats collection
        stats_data = run_docker_stats(args.timeout)
        
        # Write to CSV
        write_to_csv(stats_data, csv_file)
        
        print(f"Collection completed. Total records: {len(stats_data)}")
        
        # Print statistics
        print_statistics(csv_file)
        
        # Create plots in subfolder
        plots_dir = os.path.join(args.output_dir, "plots")
        create_plots(csv_file, plots_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Docker command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
