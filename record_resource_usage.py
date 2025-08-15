#!/usr/bin/env python3
"""
Docker Stats Recorder

This script runs 'docker stats --format table json' and captures the output,
converting it to CSV format with a configurable timeout.
"""

import json
import csv
import subprocess
import signal
import sys
import time
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
        required=True,
        help='Output directory for CSV file (required)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate CSV filename with timestamp
    csv_file = os.path.join(args.output_dir, "resource-usage.csv")

    try:
        # Run docker stats collection
        stats_data = run_docker_stats(args.timeout)

        # Write to CSV
        write_to_csv(stats_data, csv_file)

        print(f"Collection completed. Total records: {len(stats_data)}")
        print(f"CSV file saved to: {csv_file}")

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
