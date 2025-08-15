#!/usr/bin/env python3
"""
Results Comparison Script

This script parses results.txt files from different test configurations
and creates comparison plots of mean CPU and memory usage.
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple


def parse_results_file(file_path: str) -> Tuple[float, float]:
    """
    Parse a results.txt file and extract mean CPU and memory usage.

    Args:
        file_path: Path to the results.txt file

    Returns:
        Tuple of (mean_cpu, mean_memory) in percentage and MiB respectively
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None, None

    mean_cpu = None
    mean_memory = None

    with open(file_path, 'r') as f:
        content = f.read()

        # Extract mean CPU usage
        cpu_match = re.search(r'Mean: (\d+\.\d+)', content)
        if cpu_match:
            mean_cpu = float(cpu_match.group(1))

        # Extract mean memory usage (in MiB)
        # Look for the Memory Usage Statistics section
        memory_section = re.search(r'Memory Usage Statistics \(MiB\):(.*?)Memory Percentage Statistics',
                                  content, re.DOTALL)
        if memory_section:
            memory_match = re.search(r'Mean: (\d+\.\d+)', memory_section.group(1))
            if memory_match:
                mean_memory = float(memory_match.group(1))

    return mean_cpu, mean_memory


def create_comparison_plots(base_path: str):
    """
    Create comparison plots for CPU and memory usage across different configurations.

    Args:
        base_path: Base directory path containing the test results
    """

    # Define test configurations
    node_counts = [1, 5, 10]
    configs = {
        'no_traffic': {
            '1-nodes': f"{base_path}/1-nodes-no-traffic/results.txt",
            '5-nodes': f"{base_path}/5-nodes-no-traffic/results.txt",
            '10-nodes': f"{base_path}/10-nodes-no-traffic/results.txt"
        },
        'with_traffic': {
            '1-nodes': f"{base_path}/1-nodes/results.txt",
            '5-nodes': f"{base_path}/5-nodes/results.txt",
            '10-nodes': f"{base_path}/10-nodes/results.txt"
        }
    }

    # Parse all results
    results = {}
    for traffic_type, configs_dict in configs.items():
        results[traffic_type] = {}
        for node_count, file_path in configs_dict.items():
            mean_cpu, mean_memory = parse_results_file(file_path)
            results[traffic_type][node_count] = {
                'cpu': mean_cpu,
                'memory': mean_memory
            }
            print(f"{traffic_type} - {node_count}: CPU={mean_cpu:.3f}%, Memory={mean_memory:.3f}MiB")

        # Set up plotting style to match measure_resource_usage.py
    plt.style.use('seaborn-v0_8')
    import seaborn as sns
    sns.set_palette("Blues")

    x_pos = np.arange(len(node_counts))
    width = 0.35  # Width of the bars

    # Create CPU usage comparison plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    cpu_no_traffic = [results['no_traffic'][f'{n}-nodes']['cpu'] for n in node_counts]
    cpu_with_traffic = [results['with_traffic'][f'{n}-nodes']['cpu'] for n in node_counts]

    ax1.bar(x_pos - width/2, cpu_no_traffic, width, label='No Traffic', color='blue', alpha=0.8)
    ax1.bar(x_pos + width/2, cpu_with_traffic, width, label='With Traffic', color='green', alpha=0.8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Mean CPU Usage (%)')
    ax1.set_title('CPU Usage Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(node_counts)

    plt.tight_layout()
    cpu_output_file = f"{base_path}/cpu_usage_comparison.png"
    plt.savefig(cpu_output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Create memory usage comparison plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    memory_no_traffic = [results['no_traffic'][f'{n}-nodes']['memory'] for n in node_counts]
    memory_with_traffic = [results['with_traffic'][f'{n}-nodes']['memory'] for n in node_counts]

    ax2.bar(x_pos - width/2, memory_no_traffic, width, label='No Traffic', color='blue', alpha=0.8)
    ax2.bar(x_pos + width/2, memory_with_traffic, width, label='With Traffic', color='green', alpha=0.8)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Mean Memory Usage (MiB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(node_counts)

    plt.tight_layout()
    memory_output_file = f"{base_path}/memory_usage_comparison.png"
    plt.savefig(memory_output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCPU comparison plot saved to: {cpu_output_file}")
    print(f"Memory comparison plot saved to: {memory_output_file}")

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Config':<15} {'CPU (%)':<10} {'Memory (MiB)':<12}")
    print("-" * 60)

    for traffic_type in ['no_traffic', 'with_traffic']:
        traffic_label = "No Traffic" if traffic_type == 'no_traffic' else "With Traffic"
        print(f"\n{traffic_label}:")
        for node_count in node_counts:
            config_key = f'{node_count}-nodes'
            cpu = results[traffic_type][config_key]['cpu']
            memory = results[traffic_type][config_key]['memory']
            print(f"  {config_key:<12} {cpu:<10.3f} {memory:<12.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare resource usage results across different test configurations')
    parser.add_argument('--base-path', '-b',
                        required=True,
                        help='Base directory path containing the test results')

    args = parser.parse_args()
    create_comparison_plots(args.base_path)
