#!/bin/bash

# ROS2 Gateway Analysis - Run All Script
# This script runs all analysis scripts on the recordings folder

set -e  # Exit on any error

echo "=========================================="
echo "ROS2 Gateway Analysis - Running All Scripts"
echo "=========================================="

# Base directory
RECORDINGS_DIR="recordings"

# Check if recordings directory exists
if [ ! -d "$RECORDINGS_DIR" ]; then
    echo "Error: Recordings directory not found: $RECORDINGS_DIR"
    exit 1
fi

echo "Found recordings directory: $RECORDINGS_DIR"
echo ""

# 1. Analyze CAN to ROS Latency
echo "1. Analyzing CAN to ROS Latency..."

# Check if filtered logs exist, if not generate them from raw logs
if [ ! -f "$RECORDINGS_DIR/can-to-ros-latency/filtered-logs.csv" ]; then
    if [ -f "$RECORDINGS_DIR/can-to-ros-latency/candump-can-to-ros.log" ] && [ -f "$RECORDINGS_DIR/can-to-ros-latency/rosbag-can-to-ros.mcap" ]; then
        echo "  Generating filtered logs from raw CAN and ROS logs..."
        uv run latency_logs_to_csv.py \
            --can-log "$RECORDINGS_DIR/can-to-ros-latency/candump-can-to-ros.log" \
            --ros-log "$RECORDINGS_DIR/can-to-ros-latency/rosbag-can-to-ros.mcap"
        echo "  ✓ Filtered logs generated"
    else
        echo "⚠ Skipping CAN to ROS latency analysis - required raw logs not found"
        echo "  Need: candump-can-to-ros.log and rosbag-can-to-ros.mcap"
    fi
fi

if [ -f "$RECORDINGS_DIR/can-to-ros-latency/filtered-logs.csv" ]; then
    uv run analyze_latency.py \
        --csv-file "$RECORDINGS_DIR/can-to-ros-latency/filtered-logs.csv" \
        --output-dir "$RECORDINGS_DIR/can-to-ros-latency/analysis"
    echo "✓ CAN to ROS latency analysis completed"
else
    echo "⚠ Skipping CAN to ROS latency analysis - filtered-logs.csv not available"
fi
echo ""

# 2. Analyze Wheel RPM
echo "2. Analyzing Wheel RPM..."

# Check if filtered logs exist, if not generate them from raw logs
if [ ! -f "$RECORDINGS_DIR/wheel-rpm/filtered-logs.csv" ]; then
    if [ -f "$RECORDINGS_DIR/wheel-rpm/candump.log" ] && [ -f "$RECORDINGS_DIR/wheel-rpm/rosbag2.mcap" ]; then
        echo "  Generating filtered logs from raw CAN and ROS logs..."
        uv run wheel_logs_to_csv.py \
            --can-log "$RECORDINGS_DIR/wheel-rpm/candump.log" \
            --ros-log "$RECORDINGS_DIR/wheel-rpm/rosbag2.mcap"
        echo "  ✓ Filtered logs generated"
    else
        echo "⚠ Skipping Wheel RPM analysis - required raw logs not found"
        echo "  Need: candump.log and rosbag2.mcap"
    fi
fi

if [ -f "$RECORDINGS_DIR/wheel-rpm/filtered-logs.csv" ]; then
    uv run analyze_rpm.py \
        --csv-file "$RECORDINGS_DIR/wheel-rpm/filtered-logs.csv" \
        --output-dir "$RECORDINGS_DIR/wheel-rpm/analysis"
    echo "✓ Wheel RPM analysis completed"
else
    echo "⚠ Skipping Wheel RPM analysis - filtered-logs.csv not available"
fi
echo ""

# 3. Analyze ROS Jitter
echo "3. Analyzing ROS Jitter..."
if [ -f "$RECORDINGS_DIR/ros-jitter/candump-ros-jitter.log" ]; then
    uv run analyze_ros_jitter.py \
        --log-file "$RECORDINGS_DIR/ros-jitter/candump-ros-jitter.log" \
        --output-dir "$RECORDINGS_DIR/ros-jitter/analysis"
    echo "✓ ROS jitter analysis completed"
else
    echo "⚠ Skipping ROS jitter analysis - candump-ros-jitter.log not found"
fi
echo ""

# 4. Analyze Resource Usage for each node configuration
echo "4. Analyzing Resource Usage..."

# Function to analyze resource usage for a specific configuration
analyze_resource_usage() {
    local config_dir="$1"
    local csv_file="$config_dir/resource-usage.csv"
    local output_dir="$config_dir"
    
    if [ -f "$csv_file" ]; then
        echo "  Analyzing $config_dir..."
        uv run analyze_resource_usage.py \
            --input-csv "$csv_file" \
            --output-dir "$output_dir"
        echo "  ✓ $config_dir analysis completed"
    else
        echo "  ⚠ Skipping $config_dir - resource-usage.csv not found"
    fi
}

# Analyze each node configuration
analyze_resource_usage "$RECORDINGS_DIR/1-nodes"
analyze_resource_usage "$RECORDINGS_DIR/1-nodes-no-traffic"
analyze_resource_usage "$RECORDINGS_DIR/5-nodes"
analyze_resource_usage "$RECORDINGS_DIR/5-nodes-no-traffic"
analyze_resource_usage "$RECORDINGS_DIR/10-nodes"
analyze_resource_usage "$RECORDINGS_DIR/10-nodes-no-traffic"

echo ""

# 5. Compare Resource Usage Results
echo "5. Comparing Resource Usage Results..."
uv run compare_resource_usage_results.py \
    --base-path "$RECORDINGS_DIR"
echo "✓ Resource usage comparison completed"
echo ""

echo "=========================================="
echo "All analyses completed successfully!"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "- CAN to ROS latency: $RECORDINGS_DIR/can-to-ros-latency/analysis/"
echo "- Wheel RPM analysis: $RECORDINGS_DIR/wheel-rpm/analysis/"
echo "- ROS jitter analysis: $RECORDINGS_DIR/ros-jitter/analysis/"
echo "- Resource usage analysis: $RECORDINGS_DIR/*/plots/ and results.txt"
echo "- Resource usage comparison: $RECORDINGS_DIR/resource_usage_comparison.pdf"
echo ""
echo "Check the individual analysis directories for detailed results and plots."
