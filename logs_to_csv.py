#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-log", type=Path, required=True)
    parser.add_argument("--ros-log", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.can_log.parent
    output_file = output_dir / 'filtered-logs.csv'

    csv_data = list()

    # Parse ROS log
    for msg in read_ros2_messages(args.ros_log):
        log_time = str(msg.publish_time_ns)
        formatted_log_time = log_time[:-9] + "." + log_time[-9:]
        csv_data.append((formatted_log_time, msg.channel.topic))

    # Parse CAN log
    with open(args.can_log, 'r') as can_log:
        for line in can_log:
            entries = line.split(' ')
            if entries[2].startswith('101'):
                message_name = "can_throttle"
                timestamp = entries[0].lstrip('(').rstrip(')')
                csv_data.append((timestamp, message_name))

    csv_data.sort()

    # Remove all messages before first cmd_vel
    for i, line in enumerate(csv_data):
        if "cmd_vel" in line[1]:
            # Include preceding throttle message
            csv_data = csv_data[i-1:]
            break

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "message"])
        for line in csv_data:
            writer.writerow(line)

    print(f"wrote file: {output_file}")


if __name__ == "__main__":
    main()
