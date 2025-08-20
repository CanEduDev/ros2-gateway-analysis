#!/usr/bin/env python3
import struct
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
        csv_data.append((formatted_log_time, msg.channel.topic, msg.ros_msg.rpm))

    # Parse CAN log
    with open(args.can_log, 'r') as can_log:
        for line in can_log:
            entries = line.split(' ')
            if entries[2].startswith('210'):
                message_name = "can_wheel_rpm"
                timestamp = entries[0].lstrip('(').rstrip(')')
                rpm_raw = entries[2].split('#')[1][:4*2] # Extract first 8 hex digits, this is rpm
                rpm = struct.unpack('f', bytes.fromhex(rpm_raw))[0]
                csv_data.append((timestamp, message_name, rpm))

    csv_data.sort()

    # Remove all messages before first wheel_status
    for i, line in enumerate(csv_data):
        if "wheel_status" in line[1]:
            # Include preceding wheel message
            csv_data = csv_data[i-1:]
            break

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "message", "rpm"])
        for line in csv_data:
            writer.writerow(line)

    print(f"wrote file: {output_file}")


if __name__ == "__main__":
    main()
