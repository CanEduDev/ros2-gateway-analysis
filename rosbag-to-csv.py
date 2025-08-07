import argparse
import csv
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_file", type=Path)
    parser.add_argument("--topic", type=str, default=None,
                       help="Topic to filter (default: all topics)")
    args = parser.parse_args()

    # Create output CSV filename based on input file
    output_file = args.mcap_file.with_suffix('.csv')

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "topic", "value"])

        for msg in read_ros2_messages(args.mcap_file):
            log_time = str(msg.log_time_ns)
            formatted_log_time = log_time[:-9] + "." + log_time[-9:]

            # Filter by topic if specified, otherwise include all topics
            if args.topic is None or msg.channel.topic == args.topic:
                writer.writerow([formatted_log_time, msg.channel.topic, msg.ros_msg.data])

    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    main()
