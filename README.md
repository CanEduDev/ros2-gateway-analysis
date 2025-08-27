# ROS2 Gateway application analysis

Repo contains scripts and instructions to generate the figures for the ROS2
paper. Everything tested with v0.13.0 in the Rover repo.

## CAN to ROS latency analysis

### Recording data
Start car and enable radio safety override.

Start rover-ros-gateway through run-ros-gateway script. Then:

```
docker exec -it --user root rover-ros-gateway bash

# in container:
apt update
apt install -y ros-jazzy-rosbag2 can-utils
su ubuntu
source install/setup.bash
```

Use timeout command to specify duration of recording.
```
timeout 60 bash -c "candump -d -e -x -l can0 & ros2 bag record --topics /rover/radio/cmd_vel"
```

Copy the data from the container to your filesystem. Adjust filenames to be correct.
```
docker cp rover-ros-gateway:/ros2_ws/candump.log .
docker cp rover-ros-gateway:/ros2_ws/rosbag2/rosbag2.mcap .
```

### Convert logs to CSV
```
./latency_logs_to_csv.py --can-log candump.log --ros-log rosbag2.mcap
```

### Generating plots
```
./analyze_latency.py --csv-file filtered-logs.csv --output-dir output
```

## ROS CAN transmission jitter analysis

Start car and enable radio safety override.

Start rover-ros-gateway through run-ros-gateway script.

Disable safety override.

Log CAN data:
```
timeout 60 candump -d -e -x -l can0
```

### Generating plots
```
./analyze_ros_jitter.py --log-file candump.log --output-dir output
```

## Resource usage measurements

Use different launch file configurations for 1, 5, 10 nodes in the ros-gateway. Rebuild it for each.

Start the container, but keep the car off. Run a test without traffic:
```
./record_resource_usage.py -o 1-nodes-no-traffic
```
Then start the car and run the test with CAN traffic:
```
./record_resource_usage.py -o 1-nodes
```

Repeat for different launch configurations.

Run analysis:
```
./analyze_resource_usage.py -i 1-nodes-no-traffic/resource-usage.csv -o 1-nodes-no-traffic
./analyze_resource_usage.py -i 1-nodes/resource-usage.csv -o 1-nodes
```

When performed for all 6 variants, run:
```
./compare_resource_usage_results.py -b .
```

## Wheel RPM measurements

### Recording data

Start rover-ros-gateway through run-ros-gateway script. Then:

```
docker exec -it --user root rover-ros-gateway bash

# in container:
apt update
apt install -y ros-jazzy-rosbag2 can-utils
su ubuntu
source install/setup.bash
```

Use timeout command to specify duration of recording.
During recording, move the wheels using the radio, to get values other than 0.
```
timeout 60 bash -c "candump -d -e -x -l can0 & ros2 bag record --topics /rover/wheel_front_left/wheel_status"
```


Copy the data from the container to your filesystem. Adjust filenames to be correct.
```
docker cp rover-ros-gateway:/ros2_ws/candump.log .
docker cp rover-ros-gateway:/ros2_ws/rosbag2/rosbag2.mcap .
```

### Convert logs to CSV
```
./wheel_logs_to_csv.py --can-log candump.log --ros-log rosbag2.mcap
```

### Generating plots
```
./analyze_rpm.py --csv-file filtered-logs.csv --output-dir output
```
