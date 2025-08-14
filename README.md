# ROS2 Gateway application analysis

Repo contains scripts and instructions to generate the figures for the ROS2 paper.

## Setting up docker container for recording

Start rover-ros-gateway through run-ros-gateway script. Then:

```
docker exec -it --user root rover-ros-gateway bash

# in container:
apt update
apt install -y ros-jazzy-rosbag2 can-utils
su ubuntu
source install/setup.bash
```

## Recording data
For CAN-to-ROS recordings, enable radio safety override. For ROS-to-CAN, disable it.

Use timeout command to specify duration of recording.

CAN-to-ROS:
```
timeout 60 bash -c "candump -d -e -x -l can0 & ros2 bag record --topics /rover/radio/cmd_vel"
```

TODO: think about this some more. You're not really measuring latency because the controller runs at a fixed frequency. Maybe measure jitter between CAN messages instead to see how accurate it is.
ROS-to-CAN:
```
timeout 60 bash -c "candump -d -e -x -l can0 & ros2 bag record --topics /rover/cmd_vel"
```

Copy the data from the container to your filesystem. Adjust filenames to be correct.
```
docker cp rover-ros-gateway:/ros2_ws/candump.log .
docker cp rover-ros-gateway:/ros2_ws/rosbag2/rosbag2.mcap .
```

## Convert logs to CSV
```
./logs_to_csv.py --can-log candump.log --ros-log rosbag2.mcap
```
## Generating plots
Measuring latency from CAN message transmission until ROS topic publishing:
```
./analyze_latency.py --csv-file filtered-logs.csv --output-dir output
```
