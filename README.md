# ROS2 Gateway application analysis

Repo contains scripts and instructions to generate the figures for the ROS2 paper.

Start the recordings before you start the Rover, to ensure you get as many CAN messages as ROS messages.

## Recording ROS bag
```
ros2 bag record --all
```

## Recording CAN data
```
candump -d -e -x -l can0
```

## Processing CAN data

```
cd ~/rover
source .venv/bin/activate

python rover_py/can-log-decoder.py rover.dbc candump.log candump.csv

grep "STEERING,0x100,STEERING_ANGLE" candump.csv > candump-steering-only.csv

sed s/"0x100,STEERING_ANGLE,"// -i candump-steering-only.csv

Add "timestamp,topic,value" to top of generated file.
```

## Processing ROS bag

```
./rosbag-to-csv.py --topic /rover/radio/steering rosbag.mcap
mv rosbag.csv rosbag-steering-only.csv
```

## Cleaning up logs
Make sure the filtered logs have the same line count, so that the plot generation works properly.

## Generating plots
Measuring latency from CAN message transmission until ROS topic publishing:
```
./latency_analysis.py --can-file candump-steering-only.csv --ros-file rosbag-steering-only.csv --direction can_to_ros
```

Measuring latency from ROS topic publishing until CAN message transmission:
```
./latency_analysis.py --can-file candump-steering-only.csv --ros-file rosbag-steering-only.csv --direction ros_to_can
```
