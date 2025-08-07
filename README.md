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
python rosbag-to-csv.py --topic /rover/radio/steering rosbag.mcap
mv rosbag.csv rosbag-steering-only.csv
```

## Generating plots
```
python latency_analysis.py --can-file candump-steering-only.csv --ros-file rosbag-steering-only.csv --output-dir plots
```
