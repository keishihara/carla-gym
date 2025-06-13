#!/bin/bash

set -e # exit on error

PROJECT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$SECONDS
PPM=4.0
SAVE_DIR="$PROJECT_ROOT/carla_gym/assets/maps/maps_{ppm}ppm"

source $PROJECT_ROOT/.venv/bin/activate

python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town01 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town02 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town03 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town04 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town05 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town06 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town07 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town10HD --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town12 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town13 --pixels_per_meter $PPM --save_dir $SAVE_DIR
python "$PROJECT_ROOT/carla_gym/tools/create_h5_map.py" --town Town15 --pixels_per_meter $PPM --save_dir $SAVE_DIR

echo "Done generating maps with pixels_per_meter=$PPM."
TIME_TAKEN=$(((SECONDS - START_TIME) / 60))
echo "Time taken so far: $TIME_TAKEN minutes"
