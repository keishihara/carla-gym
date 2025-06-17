#!/bin/bash


python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools
pip install -e .
cd packages/carla_garage/leaderboard_autopilot
pip install -e .
cd ../scenario_runner_autopilot
pip install -e .
cd ../../PythonAPI/carla
pip install -e .
