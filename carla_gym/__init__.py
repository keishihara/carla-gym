from pathlib import Path

from gymnasium.envs.registration import register

CARLA_GYM_ROOT_DIR = Path(__file__).resolve().parent

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    "DynamicEnv-v1": {
        "entry_point": "carla_gym.envs:DynamicEnv",
        "description": "Dynamic env for RL training and testing",
    },
}

for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"), kwargs=val.get("kwargs"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint

    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)
