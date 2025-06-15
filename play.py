import argparse
import time

import gymnasium as gym
from PIL import Image

from carla_gym.envs.task_config import TaskConfig
from carla_gym.wrappers.simple_state_action_wrapper import SimpleStateActionWrapper

# from carla_gym.wrappers.simple_state_action_wrapper_back import SimpleStateActionWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = None  # Initialize env variable
    try:
        terminal_configs = {
            "hero": {
                "entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx",
                "kwargs": {
                    "timeout_steps": 50,
                },
            }
        }

        task_config = TaskConfig(
            map_name=args.map,
            weather="dynamic_1.0",
            num_npc_vehicles=(10, 20),
            num_npc_walkers=(10, 20),
            seed=args.seed,
        )
        env = gym.make(
            "carla_gym:DynamicEnv-v1",
            gpu_id=args.gpu_id,
            nullrhi=args.cpu,
            task_config=task_config,
            terminal_configs=terminal_configs,
            render_mode="rgb_array",
        )
        env = SimpleStateActionWrapper(env)

        env.reset()

        step = 0
        start_time = time.time()
        while True:
            action = env.action_space.sample()
            action[0] = abs(action[0])  # no brake to avoid stationary

            obs, reward, terminated, truncated, info = env.step(action)

            step += 1
            steps_per_second = step / (time.time() - start_time + 1e-6)
            print(
                f"Step {step}: reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}, fps: {steps_per_second:.2f}"
            )

            if step % 1 == 0:
                rendered = env.render()
                Image.fromarray(rendered).save("rendered.png")

            if terminated or truncated:
                break

    except Exception as err:
        print(err)
        import traceback

        traceback.print_exc()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
