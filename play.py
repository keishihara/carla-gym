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

    terminal_configs = {
        "hero": {
            "entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx",
            "kwargs": {
                "timeout_steps": 3000,
            },
        }
    }
    task_config = TaskConfig(
        map_name=args.map,
        weather="ClearNoon",
        num_npc_vehicles=0,
        num_npc_walkers=0,
        seed=args.seed,
        route_file="/home/keishi_ishihara/workspace/carla-gym/packages/carla_garage/leaderboard/data/bench2drive220_1833.xml",
        route_id="1833",
    )

    env = None
    try:
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
            # action[1] = 0.0  # go straight

            obs, reward, terminated, truncated, info = env.step(action)

            step += 1
            steps_per_second = step / (time.time() - start_time + 1e-6)
            print(
                f"Step {step}: reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}, fps: {steps_per_second:.2f}"
            )

            if step % 5 == 0:
                rendered = env.render()
                Image.fromarray(rendered).save("rendered.png")
                if "front_rgb" in obs:
                    Image.fromarray(obs["front_rgb"]).save("front_rgb.png")

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
