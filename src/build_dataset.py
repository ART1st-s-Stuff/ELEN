from __future__ import annotations

import argparse
import asyncio
import os
import random
from dataclasses import asdict
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import ACTION_TOKEN_MAP, FEATURE_MODE_CHOICES, VALID_ACTIONS, SplitRatio
from src.features import FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build navigation transition dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset root directory")
    parser.add_argument("--image_dir", type=str, default="", help="Directory for saved images")
    parser.add_argument("--episodes", type=int, default=100, help="Number of rollout episodes")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval_set", type=str, default="base", help="Navigation eval_set")
    parser.add_argument("--prompt_format", type=str, default="latent_plan", help="Prompt format")
    parser.add_argument("--gpu_device", type=int, default=0, help="AI2-THOR gpu device id")
    parser.add_argument("--feature_mode", type=str, choices=FEATURE_MODE_CHOICES, default="both")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--mae_model_name", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--feature_device", type=str, default="cpu", help="Feature extraction device")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def format_action_response(action_name: str, prompt_format: str) -> str:
    if prompt_format == "latent_plan":
        return f"<|action_start|>{ACTION_TOKEN_MAP[action_name]}<|action_end|>"
    return f"<think>random explore</think><action>{action_name}</action>"


def get_obs_image(obs: dict[str, Any]):
    images = obs.get("multi_modal_input", {}).get("<image>", []) or []
    return images[0] if images else None


def save_obs_image(obs: dict[str, Any], path: str) -> str | None:
    image = get_obs_image(obs)
    if image is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)
    return path


async def generate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vagen.envs.navigation.navigation_env import NavigationEnv, NavigationEnvConfig

    random.seed(args.seed)
    env_cfg = asdict(
        NavigationEnvConfig(
            eval_set=args.eval_set,
            prompt_format=args.prompt_format,
            max_steps=args.max_steps,
            gpu_device=args.gpu_device,
        )
    )
    env = NavigationEnv(env_cfg)

    extractor = FeatureExtractor.build(
        feature_mode=args.feature_mode,
        clip_model_name=args.clip_model_name,
        mae_model_name=args.mae_model_name,
        device=args.feature_device,
    )

    img_dir = args.image_dir or os.path.join(args.output_dir, "images")
    rows: list[dict[str, Any]] = []
    try:
        pbar = tqdm(range(args.episodes), desc="Generating transition samples", unit="ep")
        for ep in pbar:
            obs, _ = await env.reset(seed=args.seed + ep)
            done = False
            step_id = 0
            while not done and step_id < args.max_steps:
                action_name = random.choice(VALID_ACTIONS)
                action_id = VALID_ACTIONS.index(action_name)

                image_path = save_obs_image(obs, os.path.join(img_dir, f"ep{ep:06d}_step{step_id:04d}.png"))
                img = get_obs_image(obs)

                response = format_action_response(action_name=action_name, prompt_format=args.prompt_format)
                next_obs, reward, done, info = await env.step(response)

                next_image_path = save_obs_image(
                    next_obs, os.path.join(img_dir, f"ep{ep:06d}_step{step_id:04d}_next.png")
                )
                next_img = get_obs_image(next_obs)

                if img is not None and next_img is not None:
                    img_feat = extractor.extract(img)
                    next_img_feat = extractor.extract(next_img)
                    rows.append(
                        {
                            "image_path": image_path,
                            "action_name": action_name,
                            "action_id": action_id,
                            "next_image_path": next_image_path,
                            "img_feat": img_feat,
                            "next_img_feat": next_img_feat,
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(info.get("success", False)),
                            "episode_id": ep,
                            "step_id": step_id,
                            "feature_mode": args.feature_mode,
                        }
                    )

                obs = next_obs
                step_id += 1
            pbar.set_postfix(samples=len(rows))
    finally:
        await env.close()
    return rows


def split_and_save(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    split_ratio = SplitRatio(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    split_ratio.validate()

    os.makedirs(args.output_dir, exist_ok=True)
    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * split_ratio.train)
    n_val = int(n * split_ratio.val)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(args.output_dir, "val.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)

    print(f"Saved train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} to {args.output_dir}")


async def _main() -> None:
    args = parse_args()
    rows = await generate_rows(args)
    split_and_save(rows=rows, args=args)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

