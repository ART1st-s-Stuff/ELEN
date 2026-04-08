"""Minimal demo for backbone interface wiring (no real inference)."""

from __future__ import annotations

from src.backbone import BackboneConfig, GenerationRequest, PostTrainingSpec, build_backbone


def main() -> None:
    cfg = BackboneConfig(
        model_family="qwen3_5",
        model_id="Qwen/Qwen3.5-0.6B-Instruct",
        backend_type="transformers",
        device="cuda",
        dtype="bfloat16",
        max_tokens=256,
        temperature=0.0,
    )
    backbone = build_backbone(cfg)

    inputs = backbone.build_inputs(
        prompt="给我一个简短导航计划。",
        system_prompt="你是一个机器人规划助手。",
    )
    print("build_inputs output keys:", list(inputs.keys()))

    handle = backbone.prepare_post_training(
        PostTrainingSpec(
            training_type="lora",
            output_dir="models/backbone_lora",
            train_data="datasets/navigation/train.jsonl",
            eval_data="datasets/navigation/val.jsonl",
            extra={"target_modules": ["q_proj", "v_proj"]},
        )
    )
    print("post-training handle:", handle)

    try:
        backbone.generate(GenerationRequest(inputs=inputs))
    except NotImplementedError as exc:
        print("expected:", exc)


if __name__ == "__main__":
    main()
