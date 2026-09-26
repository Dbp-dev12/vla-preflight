"""Extended workflows kept separate from the compatible v0.1 audit CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def register(subs):
    profile = subs.add_parser(
        "profile", help="Full numeric profile, duplicates and exploratory lag curves"
    )
    profile.add_argument("dataset", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    prep = subs.add_parser(
        "prepare", help="Create a fingerprinted split and train-only normalization bundle"
    )
    prep.add_argument("dataset", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--validation-fraction", type=float, default=0.2)
    prep.add_argument("--seed", type=int, default=7)
    prep.add_argument("--video", choices=["exists", "probe", "skip"], default="exists")
    export = subs.add_parser(
        "export", help="Copy selected episodes to a new self-contained v3 dataset"
    )
    export.add_argument("dataset", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--exclude", required=True, help="Comma-separated episode IDs")
    learning = subs.add_parser(
        "learning-demo", help="Generate a tiny image/text/action learning dataset"
    )
    learning.add_argument("destination", type=Path)
    learning.add_argument("--episodes", type=int, default=24)
    learning.add_argument("--frames", type=int, default=16)
    learning.add_argument("--seed", type=int, default=7)
    train = subs.add_parser(
        "train", help="Actually train the small reference VLA on a prepared bundle"
    )
    train.add_argument("prepared", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--chunk-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--camera")
    train.add_argument("--max-frames", type=int, default=50000)
    train.add_argument(
        "--resume", type=Path, help="Own checkpoint.pt; steps is total target, not added steps"
    )
    compare = subs.add_parser(
        "compare", help="Compare offline metrics and identify incompatible splits"
    )
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--output", type=Path)
    plan = subs.add_parser(
        "smol-plan", help="Export training split and prepare an explicit SmolVLA launch"
    )
    plan.add_argument("prepared", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--python", help="Python in a separate environment with LeRobot installed")
    plan.add_argument("--pretrained", default="lerobot/smolvla_base")
    plan.add_argument("--steps", type=int, default=1000)
    plan.add_argument("--batch-size", type=int, default=1)
    plan.add_argument("--gradient-accumulation", type=int, default=8)
    plan.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    plan.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    plan.add_argument("--environment-report", type=Path, required=True)
    launch = subs.add_parser(
        "smol-launch", help="Execute a prepared SmolVLA plan (may download weights)"
    )
    launch.add_argument("plan", type=Path)
    lerobot_check = subs.add_parser(
        "lerobot-check", help="Check a separate LeRobot 0.6 environment and requested device"
    )
    lerobot_check.add_argument("--python", required=True)
    lerobot_check.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    lerobot_check.add_argument("--output", type=Path, required=True)
    doctor = subs.add_parser(
        "doctor", help="Inspect local compute/tooling without exposing host identity or paths"
    )
    doctor.add_argument("--output", type=Path, help="Optional JSON report destination")
    robot_check = subs.add_parser(
        "robot-check", help="Run non-actuating serial, camera, calibration and safety checks"
    )
    robot_check.add_argument("config", type=Path)
    robot_check.add_argument("--output", type=Path, required=True)
    robot_check.add_argument("--skip-ports", action="store_true")
    robot_check.add_argument("--skip-cameras", action="store_true")
    robot = subs.add_parser(
        "robot-plan", help="Validate a robot declaration and generate a gated LeRobot runbook"
    )
    robot.add_argument("config", type=Path)
    robot.add_argument("--output", type=Path, required=True)
    robot.add_argument("--preflight", type=Path, required=True)
    rollout = subs.add_parser(
        "rollout-eval", help="Summarize real or simulated rollout JSONL with uncertainty"
    )
    rollout.add_argument("log", type=Path)
    rollout.add_argument("--output", type=Path, required=True)
    rollout_import = subs.add_parser(
        "rollout-import", help="Create a first-class rollout session bound to robot/data/policy"
    )
    rollout_import.add_argument("log", type=Path)
    rollout_import.add_argument("--protocol", type=Path, required=True)
    rollout_import.add_argument("--robot-plan", type=Path, required=True)
    rollout_import.add_argument("--checkpoint", type=Path, required=True)
    rollout_import.add_argument("--dataset-digest", required=True)
    rollout_import.add_argument("--output", type=Path, required=True)
    rollout_compare = subs.add_parser(
        "rollout-compare", help="Compare sessions only when protocol, robot and data match"
    )
    rollout_compare.add_argument("sessions", type=Path, nargs="+")
    rollout_compare.add_argument("--output", type=Path)
    studio = subs.add_parser("studio", help="Local data and training workbench at 127.0.0.1")
    studio.add_argument("dataset", type=Path)
    studio.add_argument("--workspace", type=Path, default=Path("workbench-output"))
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument("--open", action="store_true")


def dispatch(args):
    if args.command in ("audit", "demo"):
        return None
    from .workflow_io import atomic_json

    if args.command == "profile":
        from .analysis import profile_dataset

        if args.output.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("Profile output must be outside source dataset")
        result = profile_dataset(args.dataset)
        atomic_json(args.output, result)
        print(
            f"Profiled {len(result['episodes'])} episodes; "
            f"{len(result['duplicate_groups'])} numeric duplicate groups"
        )
    elif args.command == "prepare":
        from .analysis import prepare

        result = prepare(
            args.dataset,
            args.output,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            video=args.video,
        )
        print(f"Prepared bundle: {args.output}; source SHA256: {result['source']['digest']}")
    elif args.command == "export":
        from .export import export_episodes

        exclude = {int(x) for x in args.exclude.split(",")}
        print(json.dumps(export_episodes(args.dataset, args.output, exclude=exclude), indent=2))
    elif args.command == "learning-demo":
        from .learning_demo import learning_demo

        learning_demo(args.destination, seed=args.seed, episodes=args.episodes, frames=args.frames)
        print(f"Created synthetic image/text/action dataset: {args.destination}")
    elif args.command == "train":
        from .training import TrainOptions, train

        options = TrainOptions(
            **{
                name: getattr(args, name)
                for name in (
                    "steps",
                    "batch_size",
                    "chunk_size",
                    "learning_rate",
                    "seed",
                    "device",
                    "camera",
                    "max_frames",
                )
            }
        )
        result = train(
            args.prepared,
            args.output,
            options=options,
            resume=args.resume,
            progress=lambda item: print(json.dumps(item), flush=True),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        from .training import compare_runs

        result = compare_runs(args.runs)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "smol-plan":
        from .bridge import SmolOptions, create_smol_plan

        values = {
            name: getattr(args, name)
            for name in (
                "pretrained",
                "steps",
                "batch_size",
                "gradient_accumulation",
                "gradient_checkpointing",
                "device",
            )
        }
        if args.python:
            values["python"] = args.python
        print(
            json.dumps(
                create_smol_plan(
                    args.prepared,
                    args.output,
                    options=SmolOptions(**values),
                    environment_report=args.environment_report,
                ),
                indent=2,
            )
        )
    elif args.command == "lerobot-check":
        from .bridge import write_lerobot_environment

        result = write_lerobot_environment(args.python, args.output, device=args.device)
        print(json.dumps(result, indent=2))
        return 0 if result["compatible"] else 1
    elif args.command == "smol-launch":
        from .bridge import launch_smol

        result = launch_smol(args.plan)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1
    elif args.command == "doctor":
        from .hardware import doctor, write_doctor

        result = write_doctor(args.output) if args.output else doctor()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "robot-check":
        from .robot import write_robot_check

        result = write_robot_check(
            args.config,
            args.output,
            probe_ports=not args.skip_ports,
            probe_cameras=not args.skip_cameras,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    elif args.command == "robot-plan":
        from .robot import create_robot_plan

        result = create_robot_plan(args.config, args.output, preflight=args.preflight)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["safe_to_start"] else 1
    elif args.command == "rollout-eval":
        from .rollouts import write_rollout_evaluation

        result = write_rollout_evaluation(args.log, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "rollout-import":
        from .rollouts import create_rollout_session

        result = create_rollout_session(
            args.log,
            args.protocol,
            args.output,
            robot_plan=args.robot_plan,
            checkpoint=args.checkpoint,
            dataset_digest=args.dataset_digest,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["complete"] else 1
    elif args.command == "rollout-compare":
        from .rollouts import compare_rollout_sessions

        result = compare_rollout_sessions(args.sessions)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["comparable"] else 1
    elif args.command == "studio":
        from .studio import serve

        serve(args.dataset, args.workspace, port=args.port, open_browser=args.open)
    return 0
