"""Command-line interface with shared --config across subcommands."""
import argparse

from .utils.logging import get_logger

log = get_logger("terraforge.cli")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to YAML/JSON config")
    common.add_argument("--backend", default=None, help="backend override (mock/cpu/openai/vllm/transformers)")
    common.add_argument("--seed", type=int, default=None, help="random seed")
    common.add_argument("--verbose", action="store_true")

    parser = argparse.ArgumentParser(prog="terraforge", description="spatial multimodal world model framework")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", parents=[common], help="environment self-check")
    sub.add_parser("data", parents=[common], help="run the spatial-multimodal data production pipeline")
    sub.add_parser("train", parents=[common], help="pretrain on synthetic spatial scenes")
    sub.add_parser("sft", parents=[common], help="supervised fine-tune on spatial QA")
    sub.add_parser("bench", parents=[common], help="run the 9-benchmark spatial understanding suite")
    sub.add_parser("serve", parents=[common], help="start the stdlib HTTP server")
    sp = sub.add_parser("predict", parents=[common], help="predict on a single spatial query")
    sp.add_argument("--scene", default="scene-001", help="scene id")
    sp.add_argument("--query", default="What is left of the table?", help="spatial query")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .config import load_config

    overrides = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, **overrides)

    if args.command == "doctor":
        from .doctor import print_doctor
        print_doctor()
        return 0
    if args.command == "data":
        from .data.pipeline import run_pipeline
        run_pipeline(cfg)
        return 0
    if args.command == "train":
        from .training.pretrain import run_pretrain
        run_pretrain(cfg)
        return 0
    if args.command == "sft":
        from .training.sft import run_sft
        run_sft(cfg)
        return 0
    if args.command == "bench":
        from .benchmark.suite import run_benchmark
        run_benchmark(cfg)
        return 0
    if args.command == "serve":
        from .serving.http import serve
        serve(cfg)
        return 0
    if args.command == "predict":
        from .backends import build_backend
        be = build_backend(cfg.backend, cfg)
        ans = be.predict_spatial(args.scene, args.query)
        print(f"scene={args.scene}")
        print(f"query={args.query}")
        print(f"answer={ans}")
        return 0
    parser = build_parser()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
