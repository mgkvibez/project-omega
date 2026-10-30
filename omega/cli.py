"""kaliauto's cousin behaves: exit 0 on success, 2 when a dependency is missing."""

from __future__ import annotations

import argparse
import json
import sys

from . import SERVICE, __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omega", description="Tiny vision model: train, benchmark, serve.")
    p.add_argument("--version", action="version", version="{} {}".format(SERVICE, __version__))
    sub = p.add_subparsers(dest="cmd")

    info = sub.add_parser("info", help="show settings, predictor and torch availability")
    info.add_argument("--checkpoint", default="")

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--checkpoint", default="")

    bench = sub.add_parser("bench", help="benchmark the predictor")
    bench.add_argument("--checkpoint", default="")
    bench.add_argument("--runs", type=int, default=30)
    bench.add_argument("--batch", type=int, default=1)
    bench.add_argument("--json", action="store_true")

    train = sub.add_parser("train", help="train the tiny CNN on synthetic shapes (needs the ml extra)")
    train.add_argument("--out", default="models/tinycnn.pt")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--meta", default="")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 2

    from .config import Settings
    from .vision.model import load_predictor, torch_available

    if args.cmd == "info":
        settings = Settings.from_env(checkpoint=args.checkpoint)
        predictor, reason = load_predictor(args.checkpoint)
        print(json.dumps({
            "service": settings.service, "version": __version__,
            "predictor": {"source": predictor.source, "model_version": predictor.version,
                          "classes": list(predictor.classes), "reason": reason},
            "torch_available": torch_available(),
            "image_size": settings.image_size, "channels": settings.channels,
            "max_batch": settings.max_batch,
        }, indent=2))
        return 0

    if args.cmd == "serve":
        import uvicorn
        from .serving.app import create_app
        settings = Settings.from_env(checkpoint=args.checkpoint, log_level="INFO")
        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_config=None)
        return 0

    if args.cmd == "bench":
        from .vision.bench import main as bench_main
        argv = ["--checkpoint", args.checkpoint, "--runs", str(args.runs), "--batch", str(args.batch)]
        if args.json:
            argv.append("--json")
        return bench_main(argv)

    if args.cmd == "train":
        if not torch_available():
            print("refused: training needs torch. Install it with: pip install 'omega-vision[ml]'", file=sys.stderr)
            return 2
        from .vision.train import main as train_main
        argv = ["--out", args.out, "--epochs", str(args.epochs)]
        if args.meta:
            argv.extend(["--meta", args.meta])
        return train_main(argv)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
