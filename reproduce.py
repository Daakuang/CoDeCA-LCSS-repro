"""Start here: redraw the paper, or explicitly request new optimization."""

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("figure", "solve"), nargs="?", default="figure")
    parser.add_argument("--source", type=Path, default=root / "data/reference.json",
                        help="saved JSON for the figure command")
    parser.add_argument("--output", type=Path, default=root / "results", help="generated files and solver checkpoints")
    parser.add_argument("--section", choices=("all", "codesign", "qi", "canonical", "rfd", "sensitivity"),
                        default="all", help="solve command only")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == root or (root / "data") == output or (root / "data") in output.parents:
        parser.error("Choose an output folder outside the source files and data/")
    if args.command == "solve":
        from experiments import run
        source = run(output, args.section)
        if args.section != "all":
            print(f"Saved {args.section}: {source}")
            return
    else:
        source = args.source
    from plot_results import export
    export(source, output)


if __name__ == "__main__":
    main()
