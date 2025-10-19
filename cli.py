import argparse
import os
from typing import Optional

from src.rag_summarizer.model import train_model, summarize_file


def cmd_train(args: argparse.Namespace) -> None:
    arts = train_model(args.docs, args.targets, args.out, alpha=args.alpha)
    print(f"Saved model to: {arts}")


def cmd_summarize(args: argparse.Namespace) -> None:
    inp = args.input
    model_dir = args.model
    if os.path.isdir(inp):
        for name in sorted(os.listdir(inp)):
            if not name.lower().endswith('.txt'):
                continue
            p = os.path.join(inp, name)
            summary = summarize_file(p, model_dir, words=args.words, ratio=args.ratio, lambda_div=args.lambda_div)
            print(f"=== {name} ===")
            print(summary)
            print()
    else:
        summary = summarize_file(inp, model_dir, words=args.words, ratio=args.ratio, lambda_div=args.lambda_div)
        print(summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal Extractive RAG Summarizer (no pretrained models)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on doc/summary pairs")
    p_train.add_argument("--docs", required=True, help="Folder with .txt documents")
    p_train.add_argument("--targets", required=True, help="Folder with reference summaries (.txt) matched by filename")
    p_train.add_argument("--out", required=True, help="Output folder for artifacts")
    p_train.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    p_train.set_defaults(func=cmd_train)

    p_sum = sub.add_parser("summarize", help="Summarize file or folder")
    p_sum.add_argument("--input", required=True, help="Path to .txt file or folder")
    p_sum.add_argument("--model", required=False, default=None, help="Path to trained model folder (artifacts). If omitted, uses unsupervised fallback.")
    grp = p_sum.add_mutually_exclusive_group()
    grp.add_argument("--words", type=int, default=None, help="Word budget for summary (e.g., 100)")
    grp.add_argument("--ratio", type=float, default=None, help="Target ratio of original length (e.g., 0.2)")
    p_sum.add_argument("--lambda_div", type=float, default=0.7, help="MMR trade-off (0..1)")
    p_sum.set_defaults(func=cmd_summarize)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
