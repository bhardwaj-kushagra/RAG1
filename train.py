from src.rag_summarizer.model import train_model

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    train_model(args.docs, args.targets, args.out, alpha=args.alpha)
