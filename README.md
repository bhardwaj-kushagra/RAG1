# Minimal Extractive RAG Summarizer (No Pretrained Models)

This project builds a bare-minimum, trainable extractive summarizer with a RAG-like retrieval-and-ranking approach using only classical NLP/ML (no pretrained LLMs or Hugging Face models).

What it does:
- Splits text into sentences
- Creates TF‑IDF features + simple heuristics
- Trains a tiny Ridge regression scorer using ROUGE-1 F1 vs reference summaries
- Selects sentences via MMR to form an extractive summary under a word budget

You can train on your own pairs of .txt documents and reference summaries, then run the summarizer on new .txt files to get summaries.

## Quickstart

1) Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Train on the tiny toy data

```powershell
python cli.py train --docs data/sample/train/docs --targets data/sample/train/targets --out artifacts\model
```

3) Summarize a file or a folder of .txt files

```powershell
# Single file
python cli.py summarize --input data/sample/train/docs/doc1.txt --model artifacts\model --words 80

# Or a folder (outputs to console by default)
python cli.py summarize --input data/sample/train/docs --model artifacts\model --words 80
```

Options: use `--ratio 0.2` to target ~20% of original word count instead of a fixed `--words` budget.

## Project layout

- `src/rag_summarizer/` core library
  - `tokenize.py` simple sentence/word tokenizers (regex-based; no external data)
  - `features.py` TF-IDF features and heuristics
  - `mmr.py` Maximal Marginal Relevance sentence selection
  - `model.py` training, scoring, and summarization pipeline
- `cli.py` command-line interface for train and summarize
- `train.py` thin wrapper to train models (used by CLI)
- `data/sample/` tiny toy dataset for smoke tests

## About the “RAG” aspect here

RAG traditionally means retrieval-augmented generation with a generative model. Since we avoid pretrained LLMs, we implement a pragmatic, extractive variant: TF-IDF retrieval signals and a learned sentence scorer select sentences directly from the document. You can extend it to multi-doc retrieval by concatenating retrieved passages before scoring/selection.

## Bring your own training data

You need pairs of documents and their reference summaries in plain text files, matched by filename.

Directory structure example:

```
data/your_dataset/
  train/
    docs/
      article_001.txt
      article_002.txt
    targets/
      article_001.txt
      article_002.txt
```

Each `targets/*.txt` contains the human-written summary for the matching `docs/*.txt`.

### Public dataset options (no HF needed)
- CNN/DailyMail: original URLs and scripts exist, but third-party mirrors provide raw text; be mindful of licenses.
- XSum: BBC articles with single-sentence summaries. Find raw text mirrors or request academic access.
- SAMSum: messenger-like conversations + summaries; raw JSON can be converted to .txt easily.
- ArXiv/PubMed: scientific abstracts as summaries; raw sources available via API dumps.

Tip: You can also create your own small dataset by writing 20–50 pairs of document and short summary; the model here is simple and trains quickly.

## Notes and limits
- Extractive: the summary is composed of original sentences from the input.
- No external pretrained models used.
- For larger datasets, consider increasing n-grams, regularization, and adding more features (title similarity, section cues, etc.).

## License
Provided as-is for educational purposes. Be sure your data usage complies with relevant licenses.
