import os

from src.rag_summarizer.model import train_model, summarize_file


def test_train_and_summarize_tmp(tmp_path):
	docs = os.path.join(os.getcwd(), 'data', 'sample', 'train', 'docs')
	tgts = os.path.join(os.getcwd(), 'data', 'sample', 'train', 'targets')
	out = tmp_path / 'model'
	out.mkdir()
	train_model(docs, tgts, str(out))
	sample_doc = os.path.join(docs, 'doc1.txt')
	summary = summarize_file(sample_doc, str(out), words=50, ratio=None)
	assert isinstance(summary, str)
	assert len(summary.strip()) > 0
