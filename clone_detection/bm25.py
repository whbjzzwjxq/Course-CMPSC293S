import json
import tokenize
from collections import Counter

from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model, TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from tqdm import tqdm

from .utils import compute, logger, tokenize_code


def eval_bm25(args, file_name: str, candidate_file_name: str, cut: bool):
    idx2label = {}
    line_idx2index = {}
    label2num = Counter()
    query_indexs = []
    candidate_indexs = []
    lines = open(candidate_file_name).readlines()
    corpus = []
    for i, line in enumerate(tqdm(lines)):
        content = json.loads(line)
        tokens = [v for ty, v, _, _, _ in tokenize_code(content["func"]) if ty != tokenize.COMMENT]
        corpus.append(tokens)
        line_idx2index[i] = content["index"]
        idx2label[content["index"]] = content["label"]
        label2num[content["label"]] += 1
        candidate_indexs.append(content["index"])
    dictionary = Dictionary(corpus)
    corpus = [dictionary.doc2bow(c) for c in corpus]
    model = LuceneBM25Model(corpus, dictionary, k1=1.5, b=0.0)
    bm25_corpus = model[corpus]
    bm25_index = SparseMatrixSimilarity(
        bm25_corpus,
        num_docs=len(corpus),
        num_terms=len(dictionary),
        normalize_queries=True,
        normalize_documents=True,
    )

    lines = open(file_name).readlines()
    queries = []
    for i, line in enumerate(tqdm(lines)):
        content = json.loads(line)
        tokens = [v for ty, v, _, _, _ in tokenize_code(content["func"]) if ty != tokenize.COMMENT]
        if cut:
            tokens = tokens[: len(tokens) // 2]
        queries.append(tokens)
        query_indexs.append(content["index"])
    
    tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
    queries = [dictionary.doc2bow(q) for q in queries]
    tfidf_query = tfidf_model[queries]
    scores = bm25_index[tfidf_query].T

    query_labels = [idx2label[x] for x in query_indexs]
    candidate_labels = [idx2label[x] for x in candidate_indexs]

    result = compute(scores, query_labels, label2num, query_indexs, candidate_indexs, candidate_labels)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
