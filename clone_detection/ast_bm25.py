import json
import time
from collections import Counter

from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model, TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from tqdm import tqdm
from tree_sitter import Language, Parser

from .utils import compute, tokenize_code_ast


def eval_ast_bm25(args, file_name: str, candidate_file_name: str, cut: bool, gen_api: bool):
    timer = time.perf_counter()
    LANGUAGE = Language("tree_parser/my-languages.so", "python")
    parser = Parser()
    parser.set_language(LANGUAGE)

    idx2label = {}
    line_idx2index = {}
    label2num = Counter()
    query_indexs = []
    candidate_indexs = []
    lines = open(candidate_file_name).readlines()
    corpus = []
    api_corpus = []
    for i, line in enumerate(tqdm(lines)):
        content = json.loads(line)
        tokens, api_seqs = tokenize_code_ast(parser, content["func"], gen_api)
        corpus.append(tokens)
        api_corpus.append(api_seqs)
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

    if gen_api:
        api_dictionary = Dictionary(api_corpus)
        api_corpus = [api_dictionary.doc2bow(c) for c in api_corpus]
        api_model = LuceneBM25Model(api_corpus, api_dictionary, k1=1.5, b=0.0)
        api_bm25_corpus = api_model[api_corpus]
        api_bm25_index = SparseMatrixSimilarity(
            api_bm25_corpus,
            num_docs=len(api_corpus),
            num_terms=len(api_dictionary),
            normalize_queries=True,
            normalize_documents=True,
        )

    lines = open(file_name).readlines()
    queries = []
    api_queries = []
    for i, line in enumerate(tqdm(lines)):
        content = json.loads(line)
        tokens, api_seqs = tokenize_code_ast(parser, content["func"], gen_api)
        queries.append(tokens)
        api_queries.append(api_seqs)
        query_indexs.append(content["index"])

    tfidf_model = TfidfModel(dictionary=dictionary, smartirs="bnn")
    queries = [dictionary.doc2bow(q) for q in queries]
    tfidf_query = tfidf_model[queries]

    if gen_api:
        api_tfidf_model = TfidfModel(dictionary=api_dictionary, smartirs="bnn")
        api_queries = [api_dictionary.doc2bow(q) for q in api_queries]
        api_tfidf_query = api_tfidf_model[api_queries]

    scores = bm25_index[tfidf_query].T

    if gen_api:
        delta = 0.1
        api_scores = api_bm25_index[api_tfidf_query].T
        scores = (scores + api_scores * delta) / (1 + delta)

    query_labels = [idx2label[x] for x in query_indexs]
    candidate_labels = [idx2label[x] for x in candidate_indexs]

    result = compute(
        scores,
        query_labels,
        label2num,
        query_indexs,
        candidate_indexs,
        candidate_labels,
    )
    timecost = time.perf_counter() - timer
    result["timecost"] = timecost
    return result
