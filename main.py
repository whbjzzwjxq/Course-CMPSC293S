import argparse
import json
import logging
import random

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

from clone_detection.ast_bm25 import eval_ast_bm25
from clone_detection.bm25 import eval_bm25
from clone_detection.dense import eval_dense


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--pretrained_dir",
        default="reacc-py-retriever",
        type=str,
        help="The directory where the trained model and tokenizer are saved.",
    )

    parser.add_argument(
        "--lang", default="python", type=str, help="Language of dataset"
    )

    parser.add_argument("--num_vec", type=int, default=-1, help="number of vectors")

    parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_dir)
    model = RobertaModel.from_pretrained(args.pretrained_dir, add_pooling_layer=False)

    query_file_name = "./data/query.jsonl"
    candidate_file_name = "./data/corpus.jsonl"
    cut = True
    # Dense handles the cutting of input in its own logic.
    print("---- Dense Retrieval ----")
    result_dr = eval_dense(
        args, model, tokenizer, candidate_file_name, candidate_file_name, cut
    )
    for k, v in result_dr.items():
        if k != "timecost":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.2}")

    # BM25 handles the cutting of input in its own logic.
    # Because the original query will be failed.
    print("---- BM25 ----")
    result_bm25 = eval_bm25(args, candidate_file_name, candidate_file_name, cut)
    for k, v in result_bm25.items():
        if k != "timecost":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.2}")

    print("---- BM25 + Tokenization ----")
    result_token = eval_ast_bm25(args, query_file_name, candidate_file_name, cut, gen_api=False)
    for k, v in result_token.items():
        if k != "timecost":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.2}")

    print("---- BM25 + Tokenization + API ----")
    result_api = eval_ast_bm25(args, query_file_name, candidate_file_name, cut, gen_api=True)
    for k, v in result_api.items():
        if k != "timecost":
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.2}")

    result_dr = {}
    result_bm25 = {}

    result = {
        "dr": result_dr,
        "bm25": result_bm25,
        "bm25_token": result_token,
        "bm25_api": result_api,
    }

    print(result)

    # with open("result.json", "w") as f:
    #     json.dump(result, f)


if __name__ == "__main__":
    main()
