import logging
import tokenize
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np

from tree_sitter import Language, Parser

from tree_parser import (
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)

K = 100


@dataclass
class InputFeatures:
    """A single training/test features for a example."""

    code_ids: List[int]
    index: int
    label: int


def tokenize_code(code: str):
    tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
    return tokens


def remove_second_half_of_tokens(code: str):
    # Tokenize the source code
    tokens = list(tokenize_code(code))

    # Calculate the halfway point
    halfway_point = len(tokens) // 2

    # Keep only the first half of the tokens
    first_half_tokens = tokens[:halfway_point]

    # Reconstruct the code from the remaining tokens
    reconstructed_code = ""
    last_end = (1, 0)  # Starting with the beginning of the file
    for tok in first_half_tokens:
        # Add whitespace for proper token separation
        start_line, start_col = tok.start
        end_line, end_col = last_end
        if start_line > end_line:
            reconstructed_code += "\n" * (start_line - end_line)
            reconstructed_code += " " * start_col
        else:
            reconstructed_code += " " * (start_col - end_col)

        reconstructed_code += tok.string
        last_end = tok.end

    return reconstructed_code

def span_select(code_bytes, *nodes, indent=False):
    if not nodes:
        return ""
    start, end = nodes[0].start_byte, nodes[-1].end_byte
    select = code_bytes[start:end].decode("utf-8")
    if indent:
        return " " * nodes[0].start_point[1] + select
    return select

def get_api_seq(node, api_seq, code_bytes, tmp=None):
    if node.type == "call":
        api = node.child_by_field_name("function")
        if tmp:
            tmp.append(span_select(code_bytes, api))
            ant = False
        else:
            tmp = [span_select(code_bytes, api)]
            ant = True
        for child in node.children:
            get_api_seq(child, api_seq, code_bytes, tmp)
        if ant:
            api_seq += tmp[::-1]
            tmp = None
    else:
        for child in node.children:
            get_api_seq(child, api_seq, code_bytes, tmp)

def tokenize_code_ast(parser: Parser, code: str, gen_api: bool):
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code_lines = code.split("\n")
    code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
    api_seqs = []
    if gen_api:
        get_api_seq(root_node, api_seqs, code_bytes)
    # original_tokens = [t.string for t in tokenize_code(code) if t.type != tokenize.COMMENT]
    return code_tokens, api_seqs


def compute(
    scores, query_labels, label2num, query_indexs, candidate_indexs, candidate_labels
):
    sort_ids = np.argsort(scores, axis=-1, kind="quicksort", order=None)[:, ::-1]
    MAP = []
    MAP_K = []
    PREC = 0.0
    for i in range(scores.shape[0]):
        cont = 0
        label = int(query_labels[i])
        div = min(K, label2num[label])
        query_index = query_indexs[i]
        Avep = []
        for j, index in enumerate(list(sort_ids[i])):
            if query_index == candidate_indexs[index]:
                cont += 1
                continue
            if j - cont == K:
                MAP_K.append(sum(Avep) / div)
            if int(candidate_labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1 - cont))
                if j - cont == 0:
                    PREC += 1.0
        if len(Avep) > 0:
            MAP.append(sum(Avep) / len(Avep))
        else:
            MAP.append(0.0)
    result = {
        # "Data size": len(MAP),
        "eval_map": float(np.mean(MAP)),
        f"eval_map_at_{K}": float(np.mean(MAP_K)),
        "eval_prec": float(PREC / len(MAP)),
    }
    return result
