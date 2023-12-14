import logging

from dataclasses import dataclass

from typing import List

import tokenize

from io import BytesIO

import numpy as np

logger = logging.getLogger("ReACC")

K = 100


@dataclass
class InputFeatures:
    """A single training/test features for a example."""

    code_ids: List[int]
    index: int
    label: int


def tokenize_code(code: str):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
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
            reconstructed_code += '\n' * (start_line - end_line)
            reconstructed_code += ' ' * start_col
        else:
            reconstructed_code += ' ' * (start_col - end_col)

        reconstructed_code += tok.string
        last_end = tok.end

    return reconstructed_code


def compute(scores, query_labels, label2num, query_indexs, candidate_indexs, candidate_labels):
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
        "Data size": len(MAP),
        "eval_map": float(np.mean(MAP)),
        f"eval_map_at_{K}": float(np.mean(MAP_K)),
        "eval_prec": float(PREC / len(MAP)),
    }
    return result
