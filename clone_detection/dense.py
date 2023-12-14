import json
from collections import Counter
from functools import partial

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from .processor import Processor
from .utils import InputFeatures, compute, logger


class CodeWithDocNoRepDataset(Dataset):
    def __init__(self, tokenizer, args, file_name: str, cut_ratio=0.0):
        self.tokenizer = tokenizer
        self.args = args
        data_file = file_name
        self.proc = Processor(args.lang, remove_comments=False)

        # load index
        logger.info(f"Creating features from {data_file}")

        self.examples = []
        lines = open(data_file).readlines()
        for i, line in enumerate(lines):
            content = json.loads(line)
            self.proc.update(content["func"])
            code = self.proc.untokenize(cut_ratio=cut_ratio, fix_cut_pos=True)
            normal_code = self.proc.convert_to_normal(code)
            self.proc.update(normal_code)
            api_seq = self.proc.get_api_seq()
            token_id = self.encode_v3(code, api_seq)
            self.examples.append(
                InputFeatures(token_id, content["index"], int(content["label"]))
            )
        logger.info(f"loaded {len(self.examples)} data")

        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def encode_v3(self, code, api_seq):
        code_tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(code)
            + [self.tokenizer.sep_token]
            + self.tokenizer.tokenize(" ".join(api_seq))
            + [self.tokenizer.sep_token]
        )
        code_tokens = code_tokens[: self.args.block_size]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return code_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), self.examples[i].index


def my_collect_fn(sequences, batch_first=True, padding_value=1):
    inputs1 = []
    inputs2 = []
    for x1, x2 in sequences:
        inputs1.append(x1)
        inputs2.append(x2)
    return (pad_sequence(inputs1, batch_first, padding_value), inputs2)


def eval_dense(
    args,
    model: RobertaModel,
    tokenizer: RobertaTokenizer,
    file_name: str,
    candidate_file_name: str,
    cut: bool,
):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    query_dataset = CodeWithDocNoRepDataset(
        tokenizer, args, file_name, cut_ratio=1.0 if cut else 0.0
    )
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=partial(
            my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        num_workers=4,
    )

    candidate_dataset = CodeWithDocNoRepDataset(tokenizer, args, candidate_file_name, cut_ratio=0.0)
    candidate_sampler = SequentialSampler(candidate_dataset)
    candidate_dataloader = DataLoader(
        candidate_dataset,
        sampler=candidate_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=partial(
            my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        num_workers=4,
    )

    idx2label = {}
    label2num = Counter()
    lines = open(candidate_file_name).readlines()
    corpus = {}
    for i, line in enumerate(tqdm(lines)):
        content = json.loads(line)
        idx2label[content["index"]] = content["label"]
        label2num[int(content["label"])] += 1

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num Query = %d", len(query_dataset))
    logger.info("  Num Candidate = %d", len(candidate_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.to(args.device)

    model.eval()
    query_vecs = []
    query_indexs = []
    candidate_vecs = []
    candidate_indexs = []

    for batch in tqdm(query_dataloader, total=len(query_dataloader)):
        code_inputs = batch[0].to(args.device)
        index = batch[1]
        with torch.no_grad():
            attn_mask = (code_inputs != tokenizer.pad_token_id).clone().detach().to(torch.uint8)
            code_vec = model(code_inputs, attention_mask=attn_mask)[0]
            code_vec = torch.nn.functional.normalize(code_vec[:, 0, :], dim=1)
            query_vecs.append(code_vec.cpu().numpy())
            query_indexs.extend(index)

    for batch in tqdm(candidate_dataloader, total=len(candidate_dataloader)):
        code_inputs = batch[0].to(args.device)
        index = batch[1]
        with torch.no_grad():
            attn_mask = (code_inputs != tokenizer.pad_token_id).clone().detach().to(torch.uint8)
            code_vec = model(code_inputs, attention_mask=attn_mask)[0]
            if args.num_vec > 0:
                code_vec = torch.nn.functional.normalize(
                    code_vec[:, : args.num_vec, :], dim=2
                )
            else:
                code_vec = torch.nn.functional.normalize(code_vec[:, 0, :], dim=1)
            candidate_vecs.append(code_vec.cpu().numpy())
            candidate_indexs.extend(index)

    model.train()

    query_vecs = np.concatenate(query_vecs, 0)
    candidate_vecs = np.concatenate(candidate_vecs, 0)
    query_labels = [idx2label[x] for x in query_indexs]
    candidate_labels = [idx2label[x] for x in candidate_indexs]

    if args.num_vec > 0:
        scores = np.einsum("nd,mvd->nmv", query_vecs, candidate_vecs).max(-1)
    else:
        scores = np.matmul(query_vecs, candidate_vecs.T)
    
    result = compute(scores, query_labels, label2num, query_indexs, candidate_indexs, candidate_labels)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
