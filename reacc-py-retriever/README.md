---
license: mit
---

# ReACC-py-retriever

This is the retrieval model for [ReACC: A Retrieval-Augmented Code Completion Framework](https://arxiv.org/abs/2203.07722).

In this paper, the model is used to retrieve similar codes given an incompletion code snippet as query. The model can be also used for incomplete code-to-code search, code clone detection.

`py-retriever` is BERT-like encoder consisting of 12 transformer layers. It is continual pre-trained on [GraphCodeBERT](https://huggingface.co/microsoft/graphcodebert-base) with contrastive learning in Python programming language. More details can be found in our paper.

Note that the format of input codes is different from original source code. We normalize the source codes to better capture information from line break and indention in Python. An example of input is:
```python
sum = 0<endofline>for val in numbers:<endofline><INDENT>sum = sum+val
```
To get more information about how to convert source codes into this format, please refer to [ReACC GitHub repo](https://github.com/microsoft/ReACC).