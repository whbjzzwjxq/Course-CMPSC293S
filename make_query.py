import json

from clone_detection.utils import tokenize_code

def _main():
    candidate_file_name = "./data/corpus.jsonl"
    lines = open(candidate_file_name).readlines()
    new_lines = []
    for i, line in enumerate(lines):
        content = json.loads(line)
        tokens = [t.string for t in tokenize_code(content["func"])]
        tokens = tokens[:len(tokens) // 2]
        code = "".join(tokens)
        content["func"] = code
        new_lines.append(json.dumps(content) + "\n")
    
    query_file_name = "./data/query.jsonl"
    with open(query_file_name, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    _main()
