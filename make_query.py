import json

from clone_detection.utils import remove_second_half_of_tokens

def _main():
    candidate_file_name = "./data/corpus.jsonl"
    lines = open(candidate_file_name).readlines()
    new_lines = []
    all = 0
    for i, line in enumerate(lines):
        content = json.loads(line)
        code = remove_second_half_of_tokens(content["func"])
        all += len(content["func"]) / len(code)
        content["func"] = code.removeprefix("utf-8\n")
        new_lines.append(json.dumps(content) + "\n")

    print(all / i)
    
    query_file_name = "./data/query.jsonl"
    with open(query_file_name, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    _main()
