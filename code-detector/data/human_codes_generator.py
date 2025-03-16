from datasets import load_dataset
import os
import json

dataset_python = load_dataset("code_search_net", "python", trust_remote_code=True)
dataset_java = load_dataset("code_search_net", "java", trust_remote_code=True)
dataset_go = load_dataset("code_search_net", "go", trust_remote_code=True)
dataset_javascript = load_dataset("code_search_net", "javascript", trust_remote_code=True)

# extract human codes
human_code_python= [
    sample["whole_func_string"]
    for sample in dataset_python["train"]
][:1500]
human_code_java= [
    sample["whole_func_string"]
    for sample in dataset_java["train"]
][:1500]
human_code_go= [
    sample["whole_func_string"]
    for sample in dataset_go["train"]
][:1500]
human_code_javascript= [
    sample["whole_func_string"]
    for sample in dataset_javascript["train"]
][:1500]

codes = human_code_go + human_code_java + human_code_javascript + human_code_python
dataset = [{"codes": c, "label": 0} for c in codes]
file_path = os.path.join(os.path.dirname(__file__), "human_codes_dataset.json")

with open(file_path, "w+", encoding='utf-8') as f:
    json.dump(dataset, f, indent=2)
    f.flush()