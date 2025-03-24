from detector_dataset import DetectorDataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import os
from torch.utils.data import DataLoader, random_split


def get_ppl_vector(code_texts, model_name, batch_size = 8):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] use {model_name}, start calculating ppl")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # load_in_8bit=True, 
        torch_dtype=torch.float16,  # reduce memory usage
        low_cpu_mem_usage=True, 
    ).to(device)

    # handle the different model's parameters
    kwargs = {
        "trust_remote_code": True, 
        # "add_prefix_space": False
    }
    if "deepseek" not in model_name.lower():
        kwargs["add_prefix_space"] = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model_ppl = np.zeros(len(code_texts))

    input_args = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 512,
        "padding": True,
        "add_special_tokens": False
    }
    # if "gpt" in model_name.lower() or "starcoder" in model_name.lower():
    #     input_args["add_special_tokens"] = False
    for i in range(0, len(code_texts), batch_size):
        batch = code_texts[i : i + batch_size]
        # code_lengths = [len(tokenizer.tokenize(code)) for code in batch]
        # p100_length = int(np.percentile(code_lengths, 100))
        # max_length = min(int(p100_length * 1.1), model.config.max_position_embeddings)

        inputs = tokenizer(batch, **input_args).to(device)
        with torch.autocast("mps", dtype=torch.float16):
            with torch.no_grad():
                outputs = model(inputs["input_ids"], labels=inputs["input_ids"], attention_mask=inputs.get("attention_mask", None))
                losses = outputs.loss if outputs.loss is not None else torch.zeros(1)
        batch_ppl = torch.exp(losses).cpu().numpy()
        model_ppl[i:i+batch_size] = batch_ppl

        del inputs, outputs, batch, losses
        torch.mps.empty_cache()
        mem_usage = torch.mps.current_allocated_memory() / 1e9
        print(f"[{datetime.now().strftime('%H:%M:%S')}] use {model_name}, calculating ppl: {batch_ppl}, progress{i}, memory usage: {mem_usage}GB")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] use {model_name}, end calculating ppl")

    return model_ppl

def save_ppl(model_ppl, model_name: str, is_test_data = False):
    model_name = model_name.replace('/', '-')
    path  = os.path.join(os.path.dirname(__file__), f"./features/{model_name}{'-test' if is_test_data else '-train'}.npy")
    np.save(path, model_ppl)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] save model ppl {model_name} to {path}") 

def main():
    feature_path  = os.path.join(os.path.dirname(__file__), "./features")
    train_data_path  = os.path.join(os.path.dirname(__file__), "./data/train_dataset.json")
    test_data_path  = os.path.join(os.path.dirname(__file__), "./data/test_dataset.json")
    
    os.makedirs(feature_path, exist_ok=True)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.mps.empty_cache()

    train_dataset = DetectorDataset(train_data_path)
    test_dataset = DetectorDataset(test_data_path)
    print(f"The length of train dataset is {len(train_dataset)} | "
          f"The length of test dataset is {len(test_dataset)}")
    
    train_codes = [d['codes'] for d in train_dataset]
    train_labels = [d['label'] for d in train_dataset]
    test_codes = [d['codes'] for d in test_dataset]
    test_labels = [d['label'] for d in test_dataset]
    model_names = [
        "gpt2", 
        "gpt2-xl",
        "gpt2-medium",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "meta-llama/CodeLlama-13b-Instruct-hf",
        "bigcode/starcoder2-3b",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "Salesforce/codegen-350M-mono",
        "Salesforce/codegen-2B-mono",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct"
    ]
    for model in model_names:
        model_ppl = get_ppl_vector(train_codes, model_name=model)
        save_ppl(model_ppl, model)
        test_model_ppl = get_ppl_vector(test_codes, model_name=model)
        save_ppl(test_model_ppl, model, True)
        


if __name__ == "__main__":
    main()