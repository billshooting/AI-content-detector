import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
from classifier_model import ClassifierModel, FocalLoss
from training_visualizer import TrainingVisualizer
from detector_dataset import DetectorDataset

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model_names = [
    "gpt2", 
    # "gpt2-xl",
    # "gpt2-medium",
    # "EleutherAI/gpt-neo-125M",
    # "EleutherAI/gpt-neo-1.3B",
    # "EleutherAI/gpt-neo-2.7B",
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/CodeLlama-7b-Instruct-hf",
    "meta-llama/CodeLlama-13b-Instruct-hf",
    # "bigcode/starcoder2-3b",
    # "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    # "deepseek-ai/deepseek-coder-1.3b-instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "Salesforce/codegen-350M-mono",
    # "Salesforce/codegen-2B-mono",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    # "microsoft/Phi-4-mini-instruct",
    # "microsoft/Phi-3.5-mini-instruct"
]
LENGTH_FEATURES = 1
TRAIN_DATA_SIZE = 10000
TEST_DATA_SIZE = 2000
MODEL_NUM = 15

def merge_features(data_size: int, is_test_data = False) -> torch.Tensor:
    features_path = os.path.join(os.path.dirname(__file__), "./features")
    feature_files = sorted( # model number
        [f for f in os.listdir(features_path) if f.endswith("-train.npy")]
    )
    if (is_test_data):
        feature_files = sorted( # model number
            [f for f in os.listdir(features_path) if f.endswith("-test.npy")]
        )
    merged = np.zeros((data_size, LENGTH_FEATURES, len(feature_files)), dtype=np.float32)
    for i, file_name in enumerate(feature_files):
        print(file_name)
        # if (file_name != 'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B-train.npy'):
        #     continue
        tensor = np.load(os.path.join(os.path.dirname(__file__), f"./features/{file_name}"))
        if (tensor.shape == (data_size,)):
            tensor = tensor.reshape(-1, LENGTH_FEATURES)
        merged[:, :, i] = tensor

    return torch.from_numpy(merged)

def normalize_by_model(tensor: torch.Tensor) -> torch.Tensor:
    normalized_tensor = torch.zeros_like(input=tensor, dtype=torch.float32)
    scaler = StandardScaler()
    for i in range(LENGTH_FEATURES):
        data_to_normalize = tensor[:,i,:]
        normalized_features = scaler.fit_transform(data_to_normalize)
        normalized_tensor[:,i,:] = torch.from_numpy(normalized_features)
    return normalized_tensor

def train(model: ClassifierModel, train_data: DataLoader, device: str, epochs = 30, test_data: DataLoader = None):
    model = model.to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50)
    visualizer = TrainingVisualizer()
    total_steps = len(train_data.dataset) * epochs
    global_step = 0

    model.train()

    for epoch in range(epochs):
        epoch_step = 0
        epoch_loss = 0
        predicts, truths = [], []
        model.train()
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type='mps', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
            optimizer.step()
            scheduler.step()
            epoch_step = epoch_step + inputs.size(0)
            global_step = global_step + inputs.size(0)

            current_lr = optimizer.param_groups[0]['lr']
            visualizer.update(global_step, loss.item(), current_lr)

            # metrics
            epoch_loss += loss.item() * inputs.size(0)
            predicts.extend(torch.argmax(outputs, 1).cpu().numpy())
            truths.extend(labels.cpu().numpy())
            avg_loss = epoch_loss / epoch_step
            acc = accuracy_score(truths, predicts)
            f1 = f1_score(truths, predicts, average="macro")

            # print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} | "
            #       f"Progress: {global_step} / {total_steps} | Grad: {total_norm:.4f} | "
            #       f"Avg Loss: {avg_loss:.4f} | accuracy score: {acc * 100:.2f}% | f1 score: {f1 * 100:.2f}%")
        
        if (test_data is not None):
            print(f"Epoch {epoch} ")
            validate(model, test_data, device)
    visualizer.save_plots(epochs)

@torch.no_grad()
def validate(model: ClassifierModel, validate_data: DataLoader, device: str):
    predicts, truths = [], []
    model.eval()

    for inputs, labels in validate_data:
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs = inputs.to(device).squeeze(1)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        predicts.extend(torch.argmax(outputs, 1).cpu().numpy())
        truths.extend(labels.cpu().numpy())

    acc = accuracy_score(truths, predicts)
    f1 = f1_score(truths, predicts, average="macro")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] validation ｜ accuracy score: {acc * 100:.2f}% | f1 score: {f1 * 100:.2f}%")

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    train_data_path  = os.path.join(os.path.dirname(__file__), "./data/train_dataset.json")
    test_data_path  = os.path.join(os.path.dirname(__file__), "./data/test_dataset.json")
    train_dataset = DetectorDataset(train_data_path)
    test_dataset = DetectorDataset(test_data_path)
    train_tensor = merge_features(TRAIN_DATA_SIZE)
    test_tensor = merge_features(TEST_DATA_SIZE, True)
    train_labels = torch.tensor([d['label'] for d in train_dataset])
    test_labels = torch.tensor([d['label'] for d in test_dataset])
    print(train_labels)
    print(test_labels)

    print(f"shape of train tensor is {train_tensor.shape}, dtype is {train_tensor.dtype}")
    print(f"shape of train label is {train_labels.shape}, dtype is {train_labels.dtype}")
    print(f"shape of test tensor is {test_tensor.shape}, dtype is {test_tensor.dtype}")
    print(f"shape of test label is {test_labels.shape}, dtype is {test_labels.dtype}")
    print(train_labels[:10])

    normalized_train_tensor = normalize_by_model(train_tensor)
    normalized_test_tensor = normalize_by_model(test_tensor)
    for i in range(MODEL_NUM):
        col = normalized_train_tensor[:,0,i]
        col2 = normalized_test_tensor[:,0,i]
        print(f"train model: {i}, the mean of col: {col.mean():.2f}, the std of col is {col.std():.2f}")
        print(f"test model: {i}, the mean of col2: {col2.mean():.2f}, the std of col2 is {col2.std():.2f}")
    
    model = ClassifierModel(1, MODEL_NUM)
    train_data_loader = DataLoader(TensorDataset(normalized_train_tensor, train_labels), batch_size=128, shuffle=True)
    validate_data_loader = DataLoader(TensorDataset(normalized_test_tensor, test_labels), batch_size=64, shuffle=False)


    train(model, train_data_loader, device, 50, validate_data_loader)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "models/trained_model.pth"))
    # validate(model, validate_data_loader, device)

    # X_train_np = normalized_train_tensor.numpy().squeeze(1)
    # y_train_np = train_labels.numpy()
    # clf = LogisticRegression(max_iter=1000)
    # clf.fit(X_train_np, y_train_np)
    # train_pred = clf.predict(X_train_np)
    # print("LogReg Train F1:", f1_score(y_train_np, train_pred))

if __name__ == "__main__":
    main()