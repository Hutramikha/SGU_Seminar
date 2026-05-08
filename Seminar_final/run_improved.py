import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from model_improved import TransformerClassifier
from train import set_seed, load_split, train_one_config
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    
    with open(processed_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_loader = DataLoader(load_split(processed_dir / "train.pt"), batch_size=32, shuffle=True)
    val_loader = DataLoader(load_split(processed_dir / "val.pt"), batch_size=32)
    test_loader = DataLoader(load_split(processed_dir / "test.pt"), batch_size=32)

    model = TransformerClassifier(meta["vocab_size"], 128, 256, meta["max_len"], meta["num_classes"], dropout=0.1)
    
    print("Running Improved Model (with Dropout & Padding Mask)...")
    result = train_one_config("Transformer_d128_ff256_improved", model.to(device), train_loader, val_loader, test_loader, 20, 1e-3, device, results_dir)

    # Ghi kết quả vào summary.json
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        # Xóa kết quả improved cũ nếu có (tránh trùng lặp)
        summary = [s for s in summary if s["model_name"] != result["model_name"]]
    else:
        summary = []
    
    summary.append(result)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nKet qua da duoc luu vao: {summary_path}")

if __name__ == "__main__":
    main()
