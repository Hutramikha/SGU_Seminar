import torch
import json
from pathlib import Path
from model import TransformerClassifier
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def find_errors():
    processed_dir = Path("data/processed")
    with open(processed_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    test_data = torch.load(processed_dir / "test.pt")
    input_ids = test_data["input_ids"]
    labels = test_data["labels"]
    texts = test_data["texts"]
    
    # Sử dụng cấu hình Transformer_d32_ff64 vì cấu hình này có nhiều lỗi hơn để dễ phân tích
    model = TransformerClassifier(
        vocab_size=meta["vocab_size"],
        d_model=32,
        d_ff=64,
        max_len=meta["max_len"],
        num_classes=meta["num_classes"],
    )
    
    # Load model gốc (trước khi cải tiến Dropout)
    model.load_state_dict(torch.load("results/model_Transformer_d32_ff64.pt", map_location="cpu"))
    model.eval()
    
    errors = []
    with torch.no_grad():
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)
        
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                errors.append({
                    "text": texts[i],
                    "true_label": meta["label_names"][labels[i]],
                    "pred_label": meta["label_names"][preds[i]]
                })
                
    print("=== DANH SACH CAC CAU MO HINH PHAN LOAI SAI TRÊN TAP TEST ===")
    for i, err in enumerate(errors[:10]):
        print(f"Loi {i+1}: '{err['text']}' | Nhan dung: {err['true_label']} | Du doan: {err['pred_label']}")

if __name__ == "__main__":
    find_errors()
