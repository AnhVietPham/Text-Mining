import torch
from transformers import AutoModel, AutoTokenizer

vihealthbert = AutoModel.from_pretrained("demdecuong/vihealthbert-base-word")
tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")

if __name__ == "__main__":
    # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
    line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

    input_ids = torch.tensor([tokenizer.encode(line)])
    with torch.no_grad():
        features = vihealthbert(input_ids)
