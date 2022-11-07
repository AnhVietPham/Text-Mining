import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

if __name__ == "__main__":
    vihealthbert = AutoModel.from_pretrained("demdecuong/vihealthbert-base-word")
    tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")

    # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
    line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

    input_ids = torch.tensor([tokenizer.encode(line)])
    with torch.no_grad():
        features = vihealthbert(input_ids)  # Models outputs are now tuples
    label_indices = np.argmax(features[0].to('cpu').numpy(), axis=2)
    print(label_indices)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    print(tokens)
    # for token, label_idx in zip(tokens, label_indices[0]):
    #     if token.startswith("##"):
    #         new_tokens[-1] = new_tokens[-1] + token[2:]
    #     else:
    #         new_labels.append(tag_values[label_idx])
    #         new_tokens.append(token)
