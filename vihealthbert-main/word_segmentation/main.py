import torch
from transformers import AutoModel, AutoTokenizer, InputExample
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("/Users/sendo_mac/Documents/avp/Text-Mining/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg",
                         max_heap_size='-Xmx500m')

vihealthbert = AutoModel.from_pretrained("demdecuong/vihealthbert-base-word")
tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")


def _create_examples(texts, slots):
    """Creates examples for the training and dev sets."""
    labels = ["O", "B-DATE", "I-DATE", "B-NAME", "B-AGE", "B-LOCATION", "I-LOCATION", "B-JOB", "I-JOB",
              "B-ORGANIZATION", "I-ORGANIZATION", "B-PATIENT_ID", "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE",
              "B-GENDER", "B-TRANSPORTATION", "I-TRANSPORTATION", "I-NAME", "I-PATIENT_ID", "I-AGE", "I-GENDER"]
    examples = []
    for i, (text, slot) in enumerate(zip(texts, slots)):
        guid = "predict"
        # 1. input_text
        words = text.split()  # Some are spaced twice

        # 2. slot
        slot_labels = []
        for s in slot:
            slot_labels.append(labels.index(s) if s in labels else slot_labels.index("O"))
        try:
            assert len(words) == len(slot_labels)
        except:
            print(i)
            print(words)
            print(slot_labels)
            print(len(words))
            print(len(slot_labels))
        examples.append(InputExample(guid=guid, words=words, slot_labels=slot_labels))
    return examples


if __name__ == "__main__":
    # # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
    # line = "Tôi là sinh_viên trường đại_học Công_nghệ ."
    #
    # input_ids = torch.tensor([tokenizer.encode(line)])
    # with torch.no_grad():
    #     features = vihealthbert(input_ids)

    # Input
    text = "Từ 24-7 đến 31-7, bệnh nhân được mẹ là bà H.T.P (47 tuổi) đón về nhà ở phường Phước Hoà (bằng xe máy), không đi đâu chỉ ra Tạp hoá Phượng, chợ Vườn Lài, phường An Sơn cùng mẹ bán tạp hoá ở đây ."
    # To perform word (and sentence) segmentation
    sentences = rdrsegmenter.tokenize(text)
    for sentence in sentences:
        print(" ".join(sentence))
    slot_label = ['0'] * len(sentences)
    example = _create_examples(sentences, slot_label)
    print(example)
