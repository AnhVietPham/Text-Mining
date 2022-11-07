from vncorenlp import VnCoreNLP
from transformers import AutoModel, AutoTokenizer

rdrsegmenter = VnCoreNLP("/Users/anhvietpham/Documents/cs/text-mining/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg",
                         max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained("demdecuong/vihealthbert-base-word")

if __name__ == "__main__":
    # Input
    text = "Bác sĩ Nguyễn Trung Nguyên, Giám đốc Trung tâm Chống độc , Bệnh viện Bạch Mai, cho biết bệnh nhân được chuyển đến bệnh viện ngày 7/3, chẩn đoán ngộ độc thuốc điều trị sốt rét chloroquine."

    # To perform word (and sentence) segmentation
    sentences = rdrsegmenter.tokenize(text)
    for sentence in sentences:
        print(" ".join(sentence))

    print(sentences)
    encoded_review = tokenizer.encode_plus(
        str(sentences),
        max_length=70,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    print(input_ids)
    print(attention_mask)
