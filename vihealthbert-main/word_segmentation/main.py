from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("/Users/sendo_mac/Documents/avp/Text-Mining/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg",
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

    """ preprocess """
    tokens, valid_positions = tokenizer.encode(str(sentences))
    ## insert "[CLS]"
    tokens.insert(0, "[CLS]")
    valid_positions.insert(0, 1)
    ## insert "[SEP]"
    tokens.append("[SEP]")
    valid_positions.append(1)
    segment_ids = []
    for i in range(len(tokens)):
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < 70:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        valid_positions.append(0)

    print(input_ids)
    print(input_mask)
    print(segment_ids)
    print(valid_positions)

    # input_ids = encoded_review['input_ids']
    # attention_mask = encoded_review['attention_mask']
    # print(input_ids)
    # print(attention_mask)
