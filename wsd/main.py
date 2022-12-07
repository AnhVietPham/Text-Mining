import json
import numpy as np
import copy
import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel, AutoTokenizer, RobertaConfig
from underthesea import word_tokenize
import argparse


class Classifier(nn.Module):
    def __init__(self, config, dropout_rate=0.1):
        super().__init__()

        self.dropout_1 = nn.Dropout(dropout_rate * 2)
        self.dense_1 = nn.Linear(config.hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dense_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        feature = self.dropout_1(feature)
        feature = self.dense_1(feature)
        feature = self.relu(feature)
        feature = self.dropout_2(feature)
        feature = self.dense_2(feature).view(-1)

        feature = self.sigmoid(feature)
        return feature


class ViHnBERT(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(ViHnBERT, self).__init__(config)
        self.args = args
        self.config = config
        # init backbone
        self.roberta = RobertaModel(config)

        self.classifier = Classifier(config, args.dropout_rate)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                start_token_idx=None,
                end_token_idx=None,
                labels=None):

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        features_bert = outputs[0]
        # Features of [CLS] tokens
        features_cls = features_bert[:, 0, :].unsqueeze(1)

        # Features of acronym tokens
        if start_token_idx is None or end_token_idx is None:
            raise Exception('Require start_token_idx and end_token_idx')
        list_mean_feature_acr = []
        for idx in range(features_bert.size()[0]):
            feature_acr = features_bert[idx, start_token_idx[idx]:end_token_idx[idx] + 1, :].unsqueeze(0)
            mean_feature_acr = torch.mean(feature_acr, 1, True)
            list_mean_feature_acr.append(mean_feature_acr)
        features_arc = torch.cat(list_mean_feature_acr, dim=0)

        # Concate featrues
        features = torch.cat([features_cls, features_arc], dim=2)

        logits = self.classifier(features)
        outputs = ((logits),) + outputs[2:]

        loss_fn = nn.BCELoss()
        total_loss = 0.0

        if labels is not None:
            total_loss = loss_fn(logits, labels)

        outputs = (total_loss,) + outputs

        return outputs


class InputExample(object):
    def __init__(self, guid, id, text, text_tokens, expansion, start_char_idx, length_acronym, start_token_idx,
                 end_token_idx, label) -> None:
        super().__init__()
        self.guid = guid
        self.id = id
        self.text = text
        self.text_tokens = text_tokens
        self.expansion = expansion
        self.start_char_idx = start_char_idx
        self.length_acronym = length_acronym
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, id, input_ids, attention_mask, token_type_ids, start_token_idx, end_token_idx, label, expansion):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.label = label
        self.expansion = expansion

        self.id = id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _read_file(input_file):
    """Reads json file"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def add_label_positive_sample(data: list):
    for idx, sample in enumerate(data):
        sample['text'] = sample['text'].lower()
        sample['label'] = 1
        sample['id'] = idx
    return data


def negative_data(positive_data: list, diction: dict) -> list:
    neg_data = []
    tmp = 0
    for sample in positive_data:
        try:
            acronym = sample["text"][sample["start_char_idx"]:sample["start_char_idx"] + sample['length_acronym']]
            list_neg_expansion = diction[acronym].copy()
            if len(list_neg_expansion) > 1:
                list_neg_expansion = list_neg_expansion
            for i in list_neg_expansion:
                neg_data.append(sample.copy())
                neg_data[tmp]["expansion"] = i
                neg_data[tmp]["label"] = 0  # pseudo negative samples
                tmp += 1
        except:
            print(sample)
            continue

    return neg_data


def get_example(data):
    dictionary = _read_file(
        "/Users/anhvietpham/Documents/cs/text-mining/vihealthbert-main/dataset/acrDrAid/dictionary.json")

    examples = []

    pos_data = add_label_positive_sample(data)
    neg_data = negative_data(pos_data, dictionary)
    examples.extend(neg_data)
    return _create_examples(examples)


def _create_examples(data):
    examples = []
    for i, example in enumerate(data):
        guid = ""
        id = example['id']
        # 1. Input text
        text = example['text']
        text_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    text_tokens.append(c)
                else:
                    text_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(text_tokens) - 1)
        # 2. Expansion of acr
        expansion = example['expansion']
        # 3. Position of acr and acr
        start_char_idx = example['start_char_idx']
        length_acronym = example['length_acronym']
        start_token_idx = char_to_word_offset[start_char_idx]
        end_token_idx = char_to_word_offset[start_char_idx + length_acronym - 1]
        # 4. Label
        label = example['label']
        examples.append(InputExample(
            guid=guid,
            id=id,
            text=text,
            text_tokens=text_tokens,
            expansion=expansion,
            start_char_idx=start_char_idx,
            length_acronym=length_acronym,
            start_token_idx=start_token_idx,
            end_token_idx=end_token_idx,
            label=label
        ))
    return examples


def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []

    for (ex_index, example) in enumerate(examples):
        orig_to_tok_index = []
        all_doc_tokens = []

        for (i, token) in enumerate(example.text_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        start_token_idx = orig_to_tok_index[example.start_token_idx]
        if len(orig_to_tok_index) == (example.end_token_idx + 1):
            end_token_idx = orig_to_tok_index[-1]
        else:
            end_token_idx = orig_to_tok_index[example.end_token_idx + 1] - 1

        input_ids = []

        input_ids += [cls_token]
        input_ids += all_doc_tokens
        input_ids += [sep_token]

        token_type_ids = [0] * len(input_ids)

        expansion = example.expansion
        expansion_tokens = tokenizer.tokenize(expansion)

        input_ids += expansion_tokens
        input_ids += [sep_token]

        token_type_ids += [1] * (len(expansion_tokens) + 1)

        attention_mask = [1] * len(input_ids)

        input_ids = tokenizer.convert_tokens_to_ids(input_ids)

        padding = max_seq_len - len(input_ids)

        if padding < 0:
            print('Ignore sample has length > 256 tokens')
            continue

        input_ids = input_ids + ([pad_token_id] * padding)
        attention_mask = attention_mask + [0] * padding
        token_type_ids = token_type_ids + [0] * padding
        assert len(input_ids) == len(attention_mask) == len(
            token_type_ids), "Error with input length {} vs attention mask length {}, token type length {}".format(
            len(input_ids), len(attention_mask), len(token_type_ids))
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        id = example.id
        label = example.label

        features.append(
            InputFeatures(
                id=id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                label=label,
                expansion=expansion
            )
        )
    return features


if __name__ == "__main__":
    text = "phình thoát vị nhỏ, đa tầng các đđ c4/c5 (nh), c5/c6 (thể cạnh trung tâm lệch trái), c6/c7 (nh) ra sau, đè ép nhẹ mặt trước bao màng cứng, gây hẹp ống sống, chèn ép nhẹ tủy cổ ngang mức nhưng chưa gây phù tủy (đường kính ống sống trước-sau)"
    word_tokenize = word_tokenize(text)
    print(word_tokenize)
    with open(
            '/Users/anhvietpham/Documents/cs/text-mining/vihealthbert-main/dataset/acrDrAid/dictionary.json') as json_file:
        data = json.load(json_file)
    a = ""
    for k, v in data.items():
        if k in word_tokenize:
            a = k
    input_text = " ".join(word_tokenize)
    json_data_list = []
    json_data = {
        "id": 0,
        "text": input_text,
        "start_char_idx": input_text.index(a),
        "length_acronym": len(a),
        "expansion": "",
        "label": 1
    }
    json_data_list.append(json_data)
    pretrain_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    examples = get_example(json_data_list)
    features = convert_examples_to_features(
        examples, 256, pretrain_tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.int64)
    all_start_token_idx = torch.tensor([f.start_token_idx for f in features], dtype=torch.int64)
    all_end_token_idx = torch.tensor([f.end_token_idx for f in features], dtype=torch.int64)
    all_label = torch.tensor([f.label for f in features], dtype=torch.float)

    all_id = torch.tensor([f.id for f in features], dtype=torch.long)
    all_expansion = [f.expansion for f in features]

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    args = parser.parse_args()
    config = RobertaConfig.from_pretrained('vinai/phobert-base', finetuning_task='')
    model = ViHnBERT.from_pretrained(
        '/Users/anhvietpham/Documents/cs/text-mining/vihealthbert-main/code/finetune/wsd/src/model-save',
        config=config,
        args=args,
    )

    inputs = {
        "input_ids": all_input_ids,
        "token_type_ids": all_token_type_ids,
        "attention_mask": all_attention_mask,
        "start_token_idx": all_start_token_idx,
        "end_token_idx": all_end_token_idx,
        "labels": all_label
    }
    outputs = model(**inputs)
    tmp_eval_loss, (slot_logits) = outputs[:2]
    slot_preds = slot_logits.detach().cpu().numpy()
    slot_preds = np.argmax(slot_preds)
    predict = {
        "sentence": text,
        "acronym": a,
        "start_char_idx": text.index(a),
        "length_acronym": len(a),
        "expansion": data[a][slot_preds]
    }
