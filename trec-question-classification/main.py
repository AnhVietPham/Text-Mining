def _generate_examples(filepath):
    examples = []
    with open(filepath, "rb") as f:
        for id_, row in enumerate(f):
            # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
            label, _, text = row.replace(b"\xf0",
                                         b" ").strip().decode().partition(" ")
            coarse_label, _, fine_label = label.partition(":")
            examples.append((id_, {
                "label-coarse": coarse_label,
                "label-fine": fine_label,
                "text": text,
            }))
    return examples


if __name__ == "__main__":
    train = _generate_examples(
        "/Users/sendo_mac/Documents/avp/Text-Mining/trec-question-classification/data/train_5500.label")
    test = _generate_examples(
        "/Users/sendo_mac/Documents/avp/Text-Mining/trec-question-classification/data/TREC_10.label")
    print(len(train))
    print(train[0])

    labels = [x['label-coarse'] for _, x in train]
    labels1 = [x['label-fine'] for _, x in train]
    print(labels)
    print(len(labels))
    print(labels1)
    print(len(labels1))

    set_labels = list(set(labels))
    print("------")
    print("labels:", set_labels)

    label2id = {x: i for i, x in enumerate(set_labels)}
    print("------")
    print("label2id", label2id)

    id2label = {i: x for i, x in enumerate(set_labels)}
    print("------")
    print("label2id", id2label)
