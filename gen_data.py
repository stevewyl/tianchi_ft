import os
from collections import Counter

import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split

# TODO: 数据增强

def read_data(filename, has_label, dataset_name):
    dataset_name = dataset_name.lower()
    # df = pd.read_csv(filename, sep="\t")
    samples = [line.strip().split("\t") for line in open(filename, "r")]
    text_a, text_b = [], []
    labels = []
    for sample in samples:
        if has_label:
            labels.append(sample[-1])
        text_a.append(sample[1])
        if dataset_name == "ocnli":
            text_b.append(sample[2])
    if dataset_name == "ocnli":
        texts = [[x1, x2] for x1, x2 in zip(text_a, text_b)]
    else:
        texts = text_a
    return texts, labels

def save_data(filename, corpus, dataset_name):
    with open(filename, "w") as f:
        if dataset_name == "ocnli":
            for label, (text_a, text_b) in corpus:
                f.write(f"{label}\t{text_a}\t{text_b}\n")
        else:
            for label, text in corpus:
                f.write(f"{label}\t{text}\n")

def save_list(filename, input_list):
    with open(filename, "w") as f:
        for item in input_list:
            if isinstance(item, list):
                item = '\t'.join(item)
            f.write(f"{item}\n")

if __name__ == '__main__':

    for name in ["OCEMOTION", "OCNLI", "TNEWS"]:
        train_fn = f"data/{name}_train1128.csv"
        test_fn = f"data/{name}_a.csv"

        texts, labels = read_data(train_fn, True, name)
        test_corpus, _ = read_data(test_fn, False, name)

        x_train, x_valid, y_train, y_valid = train_test_split(
            texts, labels, stratify=labels, test_size=0.1, random_state=2020)

        train_corpus = [[y, x] for x, y in zip(x_train, y_train)]
        valid_corpus = [[y, x] for x, y in zip(x_valid, y_valid)]

        name = name.lower()
        data_dir = f"data/{name}"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        save_data(f"{data_dir}/train.txt", train_corpus, name)
        save_data(f"{data_dir}/dev.txt", valid_corpus, name)

        save_list(f"{data_dir}/test.txt", test_corpus)
        save_list(f"{data_dir}/label.txt", sorted(list(set(labels))))

        print(f"{name} dataset train/dev/test: {len(train_corpus)}/{len(valid_corpus)}/{len(test_corpus)}")
        seq_lens = [len(text) + 1 if isinstance(text, str) else len(text[0]) + len(text[1]) + 2 for text in texts + test_corpus]
        sl_50 = np.percentile(seq_lens, 50)
        sl_90 = np.percentile(seq_lens, 90)
        sl_99 = np.percentile(seq_lens, 99)
        print(f"Sequence length 50%/90%/99%: {sl_50}/{sl_90}/{sl_99}")

        print("Label Distribution:")
        pprint(Counter(labels))
