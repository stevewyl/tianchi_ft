import argparse

import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertForSequenceClassification

def get_max_seq_len(corpus):
    lengths = max([len(doc) for doc in corpus])
    return lengths + 2

def encode(corpus, tokenizer, max_seq_len, device):
    input_ids = []
    pad_token_id = tokenizer.token_to_id("[PAD]")
    for item in tokenizer.encode_batch(corpus):
        input_id = item.ids
        if len(input_id) <= max_seq_len:
            input_ids.append(input_id + [pad_token_id] * (max_seq_len - len(input_id)))
        else:
            input_ids.append(input_id[:max_seq_len])
    return torch.tensor(input_ids).to(device)

def save_results(corpus, labels, output_file="./test_predictions.txt"):
    with open(output_file, "w") as fw:
        for text, label in zip(corpus, labels):
            fw.write(f"{text}\t{label}\n")

def get_arg_parser():
    arg_parser = argparse.ArgumentParser(description="Transformers Model Predcition")
    arg_parser.add_argument("-t", "--task_name", required=True, help="Task Name")
    arg_parser.add_argument("-f", "--test_file", required=True, help="The file contains test corpus")
    arg_parser.add_argument("-d", "--device", default="cpu", help="Device used for inference")
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arg_parser()

    model_dir = f"models/{args.task_name}"
    tokenizer = BertWordPieceTokenizer(vocab=f"{model_dir}/vocab.txt")

    corpus = [line.strip() for line in open(args.test_file)]
    label_map = {i: label.strip() for i, label in enumerate(open(f"{model_dir}/label.txt"))}

    model = BertForSequenceClassification.from_pretrained(model_dir).to(args.device)
    max_seq_len = get_max_seq_len(corpus)
    with torch.no_grad():
        input_ids = encode(corpus, tokenizer, max_seq_len, args.device)
        predictions = np.argmax(model(input_ids)[0].cpu().numpy(), axis=1)
    predictions = [label_map[label_id] for label_id in predictions]
    save_results(corpus, predictions)
