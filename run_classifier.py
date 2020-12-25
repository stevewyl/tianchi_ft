import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from transformers import EvalPrediction, Trainer, set_seed
from transformers import HfArgumentParser, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import DataTrainingArguments, ClassifierDataset

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def acc_and_f1(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"acc": acc, "f1": f1}

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    return acc_and_f1(preds, labels)

TASK_OUTPUT_MODE = {
    "ocemotion": "classification",
    "ocnli": "classification",
    "tnews": "classification"
}
TASK_NUM_LABELS = {
    "ocemotion": 7,
    "ocnli": 3,
    "tnews": 15
}

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if not training_args.do_train and (training_args.do_eval or training_args.do_predict):
        model_args.model_name_or_path = training_args.output_dir

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    # Get Datasets
    train_dataset, eval_dataset, test_dataset = None, None, None
    label_map = {}
    if (training_args.do_train or training_args.do_eval or training_args.do_predict):
        if training_args.do_train:
            train_dataset = ClassifierDataset(data_args, tokenizer, cache_dir=model_args.cache_dir)
            label_map = {i: label for i, label in enumerate(train_dataset.get_labels())}
        if training_args.do_eval or training_args.do_predict:
            eval_dataset = ClassifierDataset(data_args, tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            if not label_map:
                label_map = {i: label for i, label in enumerate(eval_dataset.get_labels())}
            if training_args.do_predict:
                test_dataset = ClassifierDataset(data_args, tokenizer, mode="test", cache_dir=model_args.cache_dir)
                training_args.do_eval = False
        logger.info("Label Map: " + str(label_map))
    else:
        logger.error("Must specify mode: do_train / do_eval / do_predict")
        sys.exit()

    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=TASK_NUM_LABELS[data_args.task_name],
        id2label=label_map,
        label2id={label: i for i, label in label_map.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir
    )

    logger.info(f"Load model from {model_args.model_name_or_path}")
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if TASK_OUTPUT_MODE[task_name] == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif TASK_OUTPUT_MODE[task_name] == "regression":
                preds = np.squeeze(p.predictions)
            return compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    callbacks = [
        EarlyStoppingCallback(5)
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

        predictions = trainer.predict(eval_dataset).predictions
        if TASK_OUTPUT_MODE[test_dataset.args.task_name] == "classification":
            predictions = np.argmax(predictions, axis=1)

        label_maps = eval_dataset.get_labels()
        y_tures = [label_maps[sample.label] for sample in eval_dataset]
        y_preds = [label_maps[pred] for pred in predictions]

        logger.info("Classification Report (Dev Set)")
        logger.info("\n" + classification_report(y_tures, y_preds, digits=4))

    if training_args.do_predict:
        logging.info("*** Predict ***")

        def do_predict(dataset, has_label=False):
            predictions = trainer.predict(test_dataset=dataset).predictions
            predictions = np.argmax(predictions, axis=1)

            if has_label:
                output_test_file = os.path.join(
                    data_args.data_dir, f"{dataset.args.task_name}_eval_wrong.txt")
                eval_data_file = os.path.join(data_args.data_dir, "dev.txt")
                label_maps = dataset.get_labels()
                y_tures = [label_maps[sample.label] for sample in dataset]
                y_preds = [label_maps[pred] for pred in predictions]
                texts = [line.split("\t", 1)[1] for line in open(eval_data_file)]
                if trainer.is_world_process_zero():
                    with open(output_test_file, "w") as writer:
                        logger.info("***** Eval results {} *****".format(dataset.args.task_name))
                        for y_true, y_pred, text in zip(y_tures, y_preds, texts):
                            if y_true != y_pred:
                                writer.write(f"{y_true}\t{y_pred}\t{text}")
            else:
                output_test_file = os.path.join(
                    training_args.output_dir, f"{dataset.args.task_name}_predict.json")
                if trainer.is_world_process_zero():
                    with open(output_test_file, "w") as writer:
                        logger.info("***** Test results {} *****".format(dataset.args.task_name))
                        for index, pred in enumerate(predictions):
                            label = dataset.get_labels()[pred]
                            line = {"id": index, "label": label}
                            writer.write(str(line).replace("\'", "\"") + "\n")

        do_predict(eval_dataset, True)
        do_predict(test_dataset)

    return eval_results


if __name__ == "__main__":
    main()
