"""
Fine-tuning script for the receipt NER model with MLflow tracking.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
from datasets import Dataset
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from auto_labeling import BIO_LABELS, load_receipt_data, run_auto_labeling
from settings import (
    DB_PATH,
    FINETUNED_MODEL_DIR,
    FIX_MISTRAL_REGEX,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(FINETUNED_MODEL_DIR)


def configure_mlflow() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    artifact_root = Path(MLFLOW_ARTIFACTS_DIR).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id

    if MLFLOW_TRACKING_URI.startswith(("http://", "https://")):
        return client.create_experiment(name=MLFLOW_EXPERIMENT_NAME)

    artifact_location = (artifact_root / MLFLOW_EXPERIMENT_NAME).resolve().as_uri()
    return client.create_experiment(
        name=MLFLOW_EXPERIMENT_NAME,
        artifact_location=artifact_location,
    )


class NERFineTuner:
    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer = None
        self.model = None
        self.label2id = {label: idx for idx, label in enumerate(BIO_LABELS)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def load_base_model(self) -> None:
        logger.info("Loading base model from %s", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            use_fast=True,
            fix_mistral_regex=FIX_MISTRAL_REGEX,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_dir),
            ignore_mismatched_sizes=True,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(BIO_LABELS),
        )

    def load_training_data(
        self,
        min_samples: int,
        last_days: int | None = None,
        json_path: str | None = None,
    ) -> Dict:
        if json_path:
            with open(json_path, "r", encoding="utf-8") as file:
                return json.load(file)

        result = run_auto_labeling(
            db_path=str(DB_PATH),
            min_samples=min_samples,
            last_days=last_days,
            export_formats=["training"],
        )
        if not result:
            raise ValueError("Could not prepare training data from the receipt database.")

        training_path = next(
            (Path(path) for path in result["exported_files"] if "training" in Path(path).name),
            None,
        )
        if training_path is None or not training_path.exists():
            raise ValueError("Training dataset was not exported.")

        with open(training_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )

        labels = []
        for batch_index, label_ids in enumerate(examples["ner_tag_ids"]):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            previous_word_idx = None
            aligned = []

            for word_idx in word_ids:
                if word_idx is None:
                    aligned.append(-100)
                elif word_idx != previous_word_idx:
                    aligned.append(label_ids[word_idx])
                else:
                    aligned.append(-100)
                previous_word_idx = word_idx

            labels.append(aligned)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_datasets(self, training_data: Dict) -> Tuple[Dataset, Dataset]:
        rows = training_data["data"]
        logger.info("Dataset size: %s", len(rows))

        train_rows, eval_rows = train_test_split(rows, test_size=0.2, random_state=42)
        train_columns = Dataset.from_list(train_rows).column_names
        eval_columns = Dataset.from_list(eval_rows).column_names

        train_dataset = Dataset.from_list(train_rows).map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_columns,
        )
        eval_dataset = Dataset.from_list(eval_rows).map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=eval_columns,
        )
        return train_dataset, eval_dataset

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for pred_row, label_row in zip(predictions, labels):
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                true_labels.append(self.id2label[label_id])
                true_predictions.append(self.id2label[pred_id])

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            average="weighted",
            zero_division=0,
        )
        entity_precision, entity_recall, entity_f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            labels=["B", "I"],
            average="micro",
            zero_division=0,
        )
        accuracy = accuracy_score(true_labels, true_predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
            "accuracy": accuracy,
        }

    def train(self, training_data: Dict, hyperparams: Dict) -> Dict:
        experiment_id = configure_mlflow()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info("Started MLflow run %s", run_id)

            mlflow.log_params(
                {
                    "model_name": str(self.model_dir),
                    "learning_rate": hyperparams["learning_rate"],
                    "epochs": hyperparams["epochs"],
                    "batch_size": hyperparams["batch_size"],
                    "weight_decay": hyperparams["weight_decay"],
                    "logging_steps": hyperparams["logging_steps"],
                    "dataset_size": len(training_data["data"]),
                    "labels": ", ".join(BIO_LABELS),
                }
            )

            train_dataset, eval_dataset = self.prepare_datasets(training_data)
            data_collator = DataCollatorForTokenClassification(self.tokenizer)

            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=hyperparams["epochs"],
                per_device_train_batch_size=hyperparams["batch_size"],
                per_device_eval_batch_size=hyperparams["batch_size"],
                learning_rate=hyperparams["learning_rate"],
                weight_decay=hyperparams["weight_decay"],
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="entity_f1",
                greater_is_better=True,
                save_total_limit=2,
                logging_steps=max(1, int(hyperparams["logging_steps"])),
                logging_strategy="steps",
                report_to=["mlflow"],
                fp16=torch.cuda.is_available(),
                seed=42,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            train_result = trainer.train()
            eval_metrics = trainer.evaluate()

            mlflow.log_metrics(
                {
                    "train_loss": float(train_result.training_loss),
                    "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
                    "train_samples_per_second": float(
                        train_result.metrics.get("train_samples_per_second", 0.0)
                    ),
                    "epochs_trained": float(train_result.metrics.get("epoch", hyperparams["epochs"])),
                    "eval_loss": float(eval_metrics["eval_loss"]),
                    "eval_precision": float(eval_metrics["eval_precision"]),
                    "eval_recall": float(eval_metrics["eval_recall"]),
                    "eval_f1_score": float(eval_metrics["eval_f1_score"]),
                    "eval_entity_precision": float(eval_metrics["eval_entity_precision"]),
                    "eval_entity_recall": float(eval_metrics["eval_entity_recall"]),
                    "eval_entity_f1": float(eval_metrics["eval_entity_f1"]),
                    "eval_accuracy": float(eval_metrics["eval_accuracy"]),
                }
            )

            trainer.save_model(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))
            mlflow.log_artifacts(str(self.output_dir), artifact_path="model")

            metadata = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": hyperparams,
                "metrics": {
                    "train_loss": float(train_result.training_loss),
                    "eval_loss": float(eval_metrics["eval_loss"]),
                    "precision": float(eval_metrics["eval_precision"]),
                    "recall": float(eval_metrics["eval_recall"]),
                    "f1_score": float(eval_metrics["eval_f1_score"]),
                    "entity_precision": float(eval_metrics["eval_entity_precision"]),
                    "entity_recall": float(eval_metrics["eval_entity_recall"]),
                    "entity_f1": float(eval_metrics["eval_entity_f1"]),
                    "accuracy": float(eval_metrics["eval_accuracy"]),
                },
                "dataset_size": len(training_data["data"]),
                "tracking_uri": MLFLOW_TRACKING_URI,
            }

            metadata_path = self.output_dir / "training_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)

            return metadata

    def print_report(self, metadata: Dict) -> None:
        metrics = metadata["metrics"]
        print("\n" + "=" * 68)
        print("NER FINE-TUNING RESULTS")
        print("=" * 68)
        print(f"Run ID: {metadata['run_id']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Dataset size: {metadata['dataset_size']}")
        print(f"Tracking URI: {metadata['tracking_uri']}")
        print()
        print("Hyperparameters:")
        for key, value in metadata["hyperparameters"].items():
            print(f"  {key:16s}: {value}")
        print()
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key:16s}: {value:.4f}")
        print("=" * 68 + "\n")


def check_data_sufficiency(min_samples: int = 100, last_days: int | None = None) -> bool:
    data = load_receipt_data(db_path=str(DB_PATH), min_samples=min_samples, last_days=last_days)
    if len(data) < min_samples:
        logger.warning("Not enough data for fine-tuning: %s < %s", len(data), min_samples)
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the NER model with MLflow tracking.")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--last-days", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--training-json", type=str, help="Path to an exported training JSON file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not check_data_sufficiency(args.min_samples, args.last_days):
        print("\nNot enough data for fine-tuning.")
        return

    if args.dry_run:
        print("\nDry-run completed successfully. Training data is available.")
        return

    finetuner = NERFineTuner(Path(args.model_dir), Path(args.output_dir))
    finetuner.load_base_model()

    training_data = finetuner.load_training_data(
        min_samples=args.min_samples,
        last_days=args.last_days,
        json_path=args.training_json,
    )

    hyperparams = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
    }

    metadata = finetuner.train(training_data, hyperparams)
    finetuner.print_report(metadata)

    report_path = Path(args.output_dir) / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    logger.info("Saved metrics report to %s", report_path)


if __name__ == "__main__":
    main()
