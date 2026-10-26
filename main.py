import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.trainers import run_single_fold_training
from src.utils import get_experiment_logger, save_bag_explanation, save_explanation_manifest


def build_config():
    return {
        "data_path": "data/newsgroups/alt_atheism+.mat",
        "num_rules": 6,
        "rule_output_dim": 1,
        "aggregation_method": "attention",
        "batch_size": 1,
        "lr": 0.001,
        "max_epoch": 100,
        "fcm_m": 2.0,
        "fcm_max_iter": 100,
        "patience": 100,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "hidden_dim": 128,
        "rule_temperature": 0.5,
        "use_competitive_scoring": True,
        "explanation_samples_per_fold": 3,
        "n_splits": 5,
        "random_state": 42,
    }


def _merge_dataset_entry(base_config, entry):
    merged_config = dict(base_config)
    merged_config.update(entry.get("training", {}))
    merged_config.update(entry.get("cross_validation", {}))
    merged_config.update(entry.get("output", {}))
    dataset_config = dict(entry.get("dataset", {}))
    if "path" in dataset_config:
        merged_config["data_path"] = dataset_config["path"]
    if "name" in dataset_config:
        merged_config["dataset_name"] = dataset_config["name"]
    return merged_config


def _summary_stats(values):
    if not values:
        return 0.0, 0.0
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_value, std_value


def _best_history_value(history, key):
    history_values = history.get(key, [])
    if not history_values:
        return 0.0
    return float(max(history_values))


def run_kfold_experiment(config):
    experiment_start = time.time()
    data_path = Path(config["data_path"])
    dataset_name = config.get("dataset_name", data_path.stem)
    output_root_dir = Path(config.get("output_root_dir", "."))
    run_name = config.get("run_name")
    run_root = output_root_dir / run_name if run_name else output_root_dir
    log_dir = run_root / "logs"
    model_dir = run_root / "artifacts" / "models" / dataset_name
    explanation_dir = run_root / "artifacts" / "explanations" / dataset_name

    logger = get_experiment_logger(
        f"{dataset_name}_kfold",
        log_dir=log_dir,
        log_filename=f"{dataset_name}_kfold.log",
    )

    logger.info("Loading dataset from %s", data_path)
    bags, labels = load_data(data_path)
    logger.info("Loaded %s bags with label distribution %s", len(bags), np.bincount(labels))

    splitter = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    explanation_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    fold_scores = []
    best_epochs = []
    explanation_records = []
    indices = np.arange(len(bags))
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, labels), start=1):
        logger.info("Starting fold %s/%s", fold_idx, config["n_splits"])

        train_bags = [bags[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_bags = [bags[i] for i in val_idx]
        val_labels = labels[val_idx]

        fold_logger = get_experiment_logger(
            f"{dataset_name}_fold_{fold_idx}",
            log_dir=log_dir,
            log_filename=f"{dataset_name}_fold_{fold_idx}.log",
        )

        fold_config = dict(config)
        fold_config["log_dir"] = str(log_dir)
        training_result = run_single_fold_training(
            fold_config,
            train_bags,
            train_labels,
            val_bags,
            val_labels,
            logger=fold_logger,
        )
        history = training_result.get("history", {})
        best_acc = float(training_result["best_acc"])
        best_f1 = _best_history_value(history, "val_f1")
        best_auc = _best_history_value(history, "val_auc")
        best_epoch = int(history.get("best_epoch", 0)) + 1
        training_time_sec = float(history.get("training_time", 0.0))
        model = training_result["model"]
        explanation_samples = training_result.get("explanation_samples", [])
        fold_scores.append(best_acc)
        best_epochs.append(best_epoch)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "acc": best_acc,
                "f1": best_f1,
                "auc": best_auc,
                "best_epoch": best_epoch,
                "training_time_sec": training_time_sec,
            }
        )

        model_path = model_dir / f"{dataset_name}_fold_{fold_idx}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info("Fold %s best acc: %.4f", fold_idx, best_acc)
        logger.info("Fold %s model saved to %s", fold_idx, model_path)

        for sample in explanation_samples:
            sample_position = int(sample["sample_position"][0])
            bag_index = int(val_idx[sample_position])
            sample["bag_index"] = np.asarray([bag_index], dtype=np.int64)
            explanation_filename = (
                f"{dataset_name}_fold_{fold_idx}_sample_{sample_position}_explanation.npz"
            )
            explanation_path = explanation_dir / explanation_filename
            save_bag_explanation(sample, explanation_path)
            explanation_records.append(
                {
                    "fold": fold_idx,
                    "sample_position": sample_position,
                    "bag_index": bag_index,
                    "bag_label": int(sample["bag_label"][0]),
                    "file": explanation_filename,
                }
            )
            logger.info(
                "Fold %s explanation sample %s saved to %s",
                fold_idx,
                sample_position,
                explanation_path,
            )

    mean_acc, std_acc = _summary_stats(fold_scores)
    mean_f1, std_f1 = _summary_stats([metric["f1"] for metric in fold_metrics])
    mean_auc, std_auc = _summary_stats([metric["auc"] for metric in fold_metrics])
    experiment_time_sec = float(time.time() - experiment_start)

    logger.info("K-fold finished for %s", dataset_name)
    logger.info("Fold scores: %s", [f"{score:.4f}" for score in fold_scores])
    logger.info("Mean acc: %.4f", mean_acc)
    logger.info("Std acc: %.4f", std_acc)

    manifest_path = explanation_dir / f"{dataset_name}_explanations_manifest.json"
    save_explanation_manifest(explanation_records, manifest_path)
    logger.info("Explanation manifest saved to %s", manifest_path)

    return {
        "dataset_name": dataset_name,
        "fold_metrics": fold_metrics,
        "mean_metrics": {
            "acc": mean_acc,
            "f1": mean_f1,
            "auc": mean_auc,
        },
        "std_metrics": {
            "acc": std_acc,
            "f1": std_f1,
            "auc": std_auc,
        },
        "best_epochs": best_epochs,
        "training_time_sec": experiment_time_sec,
        "artifacts_path": str(run_root / "artifacts"),
        "log_dir": str(log_dir),
        "explanation_dir": str(explanation_dir),
        "dataset": dataset_name,
        "fold_scores": fold_scores,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
    }


def main():
    config = build_config()
    return run_kfold_experiment(config)


if __name__ == "__main__":
    main()
