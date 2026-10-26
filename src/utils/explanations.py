from pathlib import Path
import json
import csv

import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_bag_explanation(model_output, fcm_centers=None, fcm_widths=None):
    explanation = {
        "bag_prediction": _to_numpy(model_output["bag_prediction"]),
        "bag_logits": _to_numpy(model_output["bag_logits"]),
        "attention_weights": _to_numpy(model_output["attention_weights"]),
        "instance_keyness_scores": _to_numpy(model_output["instance_keyness_scores"]),
        "rule_membership": _to_numpy(model_output["rule_membership"]),
        "normalized_rule_membership": _to_numpy(model_output["normalized_rule_membership"]),
        "rule_contributions": _to_numpy(model_output["rule_contributions"]),
        "rule_outputs": _to_numpy(model_output["rule_outputs"]),
        "bag_feature": _to_numpy(model_output["bag_feature"]),
    }

    if fcm_centers is not None:
        explanation["rule_centers"] = np.asarray(fcm_centers)
    if fcm_widths is not None:
        explanation["rule_widths"] = np.asarray(fcm_widths)

    return explanation


def save_bag_explanation(explanation, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **explanation)


def save_explanation_manifest(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def export_instance_report_csv(manifest_path, explanation_dir, output_path):
    manifest_path = Path(manifest_path)
    explanation_dir = Path(explanation_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    fieldnames = [
        "fold",
        "sample_position",
        "bag_index",
        "bag_label",
        "bag_prediction",
        "instance_index",
        "instance_keyness",
        "attention_weight",
        "top_rule",
        "top_rule_prob",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            explanation_path = explanation_dir / record["file"]
            explanation = np.load(explanation_path)
            bag_prediction = float(np.asarray(explanation["bag_prediction"]).reshape(-1)[0])
            bag_label = int(np.asarray(explanation["bag_label"]).reshape(-1)[0])
            bag_index = int(np.asarray(explanation["bag_index"]).reshape(-1)[0])
            attention_weights = np.asarray(explanation["attention_weights"])[0, :, 0]
            keyness_scores = np.asarray(explanation["instance_keyness_scores"])[0]
            rule_probs = np.asarray(explanation["normalized_rule_membership"])[0]

            for instance_index in range(len(keyness_scores)):
                instance_rule_probs = rule_probs[instance_index]
                top_rule = int(np.argmax(instance_rule_probs))
                writer.writerow(
                    {
                        "fold": record["fold"],
                        "sample_position": record["sample_position"],
                        "bag_index": bag_index,
                        "bag_label": bag_label,
                        "bag_prediction": bag_prediction,
                        "instance_index": instance_index,
                        "instance_keyness": float(keyness_scores[instance_index]),
                        "attention_weight": float(attention_weights[instance_index]),
                        "top_rule": top_rule,
                        "top_rule_prob": float(instance_rule_probs[top_rule]),
                    }
                )
