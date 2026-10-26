import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data import MILDataset
from src.models import FCMClustering, MILModelFCM
from src.utils import collect_bag_explanation, get_experiment_logger

from .TSKMILTrainer import TSKMILTrainer


def _build_explanation_samples(model, val_bags_scaled, val_labels, device, fcm_centers, fcm_widths, max_samples):
    explanation_samples = []
    if not val_bags_scaled:
        return explanation_samples

    model.eval()
    num_samples = min(max_samples, len(val_bags_scaled))
    with torch.no_grad():
        for sample_idx in range(num_samples):
            sample_bag = val_bags_scaled[sample_idx]
            sample_tensor = torch.tensor(sample_bag, dtype=torch.float32).unsqueeze(0).to(device)
            explanation = collect_bag_explanation(
                model(sample_tensor, return_details=True),
                fcm_centers=fcm_centers,
                fcm_widths=fcm_widths,
            )
            explanation["bag_label"] = np.asarray([val_labels[sample_idx]], dtype=np.int64)
            explanation["sample_position"] = np.asarray([sample_idx], dtype=np.int64)
            explanation_samples.append(explanation)

    return explanation_samples


def run_single_fold_training(config, train_bags, train_labels, val_bags, val_labels, logger=None):
    log_dir = config.get("log_dir", "logs")
    logger = logger or get_experiment_logger("single_fold_training", log_dir=log_dir)
    train_batch_size = 1
    val_batch_size = 1

    all_train_instances = np.vstack(train_bags)
    scaler = StandardScaler()
    all_train_instances_scaled = scaler.fit_transform(all_train_instances)

    train_bags_scaled = []
    start_idx = 0
    for bag in train_bags:
        n_instances = bag.shape[0]
        bag_scaled = all_train_instances_scaled[start_idx:start_idx + n_instances]
        train_bags_scaled.append(bag_scaled)
        start_idx += n_instances

    val_bags_scaled = []
    for bag in val_bags:
        val_bags_scaled.append(scaler.transform(bag))

    fcm = FCMClustering(
        n_clusters=config["num_rules"],
        m=config["fcm_m"],
        max_iter=config["fcm_max_iter"],
        random_state=42,
    )
    fcm_centers, fcm_widths = fcm.fit(all_train_instances_scaled)
    if getattr(fcm, "reinitialised_", False):
        logger.info(
            "FCM prototypes coincided for this fold; re-estimated with K-Means"
        )

    train_dataset = MILDataset(train_bags_scaled, train_labels)
    val_dataset = MILDataset(val_bags_scaled, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    logger.info(
        "Using DKMIL-style bag-wise batch sizes - train: %s, val: %s",
        train_batch_size,
        val_batch_size,
    )

    model = MILModelFCM(
        input_dim=all_train_instances.shape[1],
        num_rules=config["num_rules"],
        rule_output_dim=config["rule_output_dim"],
        aggregation_method=config["aggregation_method"],
        hidden_dim=config["hidden_dim"],
        rule_temperature=config.get("rule_temperature", 0.5),
        use_competitive_scoring=config.get("use_competitive_scoring", True),
    )
    model.setup_fcm_antecedents(fcm_centers, fcm_widths)
    model = model.to(config["device"])

    trainer = TSKMILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["lr"],
        max_epoch=config["max_epoch"],
        device=config["device"],
        patience=config["patience"],
        logger=logger,
    )

    best_acc, train_history = trainer.train()

    explanation_samples = _build_explanation_samples(
        model=model,
        val_bags_scaled=val_bags_scaled,
        val_labels=val_labels,
        device=config["device"],
        fcm_centers=fcm_centers,
        fcm_widths=fcm_widths,
        max_samples=config.get("explanation_samples_per_fold", 3),
    )
    explanation_cache = explanation_samples[0] if explanation_samples else None

    return {
        "best_acc": best_acc,
        "model": model,
        "history": train_history,
        "scaler": scaler,
        "fcm_centers": fcm_centers,
        "fcm_widths": fcm_widths,
        "explanation_cache": explanation_cache,
        "explanation_samples": explanation_samples,
    }
