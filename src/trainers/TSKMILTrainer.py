import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TSKMILTrainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, max_epoch=100,
                 device='mps', patience=100, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.max_epoch = max_epoch
        self.device = device
        self.patience = patience
        self.logger = logger

        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []
        self.val_auc_history = []

        self.best_val_acc = 0
        self.best_model_state = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0

        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            bag_prediction = outputs["bag_prediction"] if isinstance(outputs, dict) else outputs

            if bag_prediction.dim() == 0:
                bag_prediction = bag_prediction.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            loss = self.criterion(bag_prediction, labels.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                bag_prediction = outputs["bag_prediction"] if isinstance(outputs, dict) else outputs

                if bag_prediction.dim() == 0:
                    bag_prediction = bag_prediction.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                loss = self.criterion(bag_prediction, labels.float())
                total_loss += loss.item() * labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(bag_prediction.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        binary_preds = (all_preds > 0.5).astype(int)

        val_loss = total_loss / len(self.val_loader.dataset)
        val_acc = accuracy_score(all_labels, binary_preds)
        val_f1 = f1_score(all_labels, binary_preds, zero_division=0)
        val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

        return val_loss, val_acc, val_f1, val_auc

    def train(self):
        self._log("Starting training...")
        start_time = time.time()

        for epoch in range(self.max_epoch):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, val_auc = self.validate()

            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            self.val_f1_history.append(val_f1)
            self.val_auc_history.append(val_auc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                self._log(
                    f'Epoch {epoch + 1}/{self.max_epoch}: '
                    f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}'
                )

            if self.epochs_without_improvement >= self.patience:
                self._log(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no val_acc improvement for {self.patience} epochs."
                )
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        training_time = time.time() - start_time
        self._log(f'训练完成! 耗时: {training_time:.2f}秒')
        self._log(f'最佳验证准确率: {self.best_val_acc:.4f} (第 {self.best_epoch + 1} 轮)')

        train_history = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'training_time': training_time,
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'val_acc': self.val_acc_history,
            'val_f1': self.val_f1_history,
            'val_auc': self.val_auc_history
        }

        return self.best_val_acc, train_history


FCMTSKTrainer = TSKMILTrainer
