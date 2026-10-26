import torch
import torch.nn as nn


class Aggregation(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, mode="attention"):
        super(Aggregation, self).__init__()
        self.mode = mode

        if self.mode == "attention":
            self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
            self.attention_w = nn.Linear(hidden_dim, 1)
            self.score_projection = nn.Linear(1, 1, bias=False)
        else:
            self.attention_V = None
            self.attention_U = None
            self.attention_w = None
            self.score_projection = None

    def forward(self, features, scores=None):
        if self.mode == "attention":
            attention_v = self.attention_V(features)
            attention_u = self.attention_U(features)
            gated_features = attention_v * attention_u
            attention_raw = self.attention_w(gated_features)
            if scores is not None:
                if scores.dim() == 2:
                    scores = scores.unsqueeze(-1)
                attention_raw = attention_raw + self.score_projection(scores)
            return torch.softmax(attention_raw, dim=1)

        if self.mode == "max":
            if scores is None:
                raise ValueError("Max pooling requires 'scores' input.")
            if scores.dim() == 3:
                scores = scores.squeeze(-1)
            _, max_indices = torch.max(scores, dim=1, keepdim=True)
            alpha = torch.zeros_like(scores)
            alpha.scatter_(1, max_indices, 1.0)
            return alpha.unsqueeze(-1)

        if self.mode == "min":
            if scores is None:
                raise ValueError("Min pooling requires 'scores' input.")
            if scores.dim() == 3:
                scores = scores.squeeze(-1)
            _, min_indices = torch.min(scores, dim=1, keepdim=True)
            alpha = torch.zeros_like(scores)
            alpha.scatter_(1, min_indices, 1.0)
            return alpha.unsqueeze(-1)

        if self.mode == "mean":
            batch_size, num_instances, _ = features.size()
            alpha = torch.ones(batch_size, num_instances, 1, device=features.device)
            return alpha / num_instances

        raise ValueError(f"Unknown aggregation mode: {self.mode}")
