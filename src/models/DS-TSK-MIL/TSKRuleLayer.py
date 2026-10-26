import numpy as np
import torch
import torch.nn as nn


class TSKRuleLayerFCM(nn.Module):
    def __init__(
        self,
        fcm_centers,
        fcm_widths,
        output_dim,
        rule_temperature=0.5,
        use_competitive_scoring=True,
    ):
        super(TSKRuleLayerFCM, self).__init__()
        self.num_rules = fcm_centers.shape[0]
        self.output_dim = output_dim
        self.rule_temperature = rule_temperature
        self.use_competitive_scoring = use_competitive_scoring
        input_dim = fcm_centers.shape[1]
        min_width = 1e-4

        self.means = nn.Parameter(
            torch.tensor(fcm_centers, dtype=torch.float32), requires_grad=False
        )
        self.vars = nn.Parameter(
            torch.tensor(np.maximum(fcm_widths, min_width) ** 2, dtype=torch.float32),
            requires_grad=False,
        )
        self.consequent = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(self.num_rules)]
        )

    def forward(self, x):
        batch_size, num_instances, input_dim = x.shape

        x_expanded = x.unsqueeze(2)
        means_expanded = self.means.unsqueeze(0).unsqueeze(0)
        vars_expanded = self.vars.unsqueeze(0).unsqueeze(0).clamp_min(1e-8)

        diff = x_expanded - means_expanded
        normalized_sqdist = (diff ** 2) / vars_expanded
        average_penalty = normalized_sqdist.mean(dim=-1)
        log_membership = -0.5 * average_penalty
        normalized_logits = log_membership
        if self.use_competitive_scoring and self.num_rules > 1:
            logit_mean = log_membership.mean(dim=2, keepdim=True)
            logit_std = log_membership.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
            normalized_logits = (log_membership - logit_mean) / logit_std
        normalized_membership = torch.softmax(
            normalized_logits / max(self.rule_temperature, 1e-6), dim=2
        )
        membership = torch.exp(log_membership).clamp_min(1e-10)

        outputs = []
        x_flat = x.reshape(-1, input_dim)
        for consequent in self.consequent:
            rule_output = consequent(x_flat)
            rule_output = rule_output.reshape(batch_size, num_instances, self.output_dim)
            outputs.append(rule_output.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)
        rule_activation = normalized_membership.unsqueeze(-1) * outputs
        weighted_output = torch.sum(rule_activation, dim=2)
        final_output = weighted_output
        instance_keyness_scores = final_output.mean(dim=-1)
        rule_contributions = rule_activation.sum(dim=-1)

        return {
            "instance_outputs": final_output,
            "rule_membership": membership,
            "normalized_rule_membership": normalized_membership,
            "rule_activation": rule_activation,
            "rule_outputs": outputs,
            "instance_keyness_scores": instance_keyness_scores,
            "rule_contributions": rule_contributions,
        }
