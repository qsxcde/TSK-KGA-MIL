import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def _load_local_module(module_filename):
    module_path = Path(__file__).resolve().parent / module_filename
    spec = importlib.util.spec_from_file_location(
        f"ds_tsk_mil_{module_filename.replace('.', '_')}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


Aggregation = _load_local_module("AggregationLayer.py").Aggregation
TSKRuleLayerFCM = _load_local_module("TSKRuleLayer.py").TSKRuleLayerFCM


class MILModelFCM(nn.Module):
    def __init__(
        self,
        input_dim,
        num_rules,
        rule_output_dim,
        hidden_dim,
        aggregation_method="attention",
        rule_temperature=0.5,
        use_competitive_scoring=True,
    ):
        super(MILModelFCM, self).__init__()

        self.input_dim = input_dim
        self.num_rules = num_rules
        self.rule_output_dim = rule_output_dim
        self.hidden_dim = hidden_dim
        self.rule_temperature = rule_temperature
        self.use_competitive_scoring = use_competitive_scoring

        self.tsk_layer = None
        instance_semantic_dim = input_dim + rule_output_dim + 1 + num_rules
        self.instance_projection = nn.Sequential(
            nn.Linear(instance_semantic_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_net = Aggregation(
            input_dim=hidden_dim, hidden_dim=hidden_dim, mode=aggregation_method
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def setup_fcm_antecedents(self, fcm_centers, fcm_widths):
        self.tsk_layer = TSKRuleLayerFCM(
            fcm_centers,
            fcm_widths,
            self.rule_output_dim,
            rule_temperature=self.rule_temperature,
            use_competitive_scoring=self.use_competitive_scoring,
        )

    def forward(self, x, return_details=None):
        if self.tsk_layer is None:
            raise ValueError("Must call setup_fcm_antecedents() before forward pass")

        tsk_outputs = self.tsk_layer(x)
        instance_outputs = tsk_outputs["instance_outputs"]
        instance_keyness_scores = tsk_outputs["instance_keyness_scores"].unsqueeze(-1)
        normalized_rule_membership = tsk_outputs["normalized_rule_membership"]

        instance_semantic_feature = torch.cat(
            [x, instance_outputs, instance_keyness_scores, normalized_rule_membership], dim=-1
        )
        aggregated_instance_feature = self.instance_projection(instance_semantic_feature)
        alpha = self.attention_net(aggregated_instance_feature, instance_keyness_scores)
        bag_feature = torch.sum(alpha * aggregated_instance_feature, dim=1)
        classifier_output = self.classifier(bag_feature)
        bag_prediction = torch.sigmoid(classifier_output)

        result = {
            "bag_prediction": bag_prediction.squeeze(-1),
            "bag_logits": classifier_output.squeeze(-1),
            "attention_weights": alpha,
            "bag_feature": bag_feature,
            "instance_semantic_feature": instance_semantic_feature,
            "aggregated_instance_feature": aggregated_instance_feature,
            **tsk_outputs,
        }

        if return_details:
            return result

        return result["bag_prediction"]
