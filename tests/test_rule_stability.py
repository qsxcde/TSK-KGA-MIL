import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "src" / "models" / "DS-TSK-MIL"


def _load_module(module_filename):
    module_path = MODEL_DIR / module_filename
    spec = importlib.util.spec_from_file_location(
        f"test_{module_filename.replace('.', '_')}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FCMClustering = _load_module("FCMClustering.py").FCMClustering
TSKRuleLayerFCM = _load_module("TSKRuleLayer.py").TSKRuleLayerFCM


class RuleStabilityTest(unittest.TestCase):
    def test_fcm_widths_keep_reasonable_feature_scale(self):
        rng = np.random.RandomState(0)
        n_samples = 120
        high_var = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 8))
        low_var = rng.normal(loc=0.0, scale=1e-3, size=(n_samples, 4))
        x = np.concatenate([high_var, low_var], axis=1)

        fcm = FCMClustering(n_clusters=3, random_state=0, max_iter=20)
        _, widths = fcm.fit(x)

        global_std = x.std(axis=0)
        protected_low_var_widths = widths[:, 8:]

        self.assertGreater(np.median(protected_low_var_widths), 1e-3)
        self.assertGreater(
            np.median(protected_low_var_widths),
            0.1 * np.median(global_std[8:]),
        )

    def test_fcm_prototypes_remain_separated_on_standardised_high_dimensional_data(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=(300, 200))
        x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) < 1e-12, 1.0, x.std(axis=0))

        fcm = FCMClustering(n_clusters=6, random_state=42, max_iter=100)
        centers, _ = fcm.fit(x)

        pairwise = [
            np.linalg.norm(centers[i] - centers[j])
            for i in range(len(centers))
            for j in range(i + 1, len(centers))
        ]
        self.assertGreater(min(pairwise), 1e-3)

    def test_fcm_iteration_is_unchanged_when_prototypes_do_not_merge(self):
        rng = np.random.RandomState(7)
        x = np.vstack(
            [
                np.array([3.0, 3.0]) + 0.3 * rng.normal(size=(60, 2)),
                np.array([-3.0, -3.0]) + 0.3 * rng.normal(size=(60, 2)),
            ]
        )

        reference = FCMClustering(
            n_clusters=2, random_state=1, max_iter=100, prototype_tol=0.0
        )
        reference_centers, reference_widths = reference.fit(x)

        checked = FCMClustering(n_clusters=2, random_state=1, max_iter=100)
        centers, widths = checked.fit(x)

        self.assertFalse(checked.reinitialised_)
        self.assertTrue(np.array_equal(reference_centers, centers))
        self.assertTrue(np.array_equal(reference_widths, widths))

    def test_rule_membership_normalization_is_well_defined_in_high_dimension(self):
        rng = np.random.RandomState(1)
        batch_size = 1
        num_instances = 24
        input_dim = 200
        num_rules = 4

        x = rng.normal(size=(batch_size, num_instances, input_dim)).astype(np.float32)
        centers = rng.normal(scale=0.3, size=(num_rules, input_dim)).astype(np.float32)
        widths = np.full((num_rules, input_dim), 0.5, dtype=np.float32)

        layer = TSKRuleLayerFCM(centers, widths, output_dim=1)
        outputs = layer(torch.tensor(x))
        membership = outputs["rule_membership"].detach().cpu().numpy()

        clamp_ratio = float((membership <= 1.0000001e-10).mean())
        normalized_row_sums = outputs["normalized_rule_membership"].sum(dim=2)

        self.assertLess(clamp_ratio, 0.05)
        self.assertTrue(torch.allclose(normalized_row_sums, torch.ones_like(normalized_row_sums)))

    def test_lower_rule_temperature_makes_rule_distribution_sharper(self):
        centers = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        widths = np.ones((3, 2), dtype=np.float32)
        x = torch.tensor([[[0.2, 0.2], [1.2, 1.2]]], dtype=torch.float32)

        default_layer = TSKRuleLayerFCM(centers, widths, output_dim=1)
        sharp_layer = TSKRuleLayerFCM(centers, widths, output_dim=1, rule_temperature=0.25)

        default_entropy = self._normalized_entropy(
            default_layer(x)["normalized_rule_membership"].detach().cpu().numpy()
        )
        sharp_entropy = self._normalized_entropy(
            sharp_layer(x)["normalized_rule_membership"].detach().cpu().numpy()
        )

        self.assertLess(sharp_entropy, default_entropy)

    def test_competitive_rule_scoring_increases_top_rule_margin(self):
        centers = np.array(
            [
                [0.0, 0.0],
                [0.6, 0.6],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
        widths = np.ones((3, 2), dtype=np.float32)
        x = torch.tensor([[[0.2, 0.2]]], dtype=torch.float32)

        baseline_layer = TSKRuleLayerFCM(
            centers,
            widths,
            output_dim=1,
            use_competitive_scoring=False,
        )
        competitive_layer = TSKRuleLayerFCM(
            centers,
            widths,
            output_dim=1,
            use_competitive_scoring=True,
        )

        baseline_probs = baseline_layer(x)["normalized_rule_membership"].detach().cpu().numpy()[0, 0]
        competitive_probs = competitive_layer(x)["normalized_rule_membership"].detach().cpu().numpy()[0, 0]

        baseline_margin = np.sort(baseline_probs)[-1] - np.sort(baseline_probs)[-2]
        competitive_margin = np.sort(competitive_probs)[-1] - np.sort(competitive_probs)[-2]

        self.assertGreater(competitive_margin, baseline_margin)

    @staticmethod
    def _normalized_entropy(rule_probs):
        clipped = np.clip(rule_probs, 1e-12, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=2)
        return float((entropy / np.log(rule_probs.shape[2])).mean())


if __name__ == "__main__":
    unittest.main()
