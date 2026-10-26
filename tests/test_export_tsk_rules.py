import numpy as np
import pytest
import torch

from src.analysis.export_tsk_rules import extract_tsk_rules, save_tsk_rule_values


def test_extract_tsk_rules_returns_centers_widths_and_consequents():
    state_dict = {
        "tsk_layer.means": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "tsk_layer.vars": torch.tensor([[4.0, 9.0], [16.0, 25.0]]),
        "tsk_layer.consequent.0.weight": torch.tensor([[0.1, 0.2]]),
        "tsk_layer.consequent.0.bias": torch.tensor([0.3]),
        "tsk_layer.consequent.1.weight": torch.tensor([[1.1, 1.2]]),
        "tsk_layer.consequent.1.bias": torch.tensor([1.3]),
    }

    rules = extract_tsk_rules(state_dict)

    np.testing.assert_allclose(rules["centers"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(rules["widths"], [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(
        rules["consequent_weights"],
        [[[0.1, 0.2]], [[1.1, 1.2]]],
    )
    np.testing.assert_allclose(rules["consequent_biases"], [[0.3], [1.3]])


def test_extract_tsk_rules_rejects_missing_tsk_parameters():
    with pytest.raises(ValueError, match="tsk_layer.means"):
        extract_tsk_rules({"classifier.0.weight": torch.tensor([[1.0]])})


def test_save_tsk_rule_values_writes_readable_parameter_files(tmp_path):
    rules = {
        "centers": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "widths": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "consequent_weights": np.array(
            [[[0.1, 0.2]], [[1.1, 1.2]]],
            dtype=np.float32,
        ),
        "consequent_biases": np.array([[0.3], [1.3]], dtype=np.float32),
    }

    output_dir = save_tsk_rule_values(rules, tmp_path / "sample_tsk_rules")

    assert output_dir.is_dir()
    np.testing.assert_allclose(np.loadtxt(output_dir / "centers.csv", delimiter=","), rules["centers"])
    np.testing.assert_allclose(np.loadtxt(output_dir / "widths.csv", delimiter=","), rules["widths"])
    np.testing.assert_allclose(
        np.loadtxt(output_dir / "consequent_weights.csv", delimiter=","),
        [[0.1, 0.2], [1.1, 1.2]],
    )
    np.testing.assert_allclose(
        np.loadtxt(output_dir / "consequent_biases.csv", delimiter=","),
        [0.3, 1.3],
    )
    assert (output_dir / "metadata.json").exists()
