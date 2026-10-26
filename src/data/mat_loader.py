from pathlib import Path

import numpy as np
import scipy.io as sio


def _normalize_label(raw_label):
    label_array = np.asarray(raw_label)
    if label_array.size != 1:
        raise ValueError(f"Expected scalar label, got shape {label_array.shape}")
    return int(label_array.reshape(-1)[0])


def _strip_instance_label_column(bag):
    if bag.shape[1] <= 1:
        raise ValueError(f"Expected bag with at least 2 columns, got shape {bag.shape}")
    return bag[:, :-1]


def load_mat_bags(path, drop_instance_label=True):
    file_path = Path(path)
    mat = sio.loadmat(file_path)

    if "data" not in mat:
        raise KeyError(f"Missing 'data' key in {file_path}")

    data = mat["data"]
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected MATLAB 'data' array with shape (n_bags, 2+), got {data.shape} in {file_path}"
        )

    bags = []
    labels = []
    for row_idx, entry in enumerate(data):
        if len(entry) < 2:
            raise ValueError(f"Entry {row_idx} in {file_path} does not contain [X, y]")

        bag = np.asarray(entry[0], dtype=np.float64)
        if bag.ndim != 2:
            raise ValueError(
                f"Expected bag matrix at row {row_idx} in {file_path}, got shape {bag.shape}"
            )

        label = _normalize_label(entry[1])
        if drop_instance_label:
            bag = _strip_instance_label_column(bag)

        bags.append(bag)
        labels.append(label)

    binary_labels = np.asarray([1 if label > 0 else 0 for label in labels], dtype=np.int64)
    return bags, binary_labels


def load_data(path, drop_instance_label=True):
    return load_mat_bags(path, drop_instance_label=drop_instance_label)
