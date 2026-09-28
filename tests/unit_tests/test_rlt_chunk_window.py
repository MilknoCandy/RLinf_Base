# Copyright 2026 The RLinf Authors.

import numpy as np
import torch

from rlinf.data.datasets.openpi_rlinf.rlt_chunk_window import (
    RLTChunkWindowDataset,
    attach_rlt_chunk_window,
    chunk_window_indices,
)


def test_chunk_window_indices_stay_inside_the_episode():
    indices, valid = chunk_window_indices(25, 0, stride=10, length=3)
    assert indices == [5, 15, 25]
    assert valid == [True, True, True]

    indices, valid = chunk_window_indices(5, 0, stride=10, length=3)
    assert indices == [0, 0, 5]
    assert valid == [False, False, True]


class _Frames(torch.utils.data.Dataset):
    def __init__(self):
        self.episode_data_index = {
            "from": torch.tensor([0, 30]),
            "to": torch.tensor([30, 60]),
        }

    def __len__(self):
        return 60

    def __getitem__(self, index):
        value = np.float32(index)
        return {
            "image": {"cam": np.full((2, 2, 3), value, dtype=np.float32)},
            "state": np.asarray([value], dtype=np.float32),
            "actions": np.asarray([value], dtype=np.float32),
            "prompt": "keep",
        }


class _Wrap(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


def test_dataset_stacks_same_episode_chunks():
    dataset = RLTChunkWindowDataset(_Wrap(_Frames()), mem_len_max=3, stride=10)
    sample = dataset[25]
    assert sample["actions"].tolist() == [25]
    assert sample["prompt"] == "keep"
    np.testing.assert_array_equal(sample["rlt_hist_valid"], [True, True, True])
    cams = sample["rlt_hist"]["image"]["cam"][:, 0, 0, 0]
    np.testing.assert_array_equal(cams, np.asarray([5, 15, 25], dtype=np.float32))

    early = dataset[5]
    np.testing.assert_array_equal(early["rlt_hist_valid"], [False, False, True])
    early_cams = early["rlt_hist"]["image"]["cam"][:, 0, 0, 0]
    np.testing.assert_array_equal(
        early_cams, np.asarray([5, 5, 5], dtype=np.float32)
    )


class _TorchLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 4


class _OpenPITorchLoader:
    def __init__(self, dataset):
        self._data_loader = _TorchLoader(dataset)

    def __iter__(self):
        item = self._data_loader.dataset[25]
        yield {
            "image": item["image"],
            "state": item["state"],
            "actions": item["actions"][None, :],
            "rlt_hist": item["rlt_hist"],
            "rlt_hist_valid": item["rlt_hist_valid"],
        }


class _OpenPILoader:
    def __init__(self, dataset):
        self._data_loader = _OpenPITorchLoader(dataset)

    def data_config(self):
        return {"name": "pi05"}


def test_attach_yields_history_the_model_can_read():
    loader = attach_rlt_chunk_window(
        _OpenPILoader(_Wrap(_Frames())), mem_len_max=3, stride=10
    )
    batch = next(iter(loader))
    assert loader.data_config()["name"] == "pi05"
    assert len(loader) == 4
    assert batch["actions"].shape[-1] == 1
    np.testing.assert_array_equal(batch["rlt_hist_valid"], [True, True, True])
    assert "rlt_hist" not in batch["observation"]
    assert "actions" not in batch["observation"]
