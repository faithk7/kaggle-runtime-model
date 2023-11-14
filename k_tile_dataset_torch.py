import collections
import glob
import hashlib
import io
import os
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch_geometric.data import Data

# 0 means not using the toy value, whereas other nnumbers mean using the toy value
_TOY_DATA_VALUE = 100


class NpzDataset(namedtuple("NpzDataset", ["train", "validation", "test"])):
    """Contains all partitions of the dataset."""

    @property
    def num_ops(self):
        return (
            int(
                torch.max(
                    torch.tensor(
                        [
                            torch.max(self.train.node_opcode),
                            torch.max(self.validation.node_opcode),
                            torch.max(self.test.node_opcode),
                        ]
                    )
                ).item()
            )
            + 1
        )

    def _get_normalizer(
        self, feature_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_feat = torch.max(feature_matrix, dim=0, keepdim=True).values
        min_feat = torch.min(feature_matrix, dim=0, keepdim=True).values
        # print the results
        print(torch.sum(min_feat[0] != max_feat[0]))
        # print the results
        return (min_feat[0] != max_feat[0], min_feat, max_feat)

    def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat):
        feature_matrix = feature_matrix[:, used_columns]
        min_feat = min_feat[:, used_columns]
        max_feat = max_feat[:, used_columns]
        return (feature_matrix - min_feat) / (max_feat - min_feat)

    def normalize(self):
        """Removes constant features and normalizes remaining onto [0, 1].

        The statistics are computed only from train partition then applied to all
        partitions {train, test, validation}.
        """
        # print the size of train partition
        print("before normalize", self.train.node_feat.shape)
        # Get normalizer arguments from train partition
        normalizer_args = self._get_normalizer(self.train.node_feat)
        # Apply normalizer to train, validation and test partitions
        self.train.node_feat = self._apply_normalizer(
            self.train.node_feat, *normalizer_args
        )
        self.validation.node_feat = self._apply_normalizer(
            self.validation.node_feat, *normalizer_args
        )
        self.test.node_feat = self._apply_normalizer(
            self.test.node_feat, *normalizer_args
        )

        # Get normalizer arguments from train partition
        normalizer_args = self._get_normalizer(self.train.config_feat)
        # Apply normalizer to train, validation and test partitions
        self.train.config_feat = self._apply_normalizer(
            self.train.config_feat, *normalizer_args
        )
        self.validation.config_feat = self._apply_normalizer(
            self.validation.config_feat, *normalizer_args
        )
        self.test.config_feat = self._apply_normalizer(
            self.test.config_feat, *normalizer_args
        )


class NpzDatasetPartition:
    """Holds one data partition (train, test, validation) on device memory."""

    def __init__(self):
        # Populated in `add_npz_file()`.
        self._data_dict: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
        # prepend with 0 to prep for cumsum.
        self._num_edges: List[int] = [0]
        self._num_configs: List[int] = [0]  # ^^
        self._num_nodes: List[int] = [0]  # ^^

        # indexed by node_ranges.
        self.node_feat: "torch.Tensor | None" = None
        self.node_opcode: "torch.Tensor | None" = None  # ^^
        # indexed by edge_ranges.
        self.edge_index: "torch.Tensor | None" = None
        # indexed by config_ranges.
        self.config_feat: "torch.Tensor | None" = None
        self.config_runtime: "torch.Tensor | None" = None  # ^^
        self.config_runtime_normalizers: "torch.Tensor | None" = None  # ^^
        self.tile_id: "torch.Tensor | None" = None

        # edge_ranges[i] = [num_edges[i], num_edges[i+1]]
        self.edge_ranges: "torch.Tensor | None" = None
        # node_ranges[i] = [num_nodes[i], num_nodes[i+1]]
        self.node_ranges: "torch.Tensor | None" = None
        # config_ranges[i] = [num_configs[i], num_configs[i+1]]
        self.config_ranges: "torch.Tensor | None" = None

    def save_to_file(self, cache_file: str):
        """Saves dataset as numpy. Can be restored with `load_from_file`."""
        assert self.node_feat is not None, "finalize() was not invoked"
        assert self.node_opcode is not None
        assert self.edge_index is not None
        assert self.config_feat is not None
        assert self.config_runtime is not None
        assert self.config_runtime_normalizers is not None
        assert self.tile_id is not None
        assert self.edge_ranges is not None
        assert self.node_ranges is not None
        assert self.config_ranges is not None

        np_dict = dict(
            node_feat=self.node_feat.cpu().numpy(),
            node_opcode=self.node_opcode.cpu().numpy(),
            edge_index=self.edge_index.cpu().numpy(),
            config_feat=self.config_feat.cpu().numpy(),
            config_runtime=self.config_runtime.cpu().numpy(),
            config_runtime_normalizers=self.config_runtime_normalizers.cpu().numpy(),
            edge_ranges=self.edge_ranges.cpu().numpy(),
            node_ranges=self.node_ranges.cpu().numpy(),
            config_ranges=self.config_ranges.cpu().numpy(),
        )
        bytes_io = io.BytesIO()
        np.savez_compressed(bytes_io, **np_dict)
        with open(cache_file, "wb") as fout:
            fout.write(bytes_io.getvalue())
        print("wrote " + cache_file)
        tile_ids_file = cache_file + ".tiles.txt"
        with open(tile_ids_file, "w") as fout:
            fout.write("\n".join(map(str, self.tile_id.cpu().numpy().tolist())))
        print("wrote " + tile_ids_file)

    def load_from_file(self, cache_file: str):
        """Loads dataset from numpy file."""
        np_dict = np.load(open(cache_file, "rb"))
        self.node_feat = torch.tensor(np_dict["node_feat"])
        self.node_opcode = torch.tensor(np_dict["node_opcode"])
        self.edge_index = torch.tensor(np_dict["edge_index"])
        self.config_feat = torch.tensor(np_dict["config_feat"])
        self.config_runtime = torch.tensor(np_dict["config_runtime"])
        self.config_runtime_normalizers = torch.tensor(
            np_dict["config_runtime_normalizers"]
        )
        self.edge_ranges = torch.tensor(np_dict["edge_ranges"])
        self.node_ranges = torch.tensor(np_dict["node_ranges"])
        self.config_ranges = torch.tensor(np_dict["config_ranges"])
        with open(cache_file + ".tiles.txt", "r") as f:
            tile_ids = f.readlines()
        self.tile_id = [tile_id.rstrip() for tile_id in tile_ids]
        print("loaded from " + cache_file)

    def add_npz_file(
        self, tile_id: str, npz_file: np.lib.npyio.NpzFile, min_configs: int = 2
    ):
        """Copies data from npz file into this class instance.

        After finishing all calls `add_npz_file()`, user must invoke `finalize()`.

        Args:
        tile_id: the filename (without extension) that npz_file was read from.
        npz_file: Output of np.load on a file from the TpuGraphs Tiles dataset.
        min_configs: The file be incorporated only if the number of module
            configurations is equal or greater than this.
        """
        npz_data = dict(npz_file.items())
        num_configs = npz_data["config_feat"].shape[0]
        if num_configs < min_configs:
            print(f"skipping tile with only {num_configs} configurations")
            return
        # Iterate through the data in the npz file and add it to the data dictionary
        for key, ndarray in npz_data.items():
            self._data_dict[key].append(torch.tensor(ndarray))
        # Add the tile id to the data dictionary
        self._data_dict["tile_id"].append(tile_id)
        # Get the number of nodes, edges, and configurations from the npz file
        num_nodes = npz_data["node_feat"].shape[0]
        num_edges = npz_data["edge_index"].shape[0]
        assert num_nodes == npz_data["node_opcode"].shape[0]
        assert num_configs == npz_data["config_runtime"].shape[0]
        assert num_configs == npz_data["config_runtime_normalizers"].shape[0]
        # Add the number of nodes, edges, and configurations to the data dictionary
        self._num_nodes.append(num_nodes)
        self._num_edges.append(num_edges)
        self._num_configs.append(num_configs)

    def finalize(self):
        print([key for key in self._data_dict.keys()])
        # Set the tile_id to the value stored in the _data_dict
        self.tile_id = self._data_dict["tile_id"]
        # Print the size of the node_feat before stacking
        print("before stacking", len(self._data_dict["node_feat"]))
        print(self._data_dict["node_feat"][0].size())
        # Stack the node_feat, node_opcode, edge_index, config_feat, config_runtime, and config_runtime_normalizers
        self.node_feat = torch.cat(self._data_dict["node_feat"], dim=0)
        print("self.node_feat", self.node_feat.size())
        self.node_opcode = torch.cat(self._data_dict["node_opcode"], dim=0)
        # print size
        print("self.node_opcode", self.node_opcode.size())
        self.edge_index = torch.cat(self._data_dict["edge_index"], dim=0)
        # print size
        print("self.edge_index", self.edge_index.size())
        self.config_feat = torch.cat(self._data_dict["config_feat"], dim=0)
        # print size
        print("self.config_feat", self.config_feat.size())
        self.config_runtime = torch.cat(self._data_dict["config_runtime"], dim=0)
        # print size
        print("self.config_runtime", self.config_runtime.size())
        self.config_runtime_normalizers = torch.cat(
            self._data_dict["config_runtime_normalizers"], dim=0
        )
        # print size
        print("self.config_runtime_normalizers", self.config_runtime_normalizers.size())
        # Calculate the edge_ranges, node_ranges, and config_ranges
        self.edge_ranges = torch.cumsum(
            torch.tensor(self._num_edges, dtype=torch.int64), dim=0
        )
        self.node_ranges = torch.cumsum(
            torch.tensor(self._num_nodes, dtype=torch.int64), dim=0
        )
        self.config_ranges = torch.cumsum(
            torch.tensor(self._num_configs, dtype=torch.int64), dim=0
        )


def get_npz_dataset(
    root_path: str, min_train_configs=-1, cache_dir: "None | str" = None
) -> NpzDataset:
    """Returns {train, test, validation} partitions of tiles dataset collection.

    All partitions will be normalized: statistics are computed from training set
    partition and applied to all partitions.

    Args:
        root_path: Path where dataset lives. It must have subdirectories 'train',
        'test' and 'valid'.
        min_train_configs: If > 0, then tile examples will be filtered to have at
        least this many configurations (features and runtimes).
        cache_dir: If given, the many files for each of {train, test, validation}
        will be stored as one file (makes loading faster, for future runs).
    """
    # Load the dataset from the given root path.
    npz_dataset = NpzDataset(
        # Get the train split from the given root path, with a minimum number of configurations, and cache directory.
        train=get_npz_split(
            os.path.join(root_path, "train"),
            min_configs=min_train_configs,
            cache_dir=cache_dir,
        ),
        # Get the validation split from the given root path, and cache directory.
        validation=get_npz_split(os.path.join(root_path, "valid"), cache_dir=cache_dir),
        # Get the test split from the given root path, and cache directory.
        test=get_npz_split(os.path.join(root_path, "test"), cache_dir=cache_dir),
    )
    # Normalize the dataset.
    npz_dataset.normalize()
    # Return the dataset.
    return npz_dataset


def get_npz_split(
    split_path: str, min_configs: int = 2, cache_dir: Optional[str] = None
) -> NpzDatasetPartition:
    """
    Loads the npz files from a dataset partition path into a NpzDatasetPartition.

    Args:
        split_path (str): The path to the dataset partition.
        min_configs (int, optional): The minimum number of configurations required for a tile to be included in the dataset. Defaults to 2.
        cache_dir (str, optional): The directory to cache the loaded dataset to. Defaults to None.

    Returns:
        NpzDatasetPartition: A NpzDatasetPartition containing the loaded data.
    """
    # Get the glob pattern for the npz files in the dataset partition
    glob_pattern = os.path.join(split_path, "*.npz")
    # Get a list of all the npz files in the dataset partition
    files = [f for f in glob.iglob(glob_pattern)]
    # If there are no files, raise an error
    if not files:
        raise ValueError("No files matched: " + glob_pattern)
    # If the _TOY_DATA_VALUE is set, only select the first 100 files
    if _TOY_DATA_VALUE:
        files = files[:100]
        print("here")

    # Initialize the cache filename
    cache_filename = None
    # If a cache directory is provided, create the directory if it does not exist
    if cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Create a hash of the split path, min configs, and _TOY_DATA_VALUE
        filename_hash = hashlib.md5(
            f"{split_path}:{min_configs}:{_TOY_DATA_VALUE}".encode()
        ).hexdigest()
        # Create the cache filename
        cache_filename = os.path.join(cache_dir, f"{filename_hash}-cache.npz")
        print("dataset cache file: ", cache_filename)

    # Initialize the NpzDatasetPartition
    npz_dataset = NpzDatasetPartition()
    # If a cache filename is provided, load the data from the cache
    if cache_filename and os.path.exists(cache_filename):
        npz_dataset.load_from_file(cache_filename)
    else:
        # Iterate through each npz file in the dataset partition
        for filename in tqdm.tqdm(files):
            # Load the npz file
            np_data = np.load(open(filename, "rb"))
            # Get the tile id from the filename
            tile_id = os.path.splitext(os.path.basename(filename))[0]
            # Add the npz file to the NpzDatasetPartition
            npz_dataset.add_npz_file(tile_id, np_data, min_configs=min_configs)
        # Finalize the NpzDatasetPartition
        npz_dataset.finalize()
        # If a cache filename is provided, save the data to the cache
        if cache_filename:
            npz_dataset.save_to_file(cache_filename)

    # Return the NpzDatasetPartition
    return npz_dataset


if __name__ == "__main__":
    # for the purpose of testing
    root_data_path = (
        "/Users/kaiqu/Desktop/kaggle-runtime-optimization/dataset/npz_all/npz/tile/xla"
    )
    npz_dataset = get_npz_dataset(root_data_path)
    train_partition = npz_dataset.train
    val_partition = npz_dataset.validation
    test_partition = npz_dataset.test
    print(train_partition.node_feat.size())
    print(len(train_partition.tile_id))
    print(val_partition.node_feat.size())
    print(len(val_partition.tile_id))
    print(test_partition.node_feat.size())
    print(len(test_partition.tile_id))
