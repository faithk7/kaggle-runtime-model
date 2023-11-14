import functools
import json
import os
from collections.abc import Sequence

import torch
from absl import app
from torch.utils.data import DataLoader

from k_metrics import *
from k_model import *
from k_tile_dataset_torch import *
from k_train_args import *

_DATA_ROOT = (
    "/Users/kaiqu/Desktop/kaggle-runtime-optimization/dataset/npz_all/npz/tile/xla"
)
_CACHE_DIR = "/Users/kaiqu/Desktop/kaggle-runtime-optimization/cache"


def should_early_stop(early_stop_val, early_stop_patience, best_val):
    return False


def train(args: TrainArgs):
    out_dir = os.path.expanduser(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_info = {
        # ...same as above...
    }

    data_root_dir = os.path.expanduser(_DATA_ROOT)
    num_configs = args.configs
    # Define your method to get a PyTorch Dataset here.
    dataset_partitions = get_npz_dataset(
        data_root_dir,
        min_configs=num_configs,
        cache_dir=os.path.expanduser(_CACHE_DIR),
    )
    batch_size = args.batch_size
    # Convert to DataLoader
    train_loader = DataLoader(
        dataset_partitions.train, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset_partitions.validation, batch_size=batch_size)

    # Define your PyTorch model here.
    # model_class = getattr(models, args.model)
    # model_kwargs = json.loads(args.model_kwargs_json)
    num_ops = dataset_partitions.num_ops
    # model = model_class(num_configs, num_ops, **model_kwargs)
    model = MLP(num_configs, num_ops)

    loss_fn = CombinedLoss(parse_loss_str(args.losses))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_opa = -1
    best_loss = float("inf")
    best_params = None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)  # Forward pass
            loss = loss_fn(outputs, batch.targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

        # Validation step
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(batch)  # Forward pass
                loss = loss_fn(outputs, batch.targets)  # Compute loss
                # ...perform validation...
                # TODO: Update best_val_opa and best_params as needed
                # if best_val_opa < opa:
                #     best_val_opa = opa
                #     best_params = model.state_dict()
                if best_loss > loss:
                    best_loss = loss
                    best_params = model.state_dict()

        if should_early_stop(args.early_stop, best_val_opa, epoch):
            break

    # Restore best parameters
    model.load_state_dict(best_params)

    # Save model and run_info
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))


def main(unused_argv: Sequence[str]) -> None:
    train(get_args())


if __name__ == "__main__":
    app.run(main)
