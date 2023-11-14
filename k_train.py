import functools
import json
import os

import torch
from torch.utils.data import DataLoader

from k_metrics import *
from k_model import *
from k_tile_dataset_torch import *

_DATA_ROOT = (
    "/Users/kaiqu/Desktop/kaggle-runtime-optimization/dataset/npz_all/npz/tile/xla"
)
_CACHE_DIR = "/Users/kaiqu/Desktop/kaggle-runtime-optimization/cache"


def should_early_stop(early_stop_val, early_stop_patience, best_val):
    return False


def train(args: train_args.TrainArgs):
    out_dir = os.path.expanduser(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_info = {
        # ...same as above...
    }

    data_root_dir = os.path.expanduser(_DATA_ROOT.value)
    num_configs = args.configs
    # Define your method to get a PyTorch Dataset here.
    dataset_partitions = get_npz_split(
        data_root_dir,
        min_configs=num_configs,
        cache_dir=os.path.expanduser(_CACHE_DIR.value),
    )
    batch_size = args.batch_size
    # Convert to DataLoader
    train_loader = DataLoader(
        dataset_partitions.train, batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(dataset_partitions.validation, batch_size=batch_size)

    # Define your PyTorch model here.
    model_class = getattr(models, args.model)
    model_kwargs = json.loads(args.model_kwargs_json)
    num_ops = dataset_partitions.num_ops
    model = model_class(num_configs, num_ops, **model_kwargs)

    loss_fn = CombinedLoss(parse_loss_str(args.losses))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_opa = -1
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
                # Update best_val_opa and best_params as needed
                if best_val_opa < opa:
                    best_val_opa = opa
                    best_params = model.state_dict()

        if should_early_stop(args.early_stop, best_val_opa, epoch):
            break

    # Restore best parameters
    model.load_state_dict(best_params)

    # Save model and run_info
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
