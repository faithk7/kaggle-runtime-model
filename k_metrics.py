import torch
import torch.nn as nn
import torch.nn.functional as F


class ListMLELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ListMLELoss, self).__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        """
        Forward pass for the ListMLE loss.

        Args:
            y_pred: Predicted scores for the items, shape (batch_size, list_size).
            y_true: True scores for the items, used to determine the true ordering, shape (batch_size, list_size).

        Returns:
            torch.Tensor: The ListMLE loss.
        """
        # Sort items by their true scores in descending order
        _, indices = torch.sort(y_true, descending=True, dim=1)

        # Gather the predicted scores according to the true ordering
        y_pred_sorted = torch.gather(y_pred, 1, indices)

        # Apply softmax to the predicted scores (scaled by temperature if necessary)
        y_pred_softmax = F.softmax(y_pred_sorted / self.temperature, dim=1)

        # Compute the negative log likelihood
        loss = -torch.mean(torch.sum(torch.log(y_pred_softmax), dim=1))
        return loss


class PairwiseHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        """
        Forward pass for the Pairwise Hinge Loss.

        Args:
            y_pred: Predicted scores, shape (batch_size, list_size).
            y_true: True labels with values indicating the relevance or rank, shape (batch_size, list_size).

        Returns:
            torch.Tensor: The Pairwise Hinge loss.
        """
        # Find the number of pairs per batch
        n_pairs = y_pred.size(1) * (y_pred.size(1) - 1) // 2

        # Create a mask for pairs
        mask = y_true[:, :, None] > y_true[:, None, :]
        mask = mask.float()

        # Compute differences matrix for predictions and ground truth
        pred_diffs = y_pred[:, :, None] - y_pred[:, None, :]
        true_diffs = y_true[:, :, None] - y_true[:, None, :]

        # Apply mask to the differences matrix
        pred_diffs = pred_diffs * mask
        true_diffs = true_diffs * mask

        # Calculate the loss
        losses = torch.nn.functional.relu(
            self.margin - pred_diffs * torch.sign(true_diffs)
        )

        # We do not consider the pair (i, i) and pairs have been counted twice
        loss = torch.sum(losses) / (2 * n_pairs * y_pred.size(0))

        return loss


# Dictionary of loss functions
LOSS_DICT = {
    "ListMLELoss": ListMLELoss(temperature=10),
    "PairwiseHingeLoss": PairwiseHingeLoss(margin=10),
    "MSE": nn.MSELoss(),
}


class CombinedLoss(nn.Module):
    def __init__(self, weighted_losses=None):
        super().__init__()
        if weighted_losses is None:
            weighted_losses = [(1.0, ListMLELoss(temperature=10))]
        self.weighted_losses = nn.ModuleList(
            [(weight, loss) for weight, loss in weighted_losses]
        )
        total_weight = sum([w for w, _ in weighted_losses])
        self.weighted_losses = nn.ModuleList(
            [(w / total_weight, loss) for w, loss in weighted_losses]
        )

    def forward(self, y_true, y_pred):
        losses = [
            weight * loss(y_true, y_pred) for weight, loss in self.weighted_losses
        ]
        return sum(losses)


def parse_loss_str(loss_str: str) -> list[tuple[float, nn.Module]]:
    weighted_losses = []
    for loss_item in loss_str.split(","):
        loss_name, loss_weight = loss_item.split(":")
        assert loss_name in LOSS_DICT, f"{loss_name} not in LOSS_DICT"
        loss_weight = float(loss_weight)
        loss = LOSS_DICT[loss_name]
        weighted_losses.append((loss_weight, loss))
    return weighted_losses


# * This is not used in training, but is used in the evaluation for the final models
def top_error_performance(test_loader, model_fn, top_k=(1, 5, 10)) -> dict:
    """
    Computes test errors as: (ModelChosen - Best) / Best.

    Args:
        test_loader: DataLoader for the test set. It should yield batches of data
        that will be passed to model_fn.
        model_fn: Function that accept a batch of data and returns a tensor with
        shape `[batch_size, N]`, where `N` is the number of configurations.
        top_k: A sequence of integers for the top-k selections.

    Returns:
        A dictionary with each of `top_k` as a key and the corresponding error as the value.
    """
    num_examples = 0
    result = {k: 0.0 for k in top_k}
    for batch in test_loader:
        # The batch is a tuple of data and targets, you might need to adjust this
        # depending on how your data loader is set up.
        data, targets = batch
        num_configs = data.shape[1]  # Assuming data shape is [batch_size, num_configs]
        preds = model_fn(data).squeeze(0)  # Remove batch dimension

        sorted_indices = preds.argsort()

        runtimes = (
            targets  # Assuming the second element of the batch tuple is the runtimes
        )
        time_best = torch.min(runtimes)
        num_examples += data.shape[0]  # Increment by the batch size

        for k in top_k:
            time_model_candidates = torch.gather(runtimes, 0, sorted_indices[:k])
            best_of_candidates = torch.min(time_model_candidates)
            result[k] += float((best_of_candidates - time_best) / time_best)

    for k in top_k:
        result[k] /= num_examples

    return result
