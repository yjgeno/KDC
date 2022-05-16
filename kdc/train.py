from collections import defaultdict
import torch
from .data import Dataset
from .model import KDC, save_model, load_model
import os
import torch.utils.tensorboard as tb


def train(args, return_=False):
    """
    Train a KDC model.
    """
    dataset = Dataset(
        data=args.data,
        perturbation_key="condition",
        dose_key="dose_val",
        split_key="split",
        control_label="control"
    )

    model = KDC(
        dataset.num_genes,
        dataset.num_drugs,
        loss_ae="gauss"
    )

    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(os.path.join(args.log_dir, "train"))
    global_step = 0
    for epoch in range(args.n_epochs):
        epoch_training_stats = defaultdict(float)  # loss per epoch
        for genes, drugs in dataset:
            # update params
            batch_training_stats = model.update(genes, drugs)  # a dict of loss
            for key, val in batch_training_stats.items():
                epoch_training_stats[key] += val  # logger
                if args.log_dir is not None:
                    train_logger.add_scalar(key, val, global_step)
            global_step += 1
        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(dataset)
        if args.verbose:
            print(f"epoch: {epoch}, training_stats: {epoch_training_stats}")

    save_model(model)

    if return_:
        return model, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("-N", "--n_epochs", type=int, default=2000)
    parser.add_argument("-B", "--batch_size", type=int, default=256)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    torch.manual_seed(42)
    train(args)
    # python -m kdc.train --data pbmc3k_sample.h5ad --log_dir logdir -N 2000 -v
