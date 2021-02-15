import os
import argparse
import multiprocessing as mp

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch_aesthetics
from torch_aesthetics.losses import RegRankLoss
from torch_aesthetics.models import RegressionNetwork
from torch_aesthetics.aadb import AADB, load_transforms


# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(cfg: DictConfig):

    # Load model
    model = RegressionNetwork(
        backbone=cfg.models.backbone,
        num_attributes=cfg.data.num_attributes,
        pretrained=cfg.models.pretrained
    )
    model = model.to(cfg.device).to(torch.float32)

    # Setup optimizer and loss func
    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum
    )
    loss_fn = RegRankLoss(margin=cfg.train.margin)

    # Load datasets
    train_dataset = AADB(
        image_dir=cfg.data.image_dir,
        labels_dir=cfg.data.labels_dir,
        split="train",
        transforms=load_transforms(input_shape=cfg.data.input_shape)
    )
    val_dataset = AADB(
        image_dir=cfg.data.image_dir,
        labels_dir=cfg.data.labels_dir,
        split="val",
        transforms=load_transforms(input_shape=cfg.data.input_shape)
    )

    # Setup dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size * 2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):

        model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, (x, y) in pbar:

            opt.zero_grad()

            x = x.to(cfg.device).to(torch.float32)
            y = y.to(cfg.device).to(torch.float32)[:, 1:]
            x1, x2 = torch.split(x, cfg.data.batch_size, dim=0)
            y1, y2 = torch.split(y, cfg.data.batch_size, dim=0)

            y_pred1 = model(x1)
            y_pred2 = model(x2)
            loss, loss_reg, loss_rank = loss_fn(
                y_pred=(y_pred1, y_pred2),
                y_true=(y1, y2)
            )

            loss.backward()
            opt.step()

            pbar.set_description("Epoch {}, Reg Loss: {:.4f}, Rank Loss: {:.4f} ".format(
                epoch, float(loss_reg), float(loss_rank)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(
                    tag="loss", scalar_value=float(loss), global_step=n_iter
                )
                writer.add_scalar(
                    tag="loss_reg", scalar_value=float(loss_reg), global_step=n_iter
                )
                writer.add_scalar(
                    tag="loss_rank", scalar_value=float(loss_rank), global_step=n_iter
                )

            n_iter += 1

        # Evaluate
        model.eval()
        test_loss = 0.0
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for batch, (x, y) in pbar:
            x = x.to(cfg.device).to(torch.float32)
            y = y.to(cfg.device).to(torch.float32)[:, 1:]

            with torch.no_grad():
                y_pred = model(x)
                test_loss += F.mse_loss(y_pred, y)

        test_loss /= len(val_dataloader)
        writer.add_scalar(
            tag="test_loss_reg", scalar_value=test_loss, global_step=n_iter
        )
        writer.add_scalar(
            tag="epoch", scalar_value=epoch, global_step=n_iter
        )


        # save checkpoint
        if cfg.train.save_dir is None:
            cfg.train.save_dir = writer.log_dir

        filename = "{}_epoch_{}_loss_{:.4f}_.pt".format(
            cfg.data.dataset, epoch, test_loss
        )
        torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, filename))        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg)
