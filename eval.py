import os
import sys

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.RNN import *
from tsai.models.TCN import *

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    # model = BasicConvClassifier(
    #     test_set.num_classes, test_set.seq_len, test_set.num_channels
    # ).to(args.device)
    
    # model = MiniRocket(
    #     c_in=test_set.num_channels,
    #     seq_len=test_set.seq_len,
    #     num_features=256,
    #     max_dilations_per_kernel=32,
    #     c_out=test_set.num_classes,
    # ).to(args.device)
    
    model = TCN(
        c_in=test_set.num_channels,
        c_out=test_set.num_classes,
        ks=7,
        layers=[25] * 3,
        conv_dropout=0.2,
        fc_dropout=0.2,
    ).to(args.device)
    
    # model = LSTM(
    #     c_in=test_set.num_channels,
    #     c_out=test_set.num_classes,
    #     hidden_size=512,
    #     n_layers=2,
    #     rnn_dropout=0.2,
    #     fc_dropout=0.2,
    #     bidirectional=True,
    # ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()