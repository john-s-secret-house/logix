import argparse

import torch
from tqdm import tqdm
from utils import construct_rn9, get_cifar10_dataloader, get_cifar2_dataloader

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator
from logix.config import Config, LoggingConfig

if __name__ == "__main__":
    lora = "pca"
    hessian = "kfac"
    save = "grad"
    save_path = "tmp_test"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging_config: LoggingConfig = LoggingConfig(flush_threshold=1000000000, num_workers=1, cpu_offload=False)
    logix = LogIX(
        project=save_path,
        config="./config.yaml",
        logging_config=logging_config,)
    logix_scheduler = LogIXScheduler(
        logix, lora=lora, hessian=hessian, save=save, epoch=1
    )
    
    print(f"len(logix_scheduler): {len(logix_scheduler)}")
    for epoch in logix_scheduler:
        print(f"Epoch {epoch}")