import os
import argparse

import torch
from tqdm import tqdm
from utils import construct_rn9, get_cifar10_dataloader, get_cifar2_dataloader

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator
from logix.config import Config, LoggingConfig

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--ckpt_path", type=str, default="../../../ckpts/CIFAR2_32_bs2048_ckpts", help="Checkpoint path")
parser.add_argument("--md_num", type=int, default=10, help="Checkpoint model number")
parser.add_argument("--data", type=str, default="CIFAR2_32", help="CIFAR10_32/CIFAR100_32")
# parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--epoch", type=int, default=-1, help="Epoch to train the model")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=list(range(30)))
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--lora", type=str, default="none")
# parser.add_argument("--hessian", type=str, default="raw")
parser.add_argument("--hessian", type=str, default="kfac")
parser.add_argument("--save", type=str, default="grad")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

DEVICE = torch.device(args.device)
dataset = args.data
# save_folder: str = f"{args.data}_md{args.md_num}_ep{args.epoch}_LoRA{args.lora}_H{args.hessian}_S{args.save}"
save_folder: str = f"{args.data}_md{args.md_num}_LoRA{args.lora}_H{args.hessian}_S{args.save}"
save_path: str = f"logix/{save_folder}/if_logix.pt"
if os.path.isfile(save_path):
    print(f"Already computed: {save_path}")
    exit(0)

model = construct_rn9().to(DEVICE)

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    # torch.load(f"checkpoints/{args.data}_0_epoch_23.pt", map_location="cpu")
    torch.load(f"{args.ckpt_path}/md_{args.md_num}/final.pt", map_location="cpu")
)
model.eval()

if dataset == 'CIFAR10_32':
    dataloader_fn = get_cifar10_dataloader
elif dataset == 'CIFAR2_32':
    dataloader_fn = get_cifar2_dataloader
else:
    raise NotImplementedError(f"dataset, {dataset} is not supported")

train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=False, augment=False
)
test_loader = dataloader_fn(
    # batch_size=16, split="valid", shuffle=False, indices=args.eval_idxs, augment=False
    batch_size=512, split="valid", shuffle=False, subsample=False, augment=False
)
logging_config: LoggingConfig = LoggingConfig(flush_threshold=1000000000, num_workers=1, cpu_offload=False)
logix = LogIX(
    project=save_folder,
    config="./config.yaml",
    logging_config=logging_config,)
logix_scheduler = LogIXScheduler(
    logix, lora=args.lora, hessian=args.hessian, save=args.save
)

# Gradient & Hessian logging
logix.watch(model)

id_gen = DataIDGenerator()
for epoch in logix_scheduler:
    for itm in tqdm(train_loader, desc="Extracting log"):
        inputs, targets = itm['input'], itm['label']
        with logix(data_id=id_gen(inputs)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    logix.finalize()

# Influence Analysis
log_loader = logix.build_log_dataloader()

logix.eval()
logix.setup({"grad": ["log"]})
results = []
if_scores_ls = []
for test_itm in test_loader:
    test_input, test_target = test_itm['input'], test_itm['label']
    with logix(data_id=id_gen(test_input)):
        test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
        model.zero_grad()
        test_out = model(test_input)
        test_loss = torch.nn.functional.cross_entropy(
            test_out, test_target, reduction="sum"
        )
        test_loss.backward()
        test_log = logix.get_log()

    # Influence computation
    result = logix.influence.compute_influence_all(
        test_log, log_loader, damping=args.damping
    )
    results.append(result)
    if_scores_ls.append(result["influence"])
    # break

# Save
# if_scores = result["influence"].numpy().tolist()
if_scores = torch.cat(if_scores_ls, dim=0).transpose(0, 1)
torch.save({'score': if_scores}, save_path)
print(f"influence: {if_scores.shape}")
