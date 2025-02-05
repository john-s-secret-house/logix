import argparse

import torch
from tqdm import tqdm
from utils import construct_rn9, get_cifar10_dataloader, get_cifar2_dataloader

from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--ckpt_path", type=str, default="../../../ckpts/CIFAR2_32_bs2048_ckpts", help="Checkpoint path")
parser.add_argument("--md_num", type=int, default=10, help="Checkpoint model number")
parser.add_argument("--data", type=str, default="CIFAR2_32", help="CIFAR10_32/CIFAR100_32")
# parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--eval-idxs", type=int, nargs="+", default=list(range(30)))
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--lora", type=str, default="none")
# parser.add_argument("--hessian", type=str, default="raw")
parser.add_argument("--hessian", type=str, default="kfac")
parser.add_argument("--save", type=str, default="grad")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = args.data

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

logix = LogIX(project=f"{args.data}_md{args.md_num}", config="./config.yaml")
logix_scheduler = LogIXScheduler(
    logix, lora=args.lora, hessian=args.hessian, save=args.save
)

# Gradient & Hessian logging
logix.watch(model)

id_gen = DataIDGenerator()
for epoch in logix_scheduler:
    for inputs, targets in tqdm(train_loader, desc="Extracting log"):
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
for test_input, test_target in test_loader:
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
if_scores = torch.cat(if_scores_ls, dim=0)
torch.save(if_scores, f"logix/{args.data}_md{args.md_num}/if_logix.pt")
print(f"influence: {if_scores.shape}")
