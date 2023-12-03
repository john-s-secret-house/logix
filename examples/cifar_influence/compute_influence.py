import time
import argparse
import torch

from analog import AnaLog
from analog.utils import DataIDGenerator
from analog.analysis import InfluenceFunction

from train import (
    get_cifar10_dataloader,
    construct_rn9,
)

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--data", type=str, default="cifar10", help="cifar10/100")
parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--damping", type=float, default=1e-5)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = construct_rn9().to(DEVICE)

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    torch.load(f"checkpoints/{args.data}_0_epoch_23.pt", map_location="cpu")
)
model.eval()

dataloader_fn = get_cifar10_dataloader
train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=True, augment=False
)
query_loader = dataloader_fn(
    batch_size=1, split="valid", shuffle=False, indices=eval_idxs, augment=False
)

analog = AnaLog(project="test", config="./config.yaml")

# Gradient & Hessian logging
analog.watch(model)

if not args.resume:
    analog_kwargs = {"log": ["grad"], "hessian": True, "save": True}
    id_gen = DataIDGenerator()
    for inputs, targets in train_loader:
        data_id = id_gen(inputs)
        with analog(data_id=data_id, **analog_kwargs):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            model.zero_grad()
            outs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
            loss.backward()
    analog.finalize()
    analog.save_hessian_state()
else:
    analog.load_hessian_state()

# Influence Analysis
log_loader = analog.build_log_dataloader()

analog.add_analysis({"influence": InfluenceFunction})
query_iter = iter(query_loader)
with analog(log=["grad"], test=True) as al:
    test_input, test_target = next(query_iter)
    test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
    model.zero_grad()
    test_out = model(test_input)
    test_loss = torch.nn.functional.cross_entropy(
        test_out, test_target, reduction="sum"
    )
    test_loss.backward()
    test_log = al.get_log()
start = time.time()
if_scores = analog.influence.compute_influence_all(
    test_log, log_loader, damping=args.damping
)

# Save
if_scores = if_scores.numpy().tolist()
torch.save(if_scores, "./if_baseline.pt")
print("Computation time:", time.time() - start)
