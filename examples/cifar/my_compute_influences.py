import argparse
import importlib.metadata
from packaging import version

import torch
from tqdm import tqdm
from utils import construct_rn9, get_cifar10_dataloader, get_cifar2_dataloader, get_cifar2_dataset, get_cifar10_dataset

from transformers import Trainer, TrainingArguments, default_data_collator
# from transfomrers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from logix.huggingface import LogIXArguments, patch_trainer
from logix import LogIX, LogIXScheduler
from logix.utils import DataIDGenerator

parser = argparse.ArgumentParser("CIFAR Influence Analysis")
parser.add_argument("--ckpt_path", type=str, default="../../../ckpts/CIFAR2_32_bs2048_ckpts", help="Checkpoint path")
parser.add_argument("--md_num", type=int, default=10, help="Checkpoint model number")
parser.add_argument("--data", type=str, default="cifar2", help="cifar10/100")
# parser.add_argument("--eval-idxs", type=int, nargs="+", default=[0])
parser.add_argument("--eval-idxs", type=int, nargs="+", default=list(range(30)))
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--lora", type=str, default="none")
# parser.add_argument("--hessian", type=str, default="raw")
parser.add_argument("--hessian", type=str, default="kfac")
parser.add_argument("--save", type=str, default="grad")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = 'cifar2'

model = construct_rn9().to(DEVICE)

# Get a single checkpoint (first model_id and last epoch).
model.load_state_dict(
    # torch.load(f"checkpoints/{args.data}_0_epoch_23.pt", map_location="cpu")
    torch.load(f"{args.ckpt_path}/md_{args.md_num}/final.pt", map_location="cpu")
)
model.eval()

if dataset == 'cifar10':
    dataloader_fn = get_cifar10_dataloader
    dataset_fn = get_cifar10_dataset
elif dataset == 'cifar2':
    dataloader_fn = get_cifar2_dataloader
    dataset_fn = get_cifar2_dataset
else:
    raise NotImplementedError(f"dataset, {dataset} is not supported")

train_dataset = dataset_fn(
    split="train", subsample=False, augment=False
)
train_loader = dataloader_fn(
    batch_size=512, split="train", shuffle=False, subsample=False, augment=False
)
test_dataset = dataset_fn(
    split="valid", subsample=False, augment=False
)
test_loader = dataloader_fn(
    # batch_size=16, split="valid", shuffle=False, indices=args.eval_idxs, augment=False
    batch_size=512, split="valid", shuffle=False, subsample=False, augment=False
)

# logix = LogIX(project=f"{args.data}_md{args.md_num}", config="./config.yaml")
# logix_scheduler = LogIXScheduler(
#     logix, lora=args.lora, hessian=args.hessian, save=args.save
# )

# # Gradient & Hessian logging
# logix.watch(model)

# id_gen = DataIDGenerator()
# for epoch in logix_scheduler:
#     for inputs, targets in tqdm(train_loader, desc="Extracting log"):
#         with logix(data_id=id_gen(inputs)):
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             model.zero_grad()
#             outs = model(inputs)
#             loss = torch.nn.functional.cross_entropy(outs, targets, reduction="sum")
#             loss.backward()
#     logix.finalize()

# # Influence Analysis
# log_loader = logix.build_log_dataloader()

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        # outputs = model(**inputs)
        outputs = model(inputs['input'])
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss.clone(), outputs.clone()) if return_outputs else loss.clone()

def logix_log(args, model, train_dataset: torch.utils.data.Dataset):
    logix_args = LogIXArguments(
        project=f"{args.data}_md{args.md_num}_hf",
        config="./config.yaml",
        lora=args.lora,
        hessian=args.hessian,
        save=args.save,
        input_key="input",
        label_key="labels",
        data_id='hash',
        log_batch_size=512,
    )
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=512,
        report_to="none",
        dataloader_drop_last=True,
    )

    LogIXTrainer = patch_trainer(Trainer)
    trainer = LogIXTrainer(
        model=model,
        # processing_class=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        compute_loss_func=lambda outputs, labels, num_items_in_batch: torch.nn.functional.cross_entropy(outputs, labels),
        args=training_args,
        logix_args=logix_args,
    )
    trainer.extract_log()
    
def logix_influence(args, model, test_dataset: torch.utils.data.Dataset):
    logix_args = LogIXArguments(
        project=f"{args.data}_md{args.md_num}_hf",
        config="./config.yaml",
        lora=args.lora,
        hessian=args.hessian,
        save=args.save,
        input_key="input",
        label_key="labels",
        data_id='hash',
        initialize_from_log=True,
        log_batch_size=512,
    )
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1,
        per_device_train_batch_size=512,
        report_to="none",
        gradient_accumulation_steps=1,
        dataloader_drop_last=True,
        # label_names=['labels'],
    )

    LogIXTrainer = patch_trainer(Trainer)
    trainer = LogIXTrainer(
        model=model,
        # processing_class=tokenizer,
        train_dataset=test_dataset,
        data_collator=default_data_collator,
        compute_loss_func=lambda outputs, labels, num_items_in_batch: torch.nn.functional.cross_entropy(outputs, labels),
        args=training_args,
        logix_args=logix_args,
    )
    if_scores = trainer.influence()
    torch.save(if_scores, "gpt_influence.pt")
    
    
logix_log(args=args, model=model, train_dataset=train_dataset)
logix_influence(args=args, model=model, test_dataset=test_dataset)

# logix.eval()
# logix.setup({"grad": ["log"]})
# results = []
# if_scores_ls = []
# for test_input, test_target in test_loader:
#     with logix(data_id=id_gen(test_input)):
#         test_input, test_target = test_input.to(DEVICE), test_target.to(DEVICE)
#         model.zero_grad()
#         test_out = model(test_input)
#         test_loss = torch.nn.functional.cross_entropy(
#             test_out, test_target, reduction="sum"
#         )
#         test_loss.backward()
#         test_log = logix.get_log()

#     # Influence computation
#     result = logix.influence.compute_influence_all(
#         test_log, log_loader, damping=args.damping
#     )
#     results.append(result)
#     if_scores_ls.append(result["influence"])
#     # break

# # Save
# # if_scores = result["influence"].numpy().tolist()
# if_scores = torch.cat(if_scores_ls, dim=0)
# torch.save(if_scores, f"logix/{args.data}_md{args.md_num}/if_logix.pt")
# print(f"influence: {if_scores.shape}")
