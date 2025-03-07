from typing import Literal, Union, List, Set, Dict, Tuple, Any, Optional, Sequence, TypeVar, cast, Callable
from dataclasses import dataclass, asdict, astuple
import itertools
import os
import math
import json
import warnings

from tqdm import tqdm
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, Subset, ConcatDataset

_T = TypeVar("_T")

class DatasetsBase:
    """
    Base class of dataset
    """
    FORMAT_NONE: str = 'FORMAT_NONE'
    FORMAT_NO_IDX: str = 'FORMAT_NO_IDX'
    FORMAT_HF: str = 'FORMAT_HF'
    @staticmethod
    def format_output(*args, output_format: str):
        if output_format == DatasetsBase.FORMAT_NONE:
            return args
        if output_format == DatasetsBase.FORMAT_NO_IDX:
            return args[1:]
        elif output_format == DatasetsBase.FORMAT_HF:
            # print(f"Output: {{'X': args[1], 'y': args[2]}}")
            return {'input': args[1], 'labels': args[2]}
        else:
            raise NotImplementedError(f"format, {output_format} isn't supported")
    @staticmethod
    def get_dataloader(dataset: Dataset,
                       batch_size: int,
                       shuffle: bool=True,
                       num_workers: int=min(8, os.cpu_count()),
                       seed: int=0):
        """
        Apply torch.utils.data.DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            generator=torch.manual_seed(seed))
    @staticmethod
    def stack_by_col(data) -> List[torch.Tensor]:
        """
        Gather data by columns
        """
        num_col: int = len(data[0])
        res: List = [[] for i in range(num_col)]
        # Iterate through the columns and rows, stack data points by columns
        for itm in data:
            for i in range(num_col):
                if torch.is_tensor(itm[i]):
                    res[i].append(itm[i])
                else:
                    res[i].append(torch.tensor(itm[i]))
        for i in range(num_col):
            res[i] = torch.stack(res[i])
        return res
    @staticmethod
    def sample_subset(dataset: Union[torch.Tensor, List],
                      sample_size: Union[float, int]=None,
                      is_gather_by_col: bool=False,
                      generator: torch.Generator=None) -> Union[List[List], List[torch.Tensor]]:
        """
        Sample a subset from the given dataset.

        Args:
            dataset (Union[torch.Tensor, List]): The dataset to sample from.
            sample_size (Union[float, int], optional): The size of the subset to sample. 
                If a float, it represents the fraction of the dataset to sample.
                If an int, it represents the exact number of samples to draw.
                Default is None, which returns the entire dataset.
            is_gather_by_col (bool, optional): If True, the data is gathered by columns.
                Default is False.
            generator (torch.Generator, optional): A random number generator for sampling.
                Default is None.

        Returns:
            Union[List[List], List[torch.Tensor]]: The sampled subset of the dataset.
                The format depends on the `is_gather_by_col` flag.

        Raises:
            TypeError: If `sample_size` is not an int or a float.
        """
        def get_sampled_subset(size: int) -> torch.Tensor:
            """
            Sample a subset of the given size from the given dataset.

            Args:
                size (int): The size of the subset to sample.

            Returns:
                torch.Tensor: The sampled subset of the dataset.
            """
            sample_idx: torch.Tensor = torch.randint(
                low=0, high=len(dataset), size=(size,), generator=generator)
            sampled_subset = Subset(dataset, sample_idx)
            return sampled_subset
        # Fetch subset for given sample_size
        if sample_size is None or sample_size == -1 or (isinstance(sample_size, float) and sample_size == 1.0):
            sampled_subset = dataset
        elif isinstance(sample_size, int):
            sampled_subset = get_sampled_subset(size=sample_size)
        elif isinstance(sample_size, float):
            sampled_subset = get_sampled_subset(size=int(len(dataset) * sample_size))
        else:
            raise TypeError(f"Arguement sample_size shouldn't be {type(sample_size)}, only int or float are acceptable.")
        # Gather by columns
        if is_gather_by_col:
            return DatasetsBase.stack_by_col(data=sampled_subset)
        else:
            return sampled_subset
    @staticmethod
    def random_split(
        dataset: Dataset[_T],
        lengths: Sequence[Union[int, float]],
        generator: Optional[torch.Generator] = torch.default_generator,
    ) -> Union[List[Subset[_T]], Tuple[List[Subset[_T]], List[List[int]]]]:
        r"""
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results, e.g.:

        Example:
            >>> # xdoctest: +SKIP
            >>> generator1 = torch.Generator().manual_seed(42)
            >>> generator2 = torch.Generator().manual_seed(42)
            >>> random_split(range(10), [3, 7], generator=generator1)
            >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

        Args:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths or fractions of splits to be produced
            generator (Generator): Generator used for the random permutation.
                If it's set to None, then it would become deterministic ordered split.
        """
        if isinstance(lengths[0], float):
            for i, length in enumerate(lengths):
                if not isinstance(length, float):
                    raise TypeError(f"The type of the elements of the arguement lengths should be consistent, \
                                    the first one is float but {i}-th element is {type(length)}")
        elif isinstance(lengths[0], int):
            for i, length in enumerate(lengths):
                if not isinstance(length, int):
                    raise TypeError(f"The type of the elements of the arguement lengths should be consistent, \
                                    the first one is int but {i}-th element is {type(length)}")
        else:
            TypeError(f"The element of the arguement lengths should be either int or float, not {type(lengths[0])}")
            
        if isinstance(lengths[0], float):
            subset_lengths: List[int] = []
            for i, frac in enumerate(lengths):
                n_items_in_split = int(
                    math.floor(len(dataset) * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            # print(f"subset_lengths: {subset_lengths}")
            if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
                remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
                # add 1 to all the lengths in round-robin fashion until the remainder is 0
                for i in range(remainder):
                    idx_to_add_at = i % len(subset_lengths)
                    subset_lengths[idx_to_add_at] += 1
                lengths = subset_lengths
                for i, length in enumerate(lengths):
                    if length == 0:
                        warnings.warn(
                            f"Length of split at index {i} is 0. "
                            f"This might result in an empty dataset."
                        )
                # print(f"lengths: {lengths}")

        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):  # type: ignore[arg-type]
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!"
            )

        if generator is None:
            indices = list(range(0, sum(lengths)))  # type: ignore[arg-type, call-overload]
        else:
            indices = torch.randperm(sum(lengths), generator=generator).tolist()
        
        lengths = cast(Sequence[int], lengths)
        # res: List = []
        if isinstance(dataset, Dataset):
            return [
                Subset(dataset, indices[offset - length : offset])
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]
        else:
            return [
                [dataset[idx] for idx in indices[offset - length : offset]]
                for offset, length in zip(itertools.accumulate(lengths), lengths)
            ]
class CustomCIFAR10(CIFAR10, DatasetsBase):
    def __init__(self, *args, output_format: str=DatasetsBase.FORMAT_NONE, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        self.output_format = output_format
    def set_output_format(self, output_format: str):
        self.output_format: str = output_format
        return self
    def customize(self, selected_class: List[int]):
        tmp_data = []
        tmp_targets = []
        for d, t in zip(self.data, self.targets):
            if t in selected_class:
                tmp_data.append(d)
                tmp_targets.append(t)
        self.data, self.targets = tmp_data, tmp_targets
        return self
    def __getitem__(self, index: int) -> Tuple[int, Any, Any]:
        img, target = super().__getitem__(index=index)
        output = DatasetsBase.format_output(
            index, img, target, output_format=self.output_format)
        # print(f"output: {output}")
        return output

class IndexCIFAR10(CIFAR10, DatasetsBase):
    def __init__(self, *args, output_format: str=DatasetsBase.FORMAT_NONE, **kwargs):
        super(IndexCIFAR10, self).__init__(*args, **kwargs)
        self.output_format: str = output_format
    def __getitem__(self, index: int) -> Tuple[int, Any, Any]:
        img, target = super().__getitem__(index=index)
        output = DatasetsBase.format_output(
            index, img, target, output_format=self.output_format)
        # print(f"output: {output}")
        return output
        
class Datasets(DatasetsBase):
    ROOT: str = "ROOT"
    CIFAR10_32: str = "CIFAR10_32"
    CIFAR2_32: str = "CIFAR2_32"
    CELEBA_HQ_256: str = "CELEBA_HQ_256"
    IMAGENET_64: str = "IMAGENET_64"
    BEDROOM_64: str = "BEDROOM_64"
    
    SELECTED_CLASS: str = "SELECTED_CLASS"
    NUM_CLASSES: str = "NUM_CLASSES"
    TOTAL_NUM_CLASSES: str = "TOTAL_NUM_CLASSES"
    
    SPLIT_TRAIN: str = 'train'
    SPLIT_TEST: str = 'test'

    def __init__(self, config: Union[str, os.PathLike]):
        self.config: Dict = {}
        with open(config, mode='r', encoding='utf-8') as json_data:
            self.config = json.load(json_data)
    @staticmethod
    def get_dataset_by_split(get_dataset_fn: Callable, splits: Union[str, List[str]]):
        """
        Generate datsets by splits, if there are two splits, the function will combine two datasets.
        """
        split2train: Dict = {
            Datasets.SPLIT_TRAIN: True,
            Datasets.SPLIT_TEST: False,
        }
        if isinstance(splits, str):
            return get_dataset_fn(split=split2train[splits])
        elif isinstance(splits, list):
            ds_list = []
            for split in splits:
                ds_list.append(get_dataset_fn(split=split2train[split]))
            return ConcatDataset(ds_list)
        else:
            raise TypeError(f"Arguement splits shouldn't be {type(splits)}, only str or List[str] are acceptable.")
    def get_dataset(self,
                    name: str,
                    splits: Union[str, List[str]],
                    output_format: Literal[DatasetsBase.FORMAT_NONE, DatasetsBase.FORMAT_NO_IDX, DatasetsBase.FORMAT_HF]=DatasetsBase.FORMAT_NONE):
        """
        Get dataset according to dataset IDs
        """
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        if name == Datasets.CIFAR10_32:
            get_dataset_fn: Callable = lambda split: IndexCIFAR10(
                root=self.config[Datasets.CIFAR10_32][Datasets.ROOT],
                train=split, download=True, transform=transforms, output_format=output_format)
        elif name == Datasets.CIFAR2_32:
            get_dataset_fn: Callable = lambda split: CustomCIFAR10(
                root=self.config[Datasets.CIFAR2_32][Datasets.ROOT],
                train=split, download=True, transform=transforms, output_format=output_format
                ).customize(selected_class=self.config[Datasets.CIFAR2_32][Datasets.SELECTED_CLASS])
        else:
            raise NotImplementedError(f"Dataset {name} hasn't implemented yet.")
        return Datasets.get_dataset_by_split(get_dataset_fn=get_dataset_fn, splits=splits)
