"""Token rows, shifted targets, and data ordering used in the language runs."""
from itertools import chain
from typing import Any, Dict, List, Optional

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def require_datasets():
    try:
        from datasets import Dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Secret Sauce data loading requires the `datasets` package. "
            "Install the language dependency extra: pip install -e '.[language]'."
        ) from exc
    return Dataset, load_from_disk


def intra_doc_causal_mask(
        doc_boundaries: List[int],
        max_seq_length: int,
        device="cpu",
) -> torch.Tensor:
    if sum(doc_boundaries) != max_seq_length:
        raise ValueError("Sum of doc_boundaries does not match max_seq_length.")
    sub_masks = [
        torch.tril(torch.ones(
            (segment_length, segment_length),
            dtype=torch.bool,
            device=device,
        ))
        for segment_length in doc_boundaries
    ]
    return torch.block_diag(*sub_masks)


def _get_docs_boundaries(
        doc_lengths: List[int],
        n_chunks: int,
        max_seq_length: int,
) -> List[List[int]]:
    doc_boundaries = [[] for _ in range(n_chunks)]
    doc_idx = 0
    current_doc_remainder = 0
    for chunk_idx in range(n_chunks):
        current_chunk_filled_length = 0
        while current_chunk_filled_length < max_seq_length:
            if current_doc_remainder == 0:
                if doc_idx < len(doc_lengths):
                    current_doc_remainder = doc_lengths[doc_idx]
                    doc_idx += 1
                else:
                    break
            space_in_chunk = max_seq_length - current_chunk_filled_length
            amount_to_add = min(current_doc_remainder, space_in_chunk)
            doc_boundaries[chunk_idx].append(amount_to_add)
            current_chunk_filled_length += amount_to_add
            current_doc_remainder -= amount_to_add
    return doc_boundaries


def concat_chunk(
        examples: Dict[str, List[Any]],
        max_seq_length: int,
) -> Dict[str, List[Any]]:
    concatenated_examples = {
        key: list(chain(*examples[key]))
        for key in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        key: [
            values[i:i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for key, values in concatenated_examples.items()
    }
    doc_lengths = [len(example) for example in examples["input_ids"]]
    result["docs_lengths"] = _get_docs_boundaries(
        doc_lengths,
        len(result["input_ids"]),
        max_seq_length,
    )
    return result


def collate_secret_sauce_batch(batch):
    output = {
        "input_ids": torch.stack([
            item["input_ids"] for item in batch
        ], dim=0)
    }
    if "docs_lengths" in batch[0]:
        output["docs_lengths"] = [
            (
                item["docs_lengths"].tolist()
                if hasattr(item["docs_lengths"], "tolist")
                else list(item["docs_lengths"])
            )
            for item in batch
        ]
    return output


def move_batch_to_device(
        batch,
        seq_len: int,
        device,
        intra_doc_masking: bool = False,
):
    inputs = batch["input_ids"][:, :seq_len]
    targets = batch["input_ids"][:, 1:seq_len + 1]
    if intra_doc_masking:
        masks = [
            intra_doc_causal_mask(doc_lengths, seq_len + 1, device)
            for doc_lengths in batch["docs_lengths"]
        ]
        attn_mask = torch.stack(masks, dim=0)
        attn_mask = attn_mask[:, :seq_len, :seq_len].contiguous()
    else:
        attn_mask = None
    if isinstance(device, torch.device):
        device_str = device.type
    else:
        device_str = str(device)
    if "cuda" in device_str:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)
    return inputs, targets, attn_mask


def _make_sampler(dataset, sampler_name: str, sampler_seed: Optional[int]):
    ddp = dist.is_initialized()
    if sampler_name == "sequential":
        if ddp:
            return DistributedSampler(dataset, shuffle=False, drop_last=True)
        return SequentialSampler(dataset)
    if sampler_name == "random":
        if ddp:
            return DistributedSampler(
                dataset,
                shuffle=True,
                seed=0 if sampler_seed is None else sampler_seed,
                drop_last=True,
            )
        generator = None
        if sampler_seed is not None:
            generator = torch.Generator().manual_seed(sampler_seed)
        return RandomSampler(dataset, generator=generator)
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def make_dataloader(
        dataset,
        micro_batch_size: int,
        num_workers: int,
        sampler_name: str = "sequential",
        sampler_seed: Optional[int] = None,
        drop_last: bool = False,
):
    sampler = _make_sampler(dataset, sampler_name, sampler_seed)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_secret_sauce_batch,
    )


def load_secret_sauce_dataset(path):
    Dataset, load_from_disk = require_datasets()
    dataset = load_from_disk(path)
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Expected a datasets.Dataset at {path}")
    return dataset


def make_secret_sauce_dataloaders(args):
    train_set = load_secret_sauce_dataset(args.trainset_path)
    train_loader = make_dataloader(
        train_set,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        sampler_name=args.sampler,
        sampler_seed=args.sampler_seed,
        drop_last=dist.is_initialized(),
    )
    valid_loader = None
    if args.validset_path:
        valid_set = load_secret_sauce_dataset(args.validset_path)
        if args.valid_tokens:
            valid_rows = args.valid_tokens // (args.seq_len + 1)
            valid_set = valid_set.take(valid_rows)
        valid_loader = make_dataloader(
            valid_set,
            micro_batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            sampler_name="sequential",
            sampler_seed=None,
            drop_last=True,
        )
    return train_loader, valid_loader
