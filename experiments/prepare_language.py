"""Materialize and tokenize the SlimPajama pools used in the paper."""
import argparse
from functools import partial
from pathlib import Path

from experiments.language_data import concat_chunk


def require_hf_dependencies():
    try:
        from datasets import Dataset, load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Secret Sauce data preparation requires `datasets` and "
            "`transformers`. Install the language dependency extra first."
        ) from exc
    return Dataset, load_dataset, AutoTokenizer


def tokenizer_path_name(tokenizer_name: str) -> str:
    return tokenizer_name.replace("/", "_")


def tokenize_batched(examples, tokenizer):
    if tokenizer.eos_token is None:
        raise ValueError("The selected tokenizer must define eos_token.")
    texts = [
        f"{text}{tokenizer.eos_token}" if text else text
        for text in examples["text"]
    ]
    output = tokenizer(
        texts,
        add_special_tokens=False,
        return_special_tokens_mask=False,
        return_attention_mask=False,
    )
    return {"input_ids": output["input_ids"]}


def iterate_dataset(dataset):
    yield from dataset


def build_chunked_dataset(
    raw_dataset,
    tokenizer,
    seq_len: int,
    target_chunks: int,
    map_batch_size: int,
    num_proc: int,
    raw_shuffle_seed: int,
    chunk_shuffle_seed: int,
):
    raw_dataset = raw_dataset.shuffle(seed=raw_shuffle_seed)
    tokenized_dataset = raw_dataset.map(
        partial(tokenize_batched, tokenizer=tokenizer),
        remove_columns=raw_dataset.column_names,
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
    )
    chunked_dataset = tokenized_dataset.map(
        partial(concat_chunk, max_seq_length=seq_len + 1),
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
    )
    if len(chunked_dataset) < target_chunks:
        raise RuntimeError(
            f"Source pool produced {len(chunked_dataset):,} chunks; "
            f"requested {target_chunks:,}."
        )
    chunked_dataset = chunked_dataset.shuffle(seed=chunk_shuffle_seed)
    chunked_dataset = chunked_dataset.select(range(target_chunks))
    chunked_dataset.set_format("torch")
    return chunked_dataset


def prepare_split(args, dataset_class, load_dataset, auto_tokenizer):
    tokenizer_name = tokenizer_path_name(args.tokenizer)
    output_root = args.out_path / f"tokenized_{tokenizer_name}"
    split_name = args.output_split or args.dataset_split
    split_path = output_root / f"ctx_{args.seq_len}" / split_name
    if split_path.exists():
        raise FileExistsError(split_path)

    streaming_dataset = load_dataset(
        args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=True,
        cache_dir=str(args.cache_path),
    )
    streaming_dataset = streaming_dataset.take(args.nrows)
    raw_dataset = dataset_class.from_generator(
        partial(iterate_dataset, streaming_dataset),
        features=streaming_dataset.features,
        cache_dir=str(args.cache_path),
    )
    if len(raw_dataset) != args.nrows:
        raise RuntimeError(
            f"Source split ended at {len(raw_dataset):,} documents; "
            f"requested {args.nrows:,}."
        )

    tokenizer = auto_tokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = int(1e30)
    chunked_dataset = build_chunked_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        target_chunks=args.target_chunks,
        map_batch_size=args.map_batch_size,
        num_proc=args.num_proc,
        raw_shuffle_seed=args.raw_shuffle_seed,
        chunk_shuffle_seed=args.chunk_shuffle_seed,
    )
    chunked_dataset.save_to_disk(split_path)

    tokenizer.model_max_length = args.seq_len
    tokenizer_path = output_root / "tokenizer"
    if not tokenizer_path.exists():
        tokenizer.save_pretrained(tokenizer_path)
    return chunked_dataset, len(raw_dataset), split_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Secret Sauce/plainLM-style dataset split."
    )
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path.home() / ".cache/huggingface/datasets",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="gmongaras/SlimPajama-627B_Reupload",
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--output-split", type=str, default=None)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--target-chunks", type=int, required=True)
    parser.add_argument("--nrows", type=int, required=True)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--map-batch-size", type=int, default=1024)
    parser.add_argument("--raw-shuffle-seed", type=int, default=1996)
    parser.add_argument("--chunk-shuffle-seed", type=int, default=96)
    args = parser.parse_args()
    args.out_path = args.out_path.expanduser()
    args.cache_path = args.cache_path.expanduser()
    if args.target_chunks <= 0:
        parser.error("--target-chunks must be positive")
    if args.nrows <= 0:
        parser.error("--nrows must be positive")
    return args


def main():
    args = parse_args()
    args.out_path.mkdir(parents=True, exist_ok=True)
    dataset_class, load_dataset, auto_tokenizer = require_hf_dependencies()
    dataset, source_documents, split_name = prepare_split(
        args,
        dataset_class,
        load_dataset,
        auto_tokenizer,
    )
    print(f"{split_name.title()} source documents: {source_documents:,}")
    print(f"{split_name.title()} rows: {len(dataset):,}")
    print(f"{split_name.title()} prediction tokens: {len(dataset) * args.seq_len:,}")


if __name__ == "__main__":
    main()
