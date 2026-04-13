"""Extract sequence embeddings from an ESM-2 model for peptides or microproteins."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from hmpa_bioactivity.embeddings import extract_esm2_embeddings, read_fasta_records


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument(
        "--model-name",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument("--output-pt", required=True, help="Output tensor file.")
    parser.add_argument(
        "--output-ids",
        help="Optional text file with one sequence id per line in embedding order.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy for token embeddings.",
    )
    parser.add_argument(
        "--save-format",
        choices=["tensor", "dict"],
        default="tensor",
        help="Save embeddings either as a matrix tensor or an id-to-tensor dict.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the tokenizer and model only from local cached files.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    records = read_fasta_records(args.fasta)
    if not records:
        raise ValueError("No FASTA records were found in the input file.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    ids, embeddings = extract_esm2_embeddings(
        records=records,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        pooling=args.pooling,
        max_length=args.max_length,
    )

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_format == "tensor":
        torch.save(embeddings, output_path)
    else:
        embedding_dict = {}
        for index, item_id in enumerate(ids):
            embedding_dict[item_id] = embeddings[index]
        torch.save(embedding_dict, output_path)

    if args.output_ids:
        output_ids_path = Path(args.output_ids)
        output_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_ids_path, "w", encoding="utf-8") as handle:
            for item_id in ids:
                handle.write(item_id + "\n")

    print("Saved embeddings to {}".format(output_path))
    if args.output_ids:
        print("Saved id order to {}".format(args.output_ids))


if __name__ == "__main__":
    main()
