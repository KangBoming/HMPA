"""Predict microprotein bioactivity scores from precomputed embeddings."""

import argparse

from hmpa_bioactivity.inference import run_inference
from hmpa_bioactivity.io_utils import load_embedding_matrix, read_fasta_ids, read_id_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-pt", required=True, help="Tensor or dict-style embedding file.")
    parser.add_argument("--performance-csv", required=True, help="Training performance CSV with best_model_path.")
    parser.add_argument("--models-root", required=True, help="Directory that contains one subdirectory per task.")
    parser.add_argument("--output-path", required=True, help="Output .csv or .pkl file.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--fasta",
        help="FASTA file used to define input id order when the embedding file stores a dict.",
    )
    parser.add_argument(
        "--id-file",
        help="Plain text file with one id per line. Alternative to --fasta.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ids = None
    if args.fasta:
        ids = read_fasta_ids(args.fasta)
    elif args.id_file:
        ids = read_id_list(args.id_file)

    item_ids, features = load_embedding_matrix(args.embedding_pt, ids=ids)
    run_inference(
        features=features,
        item_ids=item_ids,
        performance_csv=args.performance_csv,
        models_root=args.models_root,
        output_path=args.output_path,
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()
