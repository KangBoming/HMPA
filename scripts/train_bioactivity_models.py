"""Train task-wise peptide bioactivity classifiers from precomputed embeddings."""

import argparse
from pathlib import Path

from hmpa_bioactivity.io_utils import load_feature_tensor, load_label_table
from hmpa_bioactivity.training import TrainingConfig, train_all_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", required=True, help="Path to the peptide multi-label CSV file.")
    parser.add_argument("--feature-pt", required=True, help="Path to the precomputed feature tensor (.pt).")
    parser.add_argument("--output-dir", required=True, help="Directory for trained models and performance summaries.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = load_label_table(args.label_csv)
    features = load_feature_tensor(args.feature_pt)

    config = TrainingConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_folds=args.num_folds,
        random_seed=args.random_seed,
        device=args.device,
    )

    performance_df = train_all_tasks(labels, features, Path(args.output_dir), config)
    print(f"Saved performance summary to {Path(args.output_dir) / 'performance.csv'}")
    print(f"Trained {performance_df['Task'].nunique()} tasks.")


if __name__ == "__main__":
    main()
