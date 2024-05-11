import argparse
import logzero

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIS Hamiltanian")
    parser.add_argument("operation", type=str, help="Operation to perform.", choices=["train", "solve"])
    parser.add_argument("input", type=Path, action="store", help="Directory containing input graphs (to be solved/trained on).")
    parser.add_argument("output", type=Path, action="store",  help="Folder in which the output (e.g. json containg statistics and solution will be stored)")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout_frac', type=float, default=5e-2)

    args = parser.parse_args()

    # TODO: Import train and solve
    if args.operation == "train":
        train(args)
    else:
        solve(args)