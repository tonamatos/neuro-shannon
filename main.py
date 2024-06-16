import argparse
from train import Train
from solve import Solve
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIS Hamiltanian")
    parser.add_argument("operation", type=str, help="Operation to perform.", choices=["train", "solve"])
    parser.add_argument("input", type=Path, action="store", help="Directory containing input graphs (to be solved/trained on).")
    parser.add_argument("output", type=Path, action="store",  help="Folder in which the output (e.g. json containg statistics and solution will be stored)")

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--dropout_frac', type=float, default=5e-2)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--supervised', action="store_true", default=False)
    parser.add_argument('--pretrained', type=Path, action="store")
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--penalty_threshold', type=float, help="Avoid poor minimization due to random initialization of parameters. Avoid stuck at local minimum.")

    parser.add_argument('--p1', type=float, default=2)
    parser.add_argument('--p2', type=float, default=2)
    parser.add_argument('--c1', type=float, default=1)
    parser.add_argument('--c2', type=float, default=6)

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # TODO: Import train and solve
    if args.operation == "train":
        if args.supervised:
            Train(args).train()
        else:
            raise ValueError("Unsupervised Learning should solve directly")
    else:
        if args.supervised:
            if args.pretrained is None:
                raise ValueError("Pretrained model is required. Please train the model before process!")
            if args.penalty_threshold is None:
                raise ValueError("Please provide a penalty threshold to avoid poor solution. Please give a resonable threshold.")
            Solve(args, data=args.input).run()
        else:
            Solve(args=args).run()