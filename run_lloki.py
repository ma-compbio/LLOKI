import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scGPT tasks")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory for input data")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for saving output")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory for model")    
    parser.add_argument('--name', type=str, default='merfish1100', help="Name for this run")
    parser.add_argument('--k', type=int, default=40, help="K for KNN")
    parser.add_argument('--iter', type=int, default=40, help="Number of iterations")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter")

    parser.add_argument('--device', type=int, default=0, help="CUDA device ID (default: 0)")


    args = parser.parse_args()
    args.n_runs=0
    args.drop_rate=0

    main(args, data_dir=args.data_dir, output_dir=args.output_dir, model_dir=args.model_dir)
