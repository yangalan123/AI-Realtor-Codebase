import argparse
import pickle


SYNTHETIC_RATINGS = {
    "basic gpt-4o": 820.0,
    "human gpt-4o": 1080.0,
    "none gpt-4o-mini": 910.0,
    "none gpt-4o": 980.0,
    "highlight gpt-4o-mini": 1010.0,
    "highlight gpt-4o": 1075.0,
    "highlight_raw_preference gpt-4o-mini": 1090.0,
    "highlight_raw_preference gpt-4o": 1140.0,
    "highlight + preference + surprisal gpt-4o-mini": 1185.0,
    "highlight + preference + surprisal gpt-4o": 1240.0,
    "gpt4o-mini sft gpt-4o": 1040.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a non-sensitive synthetic ratings pickle with the same model-key "
            "schema expected by benchmark/elo_plot.py."
        )
    )
    parser.add_argument("--output", default="ratings.synthetic.pkl", help="Output pickle path.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.output, "wb") as f:
        pickle.dump(SYNTHETIC_RATINGS, f)
    print(f"Wrote synthetic ratings to {args.output}")


if __name__ == "__main__":
    main()
