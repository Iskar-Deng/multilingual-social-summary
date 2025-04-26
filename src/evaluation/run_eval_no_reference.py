import json
import argparse
import sys
from eval_LaSE import evaluate_LaSE


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions based on a json file of objects with 'input_text' as reference and another key as prediction."
    )

    parser.add_argument("json_path", help="Path to the json file containing evaluation data.")
    parser.add_argument("--LaSE", action="store_true", help="Evaluate using LaSE metric.")
    parser.add_argument("--get_average", action="store_true", help="Get average scores for each metric.")
    parser.add_argument("--hide_individual_scores", action="store_true", help="Hide individual scores for each evaluation.")

    args = parser.parse_args()

    if not args.LaSE:
        print("No evaluation metric selected. Use --LaSE.")
        sys.exit(1)
    
    if args.LaSE:
        LaSE_score_types = ["LaSE"]  # Ommitting "ms", "lc", "lp" scores for simplicity. LaSE is a combination of all three.
        LaSE_scores = {score_type: [] for score_type in LaSE_score_types}
    
    num_lines = 0

    with open(args.json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            num_lines += 1

            if "summary_text" in data.keys() and "input_text" in data.keys():
                reference = data["input_text"]
                prediction = data["summary_text"]
            else:
                print("'summary_text' or 'input_text' not found in the data.")
                sys.exit(1)

            if args.LaSE:
                scores = evaluate_LaSE(prediction, reference, get_all_scores=True)
                for score_type, score in scores.items():
                    if score_type in LaSE_score_types:
                       LaSE_scores[score_type].append(float(score))

    if not args.hide_individual_scores:
        if args.LaSE:
            print("LaSE scores:")
            print(LaSE_scores)
            print()
    
    if args.get_average:
        if args.LaSE:
            avg_LaSE_scores = {score_type: sum(LaSE_scores[score_type]) / num_lines for score_type in LaSE_score_types}
            print(f"\nLaSE Score averages across all summarizations:\n{avg_LaSE_scores}")
            print()


if __name__ == "__main__":
    main()
