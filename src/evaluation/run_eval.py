import json
import argparse
import sys
import bert_score
from eval_rouge import evaluate_rouge


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions based on a json file of objects with 'input_text' as reference and another key as prediction."
    )
    parser.add_argument("json_path", help="Path to the json file containing evaluation data.")
    parser.add_argument("--bert", action="store_true", help="Evaluate using BERTScore metric.")
    parser.add_argument("--rouge", action="store_true", help="Evaluate using ROUGE-1, ROUGE-2, ROUGE-L metric.")
    parser.add_argument("--get_average", action="store_true", help="Get average scores for each metric.")
    parser.add_argument("--hide_individual_scores", action="store_true", help="Hide individual scores for each evaluation.")


    args = parser.parse_args()

    if not args.bert and not args.rouge:
        print("No evaluation metric selected. Use --bert and/or --rouge.")
        sys.exit(1)

    score_types = ["precision", "recall", "f1"]

    if args.bert:
        bert_scores = {score_type: [] for score_type in score_types}
    
    if args.rouge:
        rouge_types = ["rouge1", "rouge2", "rougeL"]
        rouge_scores = {rouge_type: {score_type: [] for score_type in score_types} for rouge_type in rouge_types}
    
    num_lines = 0

    with open(args.json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            num_lines += 1

            for key in data.keys():
                if key == "input_text":
                    reference = data[key]
                else:
                    # Assuming the key that is not "input_text" is the prediction
                    prediction = data[key]

            if args.bert:
                scores = bert_score.score(
                    [prediction], [reference], model_type="xlm-roberta-large"
                )
                for score_type, score in zip(score_types, scores):
                    bert_scores[score_type].append(score.item())


            if args.rouge:
                scores = evaluate_rouge(prediction, reference, rouge_types=rouge_types)
                for rouge_type, metrics in scores.items():
                    for score_type in score_types:
                        rouge_scores[rouge_type][score_type].append(metrics[score_type])

    if not args.hide_individual_scores:
        if args.bert:
            print("BERT scores:")
            print(bert_scores)
            print()

        if args.rouge:
            print("ROUGE scores:")
            for rouge_type, metrics in rouge_scores.items():
                print(f"{rouge_type}:")
                for score_type in score_types:
                    print(f"  {score_type}: {metrics[score_type]}")
            print()
    
    if args.get_average:
        if args.bert:
            avg_bert_scores = {score_type: sum(bert_scores[score_type]) / num_lines for score_type in score_types}
            print(f"\nBERT Score averages across all summarizations:\n{avg_bert_scores}")
            print()

        if args.rouge:
            avg_rouge_scores = {rouge_type: {score_type: sum(rouge_scores[rouge_type][score_type]) / num_lines for score_type in score_types} for rouge_type in rouge_types}
            print("\nROUGE averages across all summarizations:")
            for rouge_type, metrics in rouge_scores.items():
                print(f"{rouge_type} averages:")
                for score_type in score_types:
                    print(f"{score_type}: {avg_rouge_scores[rouge_type][score_type]:.4f}")
            print()


if __name__ == "__main__":
    main()
