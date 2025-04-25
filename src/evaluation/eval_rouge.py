from rouge_score import rouge_scorer

def evaluate_rouge(prediction, reference, rouge_types=["rouge1", "rouge2", "rougeL"]):
    """
    Evaluate the ROUGE score of the prediction against the reference.

    Args:
        prediction (str): List of predicted sentences (summary).
        reference (str): List of reference sentences (baseline).

    Returns:
        dict: A dictionary with nested dictionaries for each ROUGE type, each, containing precision, recall, and F1 scores.
    """

    # Load the ROUGE metric
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)

    # Compute ROUGE
    results = {}

    # Compute ROUGE scores
    scores = scorer.score(target=reference, prediction=prediction)

    for rouge_type, score in scores.items():
        score_as_dict = score._asdict()
        score_as_dict["f1"] = score_as_dict.pop("fmeasure")

        results[rouge_type] = score_as_dict

    return results

