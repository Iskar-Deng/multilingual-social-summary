from evaluate import load

def evaluate_bert_score(predictions, references, get_average=True):
    """
    Evaluate the BERT score of the predictions against the references.
    https://huggingface.co/spaces/evaluate-metric/bertscore

    Args:
        predictions (list): List of predicted sentences (input_text).
        references (list): List of reference sentences (summary).
        get_average (bool): Whether to return average scores or not.
        If True, returns average scores for each BERT score type.
        If False, returns individual scores for each prediction-reference pair.
        Default is True.

    Returns:
        dict: A dictionary containing precision, recall, and F1 scores.
    """
    # Load the BERTScore metric
    scorer = load("bertscore")

    # Compute BERTScore
    results = scorer.compute(predictions=predictions, references=references, model_type="xlm-roberta-large", lang="en", rescale_with_baseline=True)
    # results = scorer.compute(predictions=predictions, references=references, model_type="bert-base-multilingual-cased")
    # Note: List of models can be found here: https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?gid=0#gid=0

    if get_average:
        # Extract precision, recall, and F1 scores
        precision = results["precision"]
        recall = results["recall"]
        f1 = results["f1"]
        # Average the scores across all predictions
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1 = sum(f1) / len(f1)

    return results
