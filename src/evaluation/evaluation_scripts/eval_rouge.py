from rouge_score import rouge_scorer
from transformers import XLMRobertaTokenizer

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

def evaluate_rouge(prediction, reference, rouge_types=["rouge1", "rouge2", "rougeL"], get_all_scores=False):
    """
    Evaluate the ROUGE score of the prediction against the reference.

    Args:
        prediction (str): List of predicted sentences (summary).
        reference (str): List of reference sentences (baseline).
        rouge_types (list): List of ROUGE types to compute.
            Default is ["rouge1", "rouge2", "rougeL"].
        get_all_scores (bool): Whether to return all scores or just the F1 scores.
            Default is False. 
            If True, returns a dictionary with precision, recall, and F1 scores for each ROUGE type.
            If False, returns a tuple of F1 scores for the specified ROUGE types.

    Returns:
        Depends on get_all_scores
    """

    # Load the ROUGE metric
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)

    # Compute ROUGE
    results = {}

    # Compute ROUGE scores
    scores = scorer.score(target=reference, prediction=prediction)

    for rouge_type, score in scores.items():
        score_as_dict = score._asdict()
        score_as_dict["f1"] = score_as_dict.pop("fmeasure")

        results[rouge_type] = score_as_dict

    if get_all_scores:
        return results
    else:
        return tuple([score["f1"] for rouge_type, score in results.items() if rouge_type in rouge_types])
    

