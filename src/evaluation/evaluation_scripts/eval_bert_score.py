import bert_score
from transformers import logging as hf_logging

# Suppress warnings from the transformers library
hf_logging.set_verbosity_error()

def evaluate_bert_score(prediction, reference, get_all_scores=False):
    """
    Evaluate the BERT score of the predictions against the references.
    https://huggingface.co/spaces/evaluate-metric/bertscore
    https://github.com/Tiiiger/bert_score

    Args:
        predictions (list): List of predicted sentences (input_text).
        references (list): List of reference sentences (summary).
        get_all_scores (bool): Whether to return all scores or just the F1 score.
            Default is False. 
            If True, returns a dictionary with precision, recall, and F1 scores.
            If False, returns the F1 score as a float.

    Returns:
        Depends on get_all_scores
    """

    # Compute BERTScore
    scores = bert_score.score([prediction], [reference], model_type="xlm-roberta-large")
    # results = scorer.compute(predictions=predictions, references=references, model_type="bert-base-multilingual-cased")
    # Note: List of models can be found here: https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?gid=0#gid=0

    if get_all_scores:
        return scores
    else:
        return scores["f1"]
