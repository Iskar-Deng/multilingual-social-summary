from LaSE import LaSEScorer 

scorer = LaSEScorer()

def evaluate_LaSE(predictions, references, target_lang=None, get_all_scores=False):
    """
    Evaluate the LaSE score of the predictions against the references.
    https://github.com/csebuetnlp/CrossSum/tree/main/LaSE

    Args:
        predictions (list): List of predicted sentences (summary).
        references (list): List of reference sentences (input_text).
        target_lang (str): The target language for the predictions.
            Default is None.
            If None, the language confidence score will be set to 1.0.
        get_all_scores (bool): Whether to return all scores or just the LaSE score.
            Default is False. 
            If True, returns a dictionary with ms, lc, lp and LaSE scores.
            If False, returns the LaSE score as a float.

    Returns:
        Depends on get_all_scores
    """
    scores = scorer.score(
        references,
        predictions,
        target_lang=target_lang
    )

    if get_all_scores:
        return {score_type: float(score) for score_type, score in scores._asdict().items()}
    else:
        return float(scores.LaSE)
    