from transformers import XLMRobertaTokenizer
from rouge_score import rouge_scorer

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")


def evaluate_rouge(prediction, reference, rouge_types=["rouge1", "rouge2", "rougeL"]):
    """
    Evaluate the ROUGE score of the prediction against the reference.

    Args:
        prediction (str): List of predicted sentences (summary).
        reference (str): List of reference sentences (input text).

    Returns:
        dict: A dictionary with nested dictionaries for each ROUGE type, each, containing precision, recall, and F1 scores.
    """

    # Custom tokenizer function using XLM-R tokenizer (basic word-like tokens)
    def tokenize_with_xlm_roberta(text):
        tokens = tokenizer.tokenize(text)
        # Optionally filter out special tokens or subword symbols (like "▁")
        clean_tokens = [token.replace("▁", "") for token in tokens if token not in tokenizer.all_special_tokens]
        return clean_tokens

    # Load the ROUGE metric
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False)

    # Compute ROUGE
    results = {}

    pred_tokens = tokenize_with_xlm_roberta(prediction)
    ref_tokens = tokenize_with_xlm_roberta(reference)

    # Compute ROUGE scores
    scores = scorer.score(target=" ".join(ref_tokens), prediction=" ".join(pred_tokens))

    for rouge_type, score in scores.items():
        score_as_dict = score._asdict()
        score_as_dict["f1"] = score_as_dict.pop("fmeasure")

        results[rouge_type] = score_as_dict

    return results

