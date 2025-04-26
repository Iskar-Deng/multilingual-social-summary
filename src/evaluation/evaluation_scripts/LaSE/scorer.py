import torch
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import namedtuple
from transformers import XLMRobertaTokenizer
import langid

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

logger = logging.getLogger(__name__)
LaSEResult = namedtuple("LaSEResult", ("ms", "lc", "lp", "LaSE"))

class LaSEScorer(object):

    def __init__(self, device=None, cache_dir=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.labse_model = SentenceTransformer('LaBSE', device=device, cache_folder=cache_dir)

    def _score_ms(self, targets, predictions, batch_size):
        """Computes batched meaning similarity score"""

        embeddings = self.labse_model.encode(targets + predictions, batch_size=batch_size, show_progress_bar=False)
        return (embeddings[:len(targets)] * embeddings[len(targets):]).sum(axis=1)

    def _score_lc(self, predictions, target_lang):
        """Computes batched language confidence score using langid"""
        langid_scores = [langid.classify(pred)[1] for pred in predictions]
        
        if target_lang:
            target_lang_score = [score if langid.classify(pred)[0] == target_lang else 1.0 for pred, score in zip(predictions, langid_scores)]
            return target_lang_score
        else:
            return langid_scores

    def _score_lp(self, targets, predictions, target_lang, alpha):
        """Computes batched length penalty score"""
        token_counts = np.asarray([len(tokenizer(s)) for s in targets + predictions])
        target_token_counts = token_counts[:len(targets)]
        prediction_token_counts = token_counts[len(targets):]

        fractions = 1 - (prediction_token_counts / (target_token_counts + alpha))
        return np.exp(fractions * (fractions <= 0.))

    def batched_score(self, targets, predictions, target_lang=None, batch_size=32, alpha=6):
        assert len(targets) == len(predictions)
        batch_size = min(batch_size, len(targets))

        ms_scores = self._score_ms(targets, predictions, batch_size)
        lc_scores = self._score_lc(predictions, target_lang)
        lp_scores = self._score_lp(targets, predictions, target_lang, alpha)
        
        return [
            LaSEResult(ms, lc, lp, ms * lc * lp)
            for ms, lc, lp in zip(ms_scores, lc_scores, lp_scores)
        ]
    
    def score(self, target, prediction, target_lang=None, alpha=6):
        return self.batched_score(
            [target], [prediction], target_lang, 1, alpha
        )[0]
