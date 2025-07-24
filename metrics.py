import argparse
import json
import sys

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

nltk.download('wordnet', quiet=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run metrics calculation"
    )
    parser.add_argument("input_json", help="Path to input JSON file with queries.")
    args = parser.parse_args()

    # Load input JSON file
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    meteor_scores = []
    bleu1_scores, bleu2_scores, bleu4_scores, corpus_bleu_scores = [], [], [], []
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []

    smoother = SmoothingFunction().method1

    for sample in data:
        true_ans = sample["response"]
        pred_ans = sample["pred_response"]

        splited_true_ans = [true_ans.split()]
        splited_pred_ans = pred_ans.split()
    
        # METEOR
        meteor_scores.append(meteor_score(splited_true_ans, splited_pred_ans))
   
        rouge = Rouge()
        rouge_score = rouge.get_scores(true_ans, pred_ans, avg=True)

        # BLEU 
        bleu1_scores.append(sentence_bleu(
            splited_true_ans, splited_pred_ans, 
            weights=(1, 0, 0, 0), 
            smoothing_function=smoother,
        ))
        bleu2_scores.append(sentence_bleu(
            splited_true_ans, splited_pred_ans, 
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoother
        ))
        bleu4_scores.append(sentence_bleu(
            splited_true_ans, splited_pred_ans, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother,
        ))

        # Rouge
        rouge1_scores.append(rouge_score['rouge-1']['f'])
        rouge2_scores.append(rouge_score['rouge-1']['f'])
        rougel_scores.append(rouge_score['rouge-1']['f'])

    return {
        "meteor": np.mean(meteor_scores),
        "bleu1": np.mean(bleu1_scores),
        "bleu2": np.mean(bleu2_scores),
        "bleu4": np.mean(bleu4_scores),
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougel": np.mean(rougel_scores),
    }


if __name__ == "__main__":
    main()
