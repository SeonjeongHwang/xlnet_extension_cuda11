"""Official evaluation script for CoQA.

The code is based partially on SQuAD 2.0 evaluation script.
"""
import argparse
import json
import re
import string
import sys

from collections import Counter, OrderedDict

OPTS = None

class CoQAEvaluator():

    def __init__(self, gold_file):
        self.gold_data = CoQAEvaluator.gold_answers_to_dict(gold_file)

    @staticmethod
    def gold_answers_to_dict(gold_file):
        dataset = json.load(open(gold_file))
        gold_dict = {}
        for conv in dataset:
            eid = conv["eid"]
            if eid in gold_dict:
                print("duplicate examples", eid)
            gold_dict[eid] = conv["current_qa"]["answer"]

        return gold_dict

    @staticmethod
    def preds_to_dict(pred_file):
        preds = json.load(open(pred_file))
        pred_dict = {}
        for pred in preds:
            pred_dict[pred['eid']] = pred['answer']
        return pred_dict

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return CoQAEvaluator.normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(CoQAEvaluator.normalize_answer(a_gold) == CoQAEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = CoQAEvaluator.get_tokens(a_gold)
        pred_toks = CoQAEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def get_scores(pred_data):
        scores = []
        for eid, pred_answer in pred_data.items():
            if not eid in self.gold_data:
                print("cannot find eid:", eid)
            else:
                gold_answer = self.gold_data[eid]
                f1 = compute_f1(gold_answer, pred_answer)
                em = compute_exact(gold_answer, pred_answer)
                scores.append({"eid": eid,
                               "f1": f1,
                               "em": em})
        return scores

def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for CoQA.')
    parser.add_argument('--data-file', dest="data_file", help='Input data JSON file.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    parser.add_argument("--out-file", dest="out_file", help="result file.")
    parser.add_argument('--verbose', '-v', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    evaluator = CoQAEvaluator(OPTS.data_file)

    if OPTS.pred_file:
        with open(OPTS.pred_file, "r") as f:
            pred_data = CoQAEvaluator.preds_to_dict(OPTS.pred_file)
        scores = evaluator.get_scores(pred_data)
        with open(OPTS.out_file, "w") as fout:
            json.dump(scores, fout, indent=1)
        print("Done")

if __name__ == '__main__':
    OPTS = parse_args()
    main()
