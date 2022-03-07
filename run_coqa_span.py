from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('xlnet') # walkaround due to submodule absolute import...

import collections
import os
import os.path
import json
import pickle
import time
import string
import tqdm
import math

import tensorflow as tf
import numpy as np
import sentencepiece as sp
from six.moves import xrange

from tool.eval_coqa import CoQAEvaluator
from xlnet import xlnet
import function_builder
import prepro_utils
import model_utils


MAX_FLOAT = 1e30
MIN_FLOAT = -1e30
FLAGS = None

import argparse

def parse_args():
    parser = argparse.ArgumentParser('run CoQA question answering task')
    parser.add_argument('--train-original', dest="train_original", action='store_true', help='Whether to train the original coqa dataset')
    parser.add_argument('--train-original-span', dest="train_original_span", action='store_true', help='Whether to train the original coqa dataset')
    parser.add_argument('--train-generated-coqa', dest="train_generated_coqa", action='store_true', help='Whether to train the generated coqa dataset')
    parser.add_argument('--train-generated-quac', dest="train_generated_quac", action='store_true', help='Whether to train the generated quac dataset')
    
    parser.add_argument('--fine-tune', dest="fine_tune", action='store_true', help='Whether this training is for fine-tuning')
    parser.add_argument('--pretrained-steps', dest="pretrained_steps", default=None, type=int, help="the number of steps in the pre-training stage")
    
    parser.add_argument('--data-dir', dest="data_dir", required=True, default=None, type=str, help='Data directory where raw data located.')
    parser.add_argument('--output-dir', dest="output_dir", required=True, default=None, type=str, help='Output directory where processed data located.')
    parser.add_argument('--model-dir', dest="model_dir", required=True, default=None, type=str, help='Model directory where checkpoints located.')
    parser.add_argument('--export-dir', dest="export_dir", required=True, default=None, type=str, help='Export directory where saved model located.')
    
    parser.add_argument('--task-name', dest="task_name", default=None, type=str, help='The name of the task to train.')
    parser.add_argument('--model-config-path', dest="model_config_path", required=True, default=None, type=str, help='Config file of the pre-trained model.')
    parser.add_argument('--init-checkpoint', dest="init_checkpoint", required=True, default=None, type=str, help='Initial checkpoint of the pre-trained model.')
    parser.add_argument('--spiece-model-file', dest="spiece_model_file", required=True, default=None, type=str, help='Sentence Piece model path.')
    parser.add_argument('--overwrite-data', dest="overwrite_data", action='store_true', help='If False, will use cached data if available.')
    parser.add_argument('--random-seed', dest="random_seed", type=int, help="Random seed for weight initialzation.")
    parser.add_argument('--predict-tag', dest="predict_tag", default=None, type=str, help='Predict tag for predict result tracking.')
    
    parser.add_argument('--do-train', dest="do_train", action='store_true', help='Whether to run training.')
    parser.add_argument('--do-predict', dest="do_predict", action='store_true', help='Whether to run prediction.')
    parser.add_argument('--do-export', dest="do_export", action='store_true', help='Whether to run exporting.')
    
    parser.add_argument('--do-predict-span', dest="do_predict_span", action='store_true', help='Whether to run prediction.')
    parser.add_argument('--do-predict-coqa', dest="do_predict_coqa", action='store_true', help='Whether to run prediction.')
    parser.add_argument('--do-predict-quac', dest="do_predict_quac", action='store_true', help='Whether to run prediction.')
    
    parser.add_argument('--do-predict-coqa-train', dest="do_predict_coqa_train", action='store_true', help='Whether to run prediction.')
    parser.add_argument('--do-predict-quac-train', dest="do_predict_quac_train", action='store_true', help='Whether to run prediction.')
    
    parser.add_argument('--init', dest="init", default="normal", type=str, help='Initialization method; [normal, uniform]')
    parser.add_argument('--init-std', dest="init_std", default=0.02, type=str, help='Initialization std when init is normal.')
    parser.add_argument('--init-range', dest="init_range", default=0.1, type=str, help='Initialization std when init is uniform.')
    parser.add_argument('--init-global-vars', dest="init_global_vars", action='store_true', help='If true, init all global vars. If false, init trainable vars only.')
    
    parser.add_argument('--lower-case', dest="lower_case", action='store_true', help='Enable lower case nor not.')
    parser.add_argument('--num-turn', dest="num_turn", default=2, type=int, help='Number of turns')
    parser.add_argument('--doc-stride', dest="doc_stride", default=128, type=int, help='Doc stride')
    parser.add_argument('--max-seq-length', dest="max_seq_length", default=512, type=int, help='Max sequence length')
    parser.add_argument('--max-query-length', dest="max_query_length", default=128, type=int, help='Max query length')
    parser.add_argument('--max-answer-length', dest="max_answer_length", default=16, type=int, help='Max answer length')
    parser.add_argument('--train-batch-size', dest="train_batch_size", default=48, type=int, help='Total batch size for training.')
    parser.add_argument('--predict-batch-size', dest="predict_batch_size", default=32, type=int, help='Total batch size for predict.')
    
    parser.add_argument('--epochs', dest="epochs", default=2, type=int, help='Number of epochs')
    parser.add_argument('--train-steps', dest="train_steps", default=20000, type=int, help='Number of training steps')
    parser.add_argument('--warmup-steps', dest="warmup_steps", default=0, type=int, help='number of warmup steps')
    parser.add_argument('--max-save', dest="max_save", default=5, type=int, help='Max number of checkpoints to save. Use 0 to save all.')
    parser.add_argument('--save-steps', dest="save_steps", default=1000, type=int, help='Save the model for every save_steps. If None, not to save any model.')
    parser.add_argument('--shuffle-buffer', dest="shuffle_buffer", default=2048, type=int, help='Buffer size used for shuffle.')
    
    parser.add_argument('--n-best-size', dest="n_best_size", default=5, type=int, help='n best size for predictions')
    parser.add_argument('--start-n-top', dest="start_n_top", default=5, type=int, help='Beam size for span start.')
    parser.add_argument('--end-n-top', dest="end_n_top", default=5, type=int, help='Beam size for span end.')
    parser.add_argument('--target-eval_key', dest="target_eval_key", default="best_f1", type=str, help='Use has_ans_f1 for Model I.')
    
    parser.add_argument('--use-bfloat16', dest="use_bfloat16", action='store_true', help='Whether to use bfloat16.')
    parser.add_argument('--dropout', dest="dropout", default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--dropatt', dest="dropatt", default=0.1, type=float, help='Attention dropout rate.')
    parser.add_argument('--clamp-len', dest="clamp_len", default=-1, type=int, help='Clamp length')
    parser.add_argument('--summary-type', dest="summary_type", default="last", type=str, help='Method used to summarize a sequence into a vector.')

    parser.add_argument('--learning-rate', dest="learning_rate", default=3e-5, type=float, help='initial learning rate"')
    parser.add_argument('--min-lr-ratio', dest="min_lr_ratio", default=0.0, type=float, help='min lr ratio for cos decay.')
    parser.add_argument('--lr-layer-decay-rate', dest="lr_layer_decay_rate", default=0.75, type=float, help='lr[L] = learning_rate, lr[l-1] = lr[l] * lr_layer_decay_rate.')
    parser.add_argument('--clip', dest="clip", default=1.0, type=float, help='Gradient clipping')
    parser.add_argument('--weight-decay', dest="weight_decay", default=0.00, type=float, help='Weight decay rate')
    parser.add_argument('--adam-epsilon', dest="adam_epsilon", default=1e-6, type=float, help='Adam epsilon')
    parser.add_argument('--decay-method', dest="decay_method", default="poly", type=str, help='poly or cos')

    parser.add_argument('--use-tpu', dest="use_tpu", action='store_true', help='Whether to use TPU or GPU/CPU.')
    parser.add_argument('--num-hosts', dest="num_hosts", default=1, type=int, help='How many TPU hosts.')
    parser.add_argument('--num-core-per-host', dest="num_core_per_host", default=1, type=int, help='Total number of TPU cores to use.')
    parser.add_argument('--tpu-job-name', dest="tpu_job_name", default=None, type=str, help='TPU worker job name.')
    parser.add_argument('--tpu', dest="tpu", default=None, type=str, help='The Cloud TPU name to use for training.')
    parser.add_argument('--tpu-zone', dest="tpu_zone", default=None, type=str, help='GCE zone where the Cloud TPU is located in.')
    parser.add_argument('--gcp-project', dest="gcp_project", default=None, type=str, help='Project name for the Cloud TPU-enabled project.')
    parser.add_argument('--master', dest="master", default=None, type=str, help='TensorFlow master URL')
    parser.add_argument('--iterations', dest="iterations", default=1000, type=int, help='number of iterations per TPU training loop.')
    
    return parser.parse_args()
   
class InputExample(object):
    """A single CoQA example."""
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 answer_type=None,
                 answer_subtype=None,
                 is_skipped=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.answer_type = answer_type
        self.answer_subtype = answer_subtype
        self.is_skipped = is_skipped
        
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = "qas_id: %s" % (prepro_utils.printable_text(self.qas_id))
        s += ", question_text: %s" % (prepro_utils.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (prepro_utils.printable_text(self.paragraph_text))
        if self.start_position >= 0:
            s += ", start_position: %d" % (self.start_position)
            s += ", orig_answer_text: %s" % (prepro_utils.printable_text(self.orig_answer_text))
            s += ", answer_type: %s" % (prepro_utils.printable_text(self.answer_type))
            s += ", answer_subtype: %s" % (prepro_utils.printable_text(self.answer_subtype))
            s += ", is_skipped: %r" % (self.is_skipped)
        return "[{0}]\n".format(s)

class InputFeatures(object):
    """A single CoQA feature."""
    def __init__(self,
                 unique_id,
                 qas_id,
                 doc_idx,
                 token2char_raw_start_index,
                 token2char_raw_end_index,
                 token2doc_index,
                 input_ids,
                 input_mask,
                 p_mask,
                 segment_ids,
                 cls_index,
                 para_length,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.doc_idx = doc_idx
        self.token2char_raw_start_index = token2char_raw_start_index
        self.token2char_raw_end_index = token2char_raw_end_index
        self.token2doc_index = token2doc_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.para_length = para_length
        self.start_position = start_position
        self.end_position = end_position

class OutputResult(object):
    """A single CoQA result."""
    def __init__(self,
                 unique_id,
                 start_prob,
                 start_index,
                 end_prob,
                 end_index):
        self.unique_id = unique_id
        self.start_prob = start_prob
        self.start_index = start_index
        self.end_prob = end_prob
        self.end_index = end_index
        
class OutputLogits(object):
    """A single CoQA result."""
    def __init__(self,
                 unique_id,
                 start_logits,
                 end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits
        
class CoqaPipeline(object):
    """Pipeline for CoQA dataset."""
    def __init__(self,
                 data_dir,
                 task_name,
                 num_turn):
        self.data_dir = data_dir
        self.task_name = task_name
        self.num_turn = num_turn
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        example_list = [example for example in example_list if not example.is_skipped]
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_dev_span_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list, only_span=True)
        return example_list
    
    def get_g_coqa_examples(self): #### Custom ####
        """Gets a collection of 'InputExamples's for the train set."""
        data_path = os.path.join(self.data_dir, "G_CoQA.json")
        data_list = self._read_generated_data_json(data_path)
        example_list = self._get_example_from_generated_dataset(data_list)
        return example_list
    
    def get_g_quac_examples(self): #### Custom ####
        """Gets a collection of 'InputExamples's for the train set."""
        data_path = os.path.join(self.data_dir, "G_QuAC.json")
        data_list = self._read_generated_data_json(data_path)
        example_list = self._get_example_from_generated_dataset(data_list)
        return example_list
    
    def get_g_coqa_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "G_CoQA_dev.json".format(self.task_name))
        data_list = self._read_generated_data_json(data_path)
        example_list = self._get_example_from_generated_dataset(data_list)
        return example_list
    
    def get_g_quac_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "G_QuAC_dev.json".format(self.task_name))
        data_list = self._read_generated_data_json(data_path)
        example_list = self._get_example_from_generated_dataset(data_list)
        return example_list
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))
            
    def _read_generated_data_json(self,
                                  data_path): #### Custom ####
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)
                return data_list
        else:
                raise FileNotFoundError("data path not found: {0}".format(data_path))
    
    def _whitespace_tokenize(self,
                             text):
        word_spans = []
        char_list = []
        for idx, char in enumerate(text):
            if char != ' ':
                char_list.append(idx)
                continue
            
            if char_list:
                word_start = char_list[0]
                word_end = char_list[-1]
                word_text = text[word_start:word_end+1]
                word_spans.append((word_text, word_start, word_end))
                char_list.clear()
        
        if char_list:
            word_start = char_list[0]
            word_end = char_list[-1]
            word_text = text[word_start:word_end+1]
            word_spans.append((word_text, word_start, word_end))
        
        return word_spans
    
    def _char_span_to_word_span(self,
                                char_start,
                                char_end,
                                word_spans):
        word_idx_list = []
        for word_idx, (_, start, end) in enumerate(word_spans):
            if end >= char_start:
                if start <= char_end:
                    word_idx_list.append(word_idx)
                else:
                    break
        
        if word_idx_list:
            word_start = word_idx_list[0]
            word_end = word_idx_list[-1]
        else:
            word_start = -1
            word_end = -1
        
        return word_start, word_end
    
    def _search_best_span(self,
                          context_tokens,
                          answer_tokens):
        best_f1 = 0.0
        best_start, best_end = -1, -1
        search_index = [idx for idx in range(len(context_tokens)) if context_tokens[idx][0] in answer_tokens]
        for i in range(len(search_index)):
            for j in range(i, len(search_index)):
                candidate_tokens = [context_tokens[k][0] for k in range(search_index[i], search_index[j]+1) if context_tokens[k][0]]
                common = collections.Counter(candidate_tokens) & collections.Counter(answer_tokens)
                num_common = sum(common.values())
                if num_common > 0:
                    precision = 1.0 * num_common / len(candidate_tokens)
                    recall = 1.0 * num_common / len(answer_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_start = context_tokens[search_index[i]][1]
                        best_end = context_tokens[search_index[j]][2]
        
        return best_f1, best_start, best_end
    
    def _get_question_text(self,
                           history,
                           question):
        question_tokens = ['<s>'] + question["input_text"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_text_for_generated_dataset(self,
                                                 history,
                                                 question):
        question_tokens = ['<s>'] + question.split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_history(self,
                              history,
                              question,
                              answer,
                              answer_type,
                              is_skipped,
                              num_turn):
        question_tokens = []
        if answer_type != "unknown":
            question_tokens.extend(['<s>'] + question["input_text"].split(' '))
            question_tokens.extend(['</s>'] + answer["input_text"].split(' '))
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _get_question_history_for_generated_dataset(self,
                                                    history,
                                                    question,
                                                    answer,
                                                    num_turn):  #### Custom ####
        question_tokens = []
        question_tokens.extend(['<s>'] + question.split(' '))
        question_tokens.extend(['</s>'] + answer.split(' '))

        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)

        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]

        return history
    
    def _find_answer_span(self,
                          answer_text,
                          rationale_text,
                          rationale_start,
                          rationale_end):
        idx = rationale_text.find(answer_text)
        answer_start = rationale_start + idx
        answer_end = answer_start + len(answer_text) - 1
        
        return answer_start, answer_end
    
    def _match_answer_span(self,
                           answer_text,
                           rationale_start,
                           rationale_end,
                           paragraph_text):
        answer_tokens = self._whitespace_tokenize(answer_text)
        answer_norm_tokens = [CoQAEvaluator.normalize_answer(token) for token, _, _ in answer_tokens]
        answer_norm_tokens = [norm_token for norm_token in answer_norm_tokens if norm_token]
        
        if not answer_norm_tokens:
            return -1, -1
        
        paragraph_tokens = self._whitespace_tokenize(paragraph_text)
        
        if not (rationale_start == -1 or rationale_end == -1):
            rationale_word_start, rationale_word_end = self._char_span_to_word_span(rationale_start, rationale_end, paragraph_tokens)
            rationale_tokens = paragraph_tokens[rationale_word_start:rationale_word_end+1]
            rationale_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in rationale_tokens]
            match_score, answer_start, answer_end = self._search_best_span(rationale_norm_tokens, answer_norm_tokens)
            
            if match_score > 0.0:
                return answer_start, answer_end
        
        paragraph_norm_tokens = [(CoQAEvaluator.normalize_answer(token), start, end) for token, start, end in paragraph_tokens]
        match_score, answer_start, answer_end = self._search_best_span(paragraph_norm_tokens, answer_norm_tokens)
        
        if match_score > 0.0:
            return answer_start, answer_end
        
        return -1, -1
    
    def _get_answer_span(self,
                         answer,
                         answer_type,
                         paragraph_text):
        input_text = answer["input_text"].strip().lower()
        span_start, span_end = answer["span_start"], answer["span_end"]
        if span_start == -1 or span_end == -1:
            span_text = ""
        else:
            span_text = paragraph_text[span_start:span_end].lower()
        
        if input_text in span_text:
            span_start, span_end = self._find_answer_span(input_text, span_text, span_start, span_end)
        else:
            span_start, span_end = self._match_answer_span(input_text, span_start, span_end, paragraph_text.lower())
        
        if span_start == -1 or span_end == -1:
            answer_text = ""
            is_skipped = (answer_type == "span")
        else:
            answer_text = paragraph_text[span_start:span_end+1]
            is_skipped = False
        
        return answer_text, span_start, span_end, is_skipped
    
    def _get_start_position_for_generated_dataset(self,
                                                  answer_text,
                                                  span_start,
                                                  paragraph_text):
        tolerance = 6
        candidate_span_start = max(0, span_start-tolerance)
        start_position = None
        for start_idx in range(candidate_span_start, len(paragraph_text)-len(answer_text)+1):
            if paragraph_text[start_idx:start_idx+len(answer_text)] == answer_text:
                start_position = start_idx
                break
                
        if start_position == None:
            print(f"{paragraph_text[candidate_span_start:candidate_span_start+len(answer_text)+tolerance]} and {answer_text}")
        
        return start_position
    
    def _normalize_answer(self,
                          answer):
        norm_answer = CoQAEvaluator.normalize_answer(answer)
        
        if norm_answer in ["yes", "yese", "ye", "es"]:
            return "yes"
        
        if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
            return "no"
        
        return norm_answer
    
    def _get_answer_type(self,
                         question,
                         answer):
        norm_answer = self._normalize_answer(answer["input_text"])
        
        if norm_answer == "unknown" or "bad_turn" in answer:
            return "unknown", None
        
        if norm_answer == "yes":
            return "yes", None
        
        if norm_answer == "no":
            return "no", None
        
        return "span", None
    
    def _process_found_answer(self,
                              raw_answer,
                              found_answer):
        raw_answer_tokens = raw_answer.split(' ')
        found_answer_tokens = found_answer.split(' ')
        
        raw_answer_last_token = raw_answer_tokens[-1].lower()
        found_answer_last_token = found_answer_tokens[-1].lower()
        
        if (raw_answer_last_token != found_answer_last_token and
            raw_answer_last_token == found_answer_last_token.rstrip(string.punctuation)):
            found_answer_tokens[-1] = found_answer_tokens[-1].rstrip(string.punctuation)
        
        return ' '.join(found_answer_tokens)
    
    def _get_example(self,
                     data_list,
                     only_span=False):
        examples = []
        for data in data_list:
            data_id = data["id"]
            paragraph_text = data["story"]
            
            questions = sorted(data["questions"], key=lambda x: x["turn_id"])
            answers = sorted(data["answers"], key=lambda x: x["turn_id"])
            
            question_history = []
            qas = list(zip(questions, answers))
            for i, (question, answer) in enumerate(qas):
                qas_id = "{0}_{1}".format(data_id, i+1)
                
                answer_type, answer_subtype = self._get_answer_type(question, answer)
                answer_text, span_start, span_end, is_skipped = self._get_answer_span(answer, answer_type, paragraph_text)
                question_text = self._get_question_text(question_history, question)
                question_history = self._get_question_history(question_history, question, answer, answer_type, is_skipped, self.num_turn)
                
                if answer_type in ["unknown", "yes", "no"] or is_skipped:
                    continue
                
                start_position = span_start
                orig_answer_text = self._process_found_answer(answer["input_text"], answer_text)
                
                example = InputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    answer_type=answer_type,
                    answer_subtype=answer_subtype,
                    is_skipped=is_skipped)

                examples.append(example)
        
        return examples
    
    def _get_example_from_generated_dataset(self,
                                            data_list):  #### Custom ####
        examples = []
        for data in data_list:
            data_id = data["id"]
            doc_tokens = data["doc_tokens"]
            QnA_list = data["QnA"]

            question_history = []
            for i, qas in enumerate(QnA_list):
                turn_id = qas["turn_id"]
                question = qas["question"]
                answer_text = qas["answer"]["text"]
                start_token_idx = qas["answer"]["start_token_idx"]
                end_token_idx = qas["answer"]["end_token_idx"]

                qas_id = "{0}_{1}".format(data_id, i+1)

                paragraph_text = ""
                start_position = None
                for idx, token in enumerate(doc_tokens):
                    if idx == start_token_idx:
                        if idx == 0:
                            start_position = 0
                        else:
                            start_position = len(paragraph_text)+1
                    if idx == 0:
                        paragraph_text += token
                    else:
                        paragraph_text += " "+token
                        
                start_position = self._get_start_position_for_generated_dataset(answer_text, start_position, paragraph_text)

                if paragraph_text[start_position:start_position+len(answer_text)] != answer_text:
                    print(f"'{paragraph_text[start_position:start_position+len(answer_text)]}' != '{answer_text}'")

                question_text = self._get_question_text_for_generated_dataset(question_history, question)
                question_history = self._get_question_history_for_generated_dataset(question_history, question, answer_text, self.num_turn)

                example = InputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    orig_answer_text=answer_text,
                    start_position=start_position,
                    answer_type="span",
                    answer_subtype=None,
                    is_skipped=False)

                examples.append(example)
        return examples

class XLNetTokenizer(object):
    """Default text tokenizer for XLNet"""
    def __init__(self,
                 sp_model_file,
                 lower_case=False):
        """Construct XLNet tokenizer"""
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(sp_model_file)
        self.lower_case = lower_case
    
    def tokenize(self,
                 text):
        """Tokenize text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        tokenized_pieces = prepro_utils.encode_pieces(self.sp_processor, processed_text, return_unicode=False)
        return tokenized_pieces
    
    def encode(self,
               text):
        """Encode text for XLNet"""
        processed_text = prepro_utils.preprocess_text(text, lower=self.lower_case)
        encoded_ids = prepro_utils.encode_ids(self.sp_processor, processed_text)
        return encoded_ids
    
    def token_to_id(self,
                    token):
        """Convert token to id for XLNet"""
        return self.sp_processor.PieceToId(token)
    
    def id_to_token(self,
                    id):
        """Convert id to token for XLNet"""
        return self.sp_processor.IdToPiece(id)
    
    def tokens_to_ids(self,
                      tokens):
        """Convert tokens to ids for XLNet"""
        return [self.sp_processor.PieceToId(token) for token in tokens]
    
    def ids_to_tokens(self,
                      ids):
        """Convert ids to tokens for XLNet"""
        return [self.sp_processor.IdToPiece(id) for id in ids]

class XLNetExampleProcessor(object):
    """Default example processor for XLNet"""
    def __init__(self,
                 max_seq_length,
                 max_query_length,
                 doc_stride,
                 tokenizer):
        """Construct XLNet example processor"""
        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<p>", "<q>", "<cls>", "<sep>", "<pad>"]
        self.segment_vocab_map = {}
        for (i, segment_vocab) in enumerate(self.segment_vocab_list):
            self.segment_vocab_map[segment_vocab] = i
        
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.tokenizer = tokenizer
        self.unique_id = 1000000000
    
    def _generate_match_mapping(self,
                                para_text,
                                tokenized_para_text,
                                N,
                                M,
                                max_N,
                                max_M):
        """Generate match mapping for raw and tokenized paragraph"""
        def _lcs_match(para_text,
                       tokenized_para_text,
                       N,
                       M,
                       max_N,
                       max_M,
                       max_dist):
            """longest common sub-sequence
            
            f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            
            unlike standard LCS, this is specifically optimized for the setting
            because the mismatch between sentence pieces and original text will be small
            """
            f = np.zeros((max_N, max_M), dtype=np.float32)
            g = {}
            
            for i in range(N):
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0:
                        continue
                    
                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]
                    
                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]
                    
                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    
                    raw_char = prepro_utils.preprocess_text(para_text[i], lower=self.tokenizer.lower_case, remove_space=False)
                    tokenized_char = tokenized_para_text[j]
                    if (raw_char == tokenized_char and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1
            
            return f, g
        
        max_dist = abs(N - M) + 5
        for _ in range(2):
            lcs_matrix, match_mapping = _lcs_match(para_text, tokenized_para_text, N, M, max_N, max_M, max_dist)
            
            if lcs_matrix[N - 1, M - 1] > 0.8 * N:
                break
            
            max_dist *= 2
        
        mismatch = lcs_matrix[N - 1, M - 1] < 0.8 * N
        return match_mapping, mismatch
    
    def _convert_tokenized_index(self,
                                 index,
                                 pos,
                                 M=None,
                                 is_start=True):
        """Convert index for tokenized text"""
        if index[pos] is not None:
            return index[pos]
        
        N = len(index)
        rear = pos
        while rear < N - 1 and index[rear] is None:
            rear += 1
        
        front = pos
        while front > 0 and index[front] is None:
            front -= 1
        
        assert index[front] is not None or index[rear] is not None
        
        if index[front] is None:
            if index[rear] >= 1:
                if is_start:
                    return 0
                else:
                    return index[rear] - 1
            
            return index[rear]
        
        if index[rear] is None:
            if M is not None and index[front] < M - 1:
                if is_start:
                    return index[front] + 1
                else:
                    return M - 1
            
            return index[front]
        
        if is_start:
            if index[rear] > index[front] + 1:
                return index[front] + 1
            else:
                return index[rear]
        else:
            if index[rear] > index[front] + 1:
                return index[rear] - 1
            else:
                return index[front]
    
    def _find_max_context(self,
                          doc_spans,
                          token_idx):
        """Check if this is the 'max context' doc span for the token.

        Because of the sliding window approach taken to scoring documents, a single
        token can appear in multiple documents. E.g.
          Doc: the man went to the store and bought a gallon of milk
          Span A: the man went to the
          Span B: to the store and bought
          Span C: and bought a gallon of
          ...
        
        Now the word 'bought' will have two scores from spans B and C. We only
        want to consider the score with "maximum context", which we define as
        the *minimum* of its left and right context (the *sum* of left and
        right context will always be the same, of course).
        
        In the example the maximum context for 'bought' would be span C since
        it has 1 left context and 3 right context, while span B has 4 left context
        and 0 right context.
        """
        best_doc_score = None
        best_doc_idx = None
        for (doc_idx, doc_span) in enumerate(doc_spans):
            doc_start = doc_span["start"]
            doc_length = doc_span["length"]
            doc_end = doc_start + doc_length - 1
            if token_idx < doc_start or token_idx > doc_end:
                continue
            
            left_context_length = token_idx - doc_start
            right_context_length = doc_end - token_idx
            doc_score = min(left_context_length, right_context_length) + 0.01 * doc_length
            if best_doc_score is None or doc_score > best_doc_score:
                best_doc_score = doc_score
                best_doc_idx = doc_idx
        
        return best_doc_idx
    
    def convert_coqa_example(self,
                             example,
                             logging=False):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        query_tokens = []
        qa_texts = example.question_text.split('<s>')
        for qa_text in qa_texts:
            qa_text = qa_text.strip()
            if not qa_text:
                continue
            
            query_tokens.append('<s>')
            
            qa_items = qa_text.split('</s>')
            if len(qa_items) < 1:
                continue
            
            q_text = qa_items[0].strip()
            q_tokens = self.tokenizer.tokenize(q_text)
            query_tokens.extend(q_tokens)
            
            if len(qa_items) < 2:
                continue
            
            query_tokens.append('</s>')
            
            a_text = qa_items[1].strip()
            a_tokens = self.tokenizer.tokenize(a_text)
            query_tokens.extend(a_tokens)
        
        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[-self.max_query_length:]
        
        para_text = example.paragraph_text
        para_tokens = self.tokenizer.tokenize(example.paragraph_text)
        
        char2token_index = []
        token2char_start_index = []
        token2char_end_index = []
        char_idx = 0
        for i, token in enumerate(para_tokens):
            char_len = len(token)
            char2token_index.extend([i] * char_len)
            token2char_start_index.append(char_idx)
            char_idx += char_len
            token2char_end_index.append(char_idx - 1)
        
        tokenized_para_text = ''.join(para_tokens).replace(prepro_utils.SPIECE_UNDERLINE, ' ')
        
        N, M = len(para_text), len(tokenized_para_text)
        max_N, max_M = 1024, 1024
        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
        
        match_mapping, mismatch = self._generate_match_mapping(para_text, tokenized_para_text, N, M, max_N, max_M)
        
        raw2tokenized_char_index = [None] * N
        tokenized2raw_char_index = [None] * M
        i, j = N-1, M-1
        while i >= 0 and j >= 0:
            if (i, j) not in match_mapping:
                break
            
            if match_mapping[(i, j)] == 2:
                raw2tokenized_char_index[i] = j
                tokenized2raw_char_index[j] = i
                i, j = i - 1, j - 1
            elif match_mapping[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1
        
        if all(v is None for v in raw2tokenized_char_index) or mismatch:
            tf.compat.v1.logging.info("raw and tokenized paragraph mismatch detected for example: %s" % example.qas_id)
        
        token2char_raw_start_index = []
        token2char_raw_end_index = []
        for idx in range(len(para_tokens)):
            start_pos = token2char_start_index[idx]
            end_pos = token2char_end_index[idx]
            raw_start_pos = self._convert_tokenized_index(tokenized2raw_char_index, start_pos, N, is_start=True)
            raw_end_pos = self._convert_tokenized_index(tokenized2raw_char_index, end_pos, N, is_start=False)
            token2char_raw_start_index.append(raw_start_pos)
            token2char_raw_end_index.append(raw_end_pos)
        
        if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
            raw_start_char_pos = example.start_position
            raw_end_char_pos = raw_start_char_pos + len(example.orig_answer_text) - 1
            tokenized_start_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_start_char_pos, is_start=True)
            tokenized_end_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_end_char_pos, is_start=False)
            tokenized_start_token_pos = char2token_index[tokenized_start_char_pos]
            tokenized_end_token_pos = char2token_index[tokenized_end_char_pos]
            assert tokenized_start_token_pos <= tokenized_end_token_pos
        else:
            tokenized_start_token_pos = tokenized_end_token_pos = -1
        
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_para_length = self.max_seq_length - len(query_tokens) - 3
        total_para_length = len(para_tokens)
        
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        doc_spans = []
        para_start = 0
        while para_start < total_para_length:
            para_length = total_para_length - para_start
            if para_length > max_para_length:
                para_length = max_para_length
            
            doc_spans.append({
                "start": para_start,
                "length": para_length
            })
            
            if para_start + para_length == total_para_length:
                break
            
            para_start += min(para_length, self.doc_stride)
        
        feature_list = []
        for (doc_idx, doc_span) in enumerate(doc_spans):
            input_tokens = []
            segment_ids = []
            p_mask = []
            doc_token2char_raw_start_index = []
            doc_token2char_raw_end_index = []
            doc_token2doc_index = {}
            
            for i in range(doc_span["length"]):
                token_idx = doc_span["start"] + i
                
                doc_token2char_raw_start_index.append(token2char_raw_start_index[token_idx])
                doc_token2char_raw_end_index.append(token2char_raw_end_index[token_idx])
                
                best_doc_idx = self._find_max_context(doc_spans, token_idx)
                doc_token2doc_index[len(input_tokens)] = (best_doc_idx == doc_idx)
                
                input_tokens.append(para_tokens[token_idx])
                segment_ids.append(self.segment_vocab_map["<p>"])
                p_mask.append(0)
            
            doc_para_length = len(input_tokens)
            
            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<p>"])
            p_mask.append(1)
            
            # We put P before Q because during pretraining, B is always shorter than A
            for query_token in query_tokens:
                input_tokens.append(query_token)
                segment_ids.append(self.segment_vocab_map["<q>"])
                p_mask.append(1)

            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<q>"])
            p_mask.append(1)
            
            cls_index = len(input_tokens)
            
            input_tokens.append("<cls>")
            segment_ids.append(self.segment_vocab_map["<cls>"])
            p_mask.append(0)
            
            input_ids = self.tokenizer.tokens_to_ids(input_tokens)
            
            # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
            input_mask = [0] * len(input_ids)
            
            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(self.special_vocab_map["<pad>"])
                input_mask.append(1)
                segment_ids.append(self.segment_vocab_map["<pad>"])
                p_mask.append(1)
            
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(p_mask) == self.max_seq_length
            
            start_position = None
            end_position = None
            
            if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
                doc_start = doc_span["start"]
                doc_end = doc_start + doc_span["length"] - 1
                if tokenized_start_token_pos >= doc_start and tokenized_end_token_pos <= doc_end:
                    start_position = tokenized_start_token_pos - doc_start
                    end_position = tokenized_end_token_pos - doc_start
                else:
                    start_position = cls_index
                    end_position = cls_index
            else:
                start_position = cls_index
                end_position = cls_index
            
            if logging:
                tf.compat.v1.logging.info("*** Example ***")
                tf.compat.v1.logging.info("unique_id: %s" % str(self.unique_id))
                tf.compat.v1.logging.info("qas_id: %s" % example.qas_id)
                tf.compat.v1.logging.info("doc_idx: %s" % str(doc_idx))
                tf.compat.v1.logging.info("doc_token2char_raw_start_index: %s" % " ".join([str(x) for x in doc_token2char_raw_start_index]))
                tf.compat.v1.logging.info("doc_token2char_raw_end_index: %s" % " ".join([str(x) for x in doc_token2char_raw_end_index]))
                tf.compat.v1.logging.info("doc_token2doc_index: %s" % " ".join(["%d:%s" % (x, y) for (x, y) in doc_token2doc_index.items()]))
                tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.compat.v1.logging.info("p_mask: %s" % " ".join([str(x) for x in p_mask]))
                tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                printable_input_tokens = [prepro_utils.printable_text(input_token) for input_token in input_tokens]
                tf.compat.v1.logging.info("input_tokens: %s" % input_tokens)
                
                if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
                    tf.compat.v1.logging.info("start_position: %s" % str(start_position))
                    tf.compat.v1.logging.info("end_position: %s" % str(end_position))
                    answer_tokens = input_tokens[start_position:end_position+1]
                    answer_text = prepro_utils.printable_text("".join(answer_tokens).replace(prepro_utils.SPIECE_UNDERLINE, " "))
                    tf.compat.v1.logging.info("answer_text: %s" % answer_text)
                    tf.compat.v1.logging.info("answer_type: %s" % example.answer_type)
                    tf.compat.v1.logging.info("answer_subtype: %s" % example.answer_subtype)
                else:
                    tf.compat.v1.logging.info("answer_type: %s" % example.answer_type)
                    tf.compat.v1.logging.info("answer_subtype: %s" % example.answer_subtype)
            
            feature = InputFeatures(
                unique_id=self.unique_id,
                qas_id=example.qas_id,
                doc_idx=doc_idx,
                token2char_raw_start_index=doc_token2char_raw_start_index,
                token2char_raw_end_index=doc_token2char_raw_end_index,
                token2doc_index=doc_token2doc_index,
                input_ids=input_ids,
                input_mask=input_mask,
                p_mask=p_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                para_length=doc_para_length,
                start_position=start_position,
                end_position=end_position)
            
            feature_list.append(feature)
            self.unique_id += 1
        
        return feature_list
    
    def convert_examples_to_features(self,
                                     examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        for (idx, example) in tqdm.tqdm(enumerate(examples), total=len(examples)):
            feature_list = self.convert_coqa_example(example, logging=False)
            features.extend(feature_list)
        
        tf.compat.v1.logging.info("Generate %d features from %d examples" % (len(features), len(examples)))
        
        return features
    
    def save_features_as_tfrecord(self,
                                  features,
                                  output_file):
        """Save a set of `InputFeature`s to a TFRecord file."""
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        
        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        
        with tf.io.TFRecordWriter(output_file) as writer:
            for feature in features:
                features = collections.OrderedDict()
                features["unique_id"] = create_int_feature([feature.unique_id])
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_float_feature(feature.input_mask)
                features["p_mask"] = create_float_feature(feature.p_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["cls_index"] = create_int_feature([feature.cls_index])
                
                features["start_position"] = create_int_feature([feature.start_position])
                features["end_position"] = create_int_feature([feature.end_position])
                
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
    
    def save_features_as_pickle(self,
                                features,
                                output_file):
        """Save a set of `InputFeature`s to a Pickle file."""
        with open(output_file, 'wb') as file:
            pickle.dump(features, file)
    
    def load_features_from_pickle(self,
                                  input_file):
        """Load a set of `InputFeature`s from a Pickle file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError("feature file not found: {0}".format(input_file))
        
        with open(input_file, 'rb') as file:
            features = pickle.load(file)
            return features

class XLNetInputBuilder(object):
    """Default input builder for XLNet"""
    @staticmethod
    def get_input_fn(input_file,
                     seq_length,
                     is_training,
                     drop_remainder,
                     shuffle_buffer=2048,
                     num_threads=16):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "unique_id": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.float32),
            "p_mask": tf.io.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "cls_index": tf.io.FixedLenFeature([], tf.int64),
        }
        
        if is_training:
            name_to_features["start_position"] = tf.io.FixedLenFeature([], tf.int64)
            name_to_features["end_position"] = tf.io.FixedLenFeature([], tf.int64)             
        
        def _decode_record(record,
                           name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(serialized=record, features=name_to_features)
            
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, dtype=tf.int32)
                example[name] = t

            return example
        
        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]
            
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=shuffle_buffer, seed=np.random.randint(10000))
            
            d = d.apply(tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_threads,
                drop_remainder=drop_remainder))
            
            return d.prefetch(1024)
        
        return input_fn
    
    @staticmethod
    def get_serving_input_fn(seq_length):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        def serving_input_fn():
            with tf.compat.v1.variable_scope("serving"):
                features = {
                    'unique_id': tf.compat.v1.placeholder(tf.int32, [None], name='unique_id'),
                    'input_ids': tf.compat.v1.placeholder(tf.int32, [None, seq_length], name='input_ids'),
                    'input_mask': tf.compat.v1.placeholder(tf.float32, [None, seq_length], name='input_mask'),
                    'p_mask': tf.compat.v1.placeholder(tf.float32, [None, seq_length], name='p_mask'),
                    'segment_ids': tf.compat.v1.placeholder(tf.int32, [None, seq_length], name='segment_ids'),
                    'cls_index': tf.compat.v1.placeholder(tf.int32, [None], name='cls_index'),
                }
                
                return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()
        
        return serving_input_fn

class XLNetModelBuilder(object):
    """Default model builder for XLNet"""
    def __init__(self,
                 model_config,
                 use_tpu=False):
        """Construct XLNet model builder"""
        self.model_config = model_config
        self.use_tpu = use_tpu
    
    def _generate_masked_data(self,
                              input_data,
                              input_mask):
        """Generate masked data"""
        return input_data * input_mask + MIN_FLOAT * (1 - input_mask)
    
    def _generate_onehot_label(self,
                               input_data,
                               input_depth):
        """Generate one-hot label"""
        return tf.one_hot(input_data, depth=input_depth, on_value=1.0, off_value=0.0, dtype=tf.float32)
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask,
                      label_smoothing=0.0):
        """Compute optimization loss"""
        masked_predict = self._generate_masked_data(predict, predict_mask)
        masked_label = tf.cast(label, dtype=tf.int32) * tf.cast(label_mask, dtype=tf.int32)
                
        if label_smoothing > 1e-10:
            onehot_label = self._generate_onehot_label(masked_label, tf.shape(input=masked_predict)[-1])
            onehot_label = (onehot_label * (1 - label_smoothing) +
                label_smoothing / tf.cast(tf.shape(input=masked_predict)[-1], dtype=tf.float32)) * predict_mask
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label, logits=masked_predict)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label, logits=masked_predict)
        
        return loss
    
    def _create_model(self,
                      is_training,
                      input_ids,
                      input_mask,
                      p_mask,
                      segment_ids,
                      cls_index,
                      start_positions=None,
                      end_positions=None):
        """Creates XLNet-CoQA model"""
        model = xlnet.XLNetModel(
            xlnet_config=self.model_config,
            run_config=xlnet.create_run_config(is_training, True, FLAGS),
            input_ids=tf.transpose(a=input_ids, perm=[1,0]),                                                               # [b,l] --> [l,b]
            input_mask=tf.transpose(a=input_mask, perm=[1,0]),                                                             # [b,l] --> [l,b]
            seg_ids=tf.transpose(a=segment_ids, perm=[1,0]))                                                               # [b,l] --> [l,b]
        
        initializer = model.get_initializer()
        seq_len = tf.shape(input=input_ids)[-1]
        output_result = tf.transpose(a=model.get_sequence_output(), perm=[1,0,2])                                      # [l,b,h] --> [b,l,h]
        
        predicts = {}
        with tf.compat.v1.variable_scope("mrc", reuse=tf.compat.v1.AUTO_REUSE):
            with tf.compat.v1.variable_scope("start", reuse=tf.compat.v1.AUTO_REUSE):
                start_result = output_result                                                                                     # [b,l,h]
                start_result_mask = 1 - p_mask                                                                                     # [b,l]
                
                start_result = tf.compat.v1.layers.dense(start_result, units=1, activation=None,
                    use_bias=True, kernel_initializer=initializer, bias_initializer=tf.compat.v1.zeros_initializer,
                    kernel_regularizer=None, bias_regularizer=None, trainable=True, name="start_project")            # [b,l,h] --> [b,l,1]
                
                start_result = tf.squeeze(start_result, axis=-1)                                                       # [b,l,1] --> [b,l]
                start_result = self._generate_masked_data(start_result, start_result_mask)                        # [b,l], [b,l] --> [b,l]
                start_prob = tf.nn.softmax(start_result, axis=-1)                                                                  # [b,l]
                                
                if not is_training:
                    start_top_prob, start_top_index = tf.nn.top_k(start_prob, k=FLAGS.start_n_top)                # [b,l] --> [b,k], [b,k]
                    predicts["start_prob"] = start_top_prob
                    predicts["start_index"] = start_top_index                    
            
            with tf.compat.v1.variable_scope("end", reuse=tf.compat.v1.AUTO_REUSE):
                if is_training:
                    # During training, compute the end logits based on the ground truth of the start position
                    start_index = self._generate_onehot_label(tf.expand_dims(start_positions, axis=-1), seq_len)         # [b] --> [b,1,l]
                    feat_result = tf.matmul(start_index, output_result)                                     # [b,1,l], [b,l,h] --> [b,1,h]
                    feat_result = tf.tile(feat_result, multiples=[1,seq_len,1])                                      # [b,1,h] --> [b,l,h]
                    
                    end_result = tf.concat([output_result, feat_result], axis=-1)                          # [b,l,h], [b,l,h] --> [b,l,2h]
                    end_result_mask = 1 - p_mask                                                                                   # [b,l]
                    
                    end_result = tf.compat.v1.layers.dense(end_result, units=self.model_config.d_model, activation=tf.tanh,
                        use_bias=True, kernel_initializer=initializer, bias_initializer=tf.compat.v1.zeros_initializer,
                        kernel_regularizer=None, bias_regularizer=None, trainable=True, name="end_modeling")        # [b,l,2h] --> [b,l,h]
                    
                    end_result = tf.keras.layers.LayerNormalization(epsilon=1e-12, center=True, scale=True, 
                                                                    trainable=True, name="end_norm")(end_result)     # [b,l,h] --> [b,l,h]
                    
                    end_result = tf.compat.v1.layers.dense(end_result, units=1, activation=None,
                        use_bias=True, kernel_initializer=initializer, bias_initializer=tf.compat.v1.zeros_initializer,
                        kernel_regularizer=None, bias_regularizer=None, trainable=True, name="end_project")          # [b,l,h] --> [b,l,1]
                    
                    end_result = tf.squeeze(end_result, axis=-1)                                                       # [b,l,1] --> [b,l]
                    end_result = self._generate_masked_data(end_result, end_result_mask)                          # [b,l], [b,l] --> [b,l]
                    end_prob = tf.nn.softmax(end_result, axis=-1)                                                                  # [b,l]
                    
                else:
                    # During inference, compute the end logits based on beam search
                    start_index = self._generate_onehot_label(start_top_index, seq_len)                                # [b,k] --> [b,k,l]
                    feat_result = tf.matmul(start_index, output_result)                                     # [b,k,l], [b,l,h] --> [b,k,h]
                    feat_result = tf.expand_dims(feat_result, axis=1)                                              # [b,k,h] --> [b,1,k,h]
                    feat_result = tf.tile(feat_result, multiples=[1,seq_len,1,1])                                # [b,1,k,h] --> [b,l,k,h]
                    
                    end_result = tf.expand_dims(output_result, axis=-2)                                            # [b,l,h] --> [b,l,1,h]
                    end_result = tf.tile(end_result, multiples=[1,1,FLAGS.start_n_top,1])                        # [b,l,1,h] --> [b,l,k,h]
                    end_result = tf.concat([end_result, feat_result], axis=-1)                       # [b,l,k,h], [b,l,k,h] --> [b,l,k,2h]
                    end_result_mask = tf.expand_dims(1 - p_mask, axis=1)                                               # [b,l] --> [b,1,l]
                    end_result_mask = tf.tile(end_result_mask, multiples=[1,FLAGS.start_n_top,1])                    # [b,1,l] --> [b,k,l]
                    
                    end_result = tf.compat.v1.layers.dense(end_result, units=self.model_config.d_model, activation=tf.tanh,
                        use_bias=True, kernel_initializer=initializer, bias_initializer=tf.compat.v1.zeros_initializer,
                        kernel_regularizer=None, bias_regularizer=None, trainable=True, name="end_modeling")    # [b,l,k,2h] --> [b,l,k,h]
                    
                    end_result = tf.keras.layers.LayerNormalization(epsilon=1e-12, center=True, scale=True, 
                                                                  trainable=True, name="end_norm")(end_result)   # [b,l,k,h] --> [b,l,k,h]
                    
                    end_result = tf.compat.v1.layers.dense(end_result, units=1, activation=None,
                        use_bias=True, kernel_initializer=initializer, bias_initializer=tf.compat.v1.zeros_initializer,
                        kernel_regularizer=None, bias_regularizer=None, trainable=True, name="end_project")      # [b,l,k,h] --> [b,l,k,1]
                    
                    end_result = tf.transpose(a=tf.squeeze(end_result, axis=-1), perm=[0,2,1])                       # [b,l,k,1] --> [b,k,l]
                    end_result = self._generate_masked_data(end_result, end_result_mask)                    # [b,k,l], [b,k,l] --> [b,k,l]
                    end_prob = tf.nn.softmax(end_result, axis=-1)                                                                # [b,k,l]
                    
                    end_top_prob, end_top_index = tf.nn.top_k(end_prob, k=FLAGS.end_n_top)                  # [b,k,l] --> [b,k,k], [b,k,k]
                    predicts["end_prob"] = end_top_prob
                    predicts["end_index"] = end_top_index                   
            
            with tf.compat.v1.variable_scope("loss", reuse=tf.compat.v1.AUTO_REUSE):
                loss = tf.constant(0.0, dtype=tf.float32)
                if is_training:
                    start_label = start_positions                                                                                    # [b]
                    start_label_mask = tf.reduce_max(input_tensor=1 - p_mask, axis=-1)                                                  # [b,l] --> [b]
                    start_loss = self._compute_loss(start_label, start_label_mask, start_result, start_result_mask)                  # [b]
                    end_label = end_positions                                                                                        # [b]
                    end_label_mask = tf.reduce_max(input_tensor=1 - p_mask, axis=-1)                                                    # [b,l] --> [b]
                    end_loss = self._compute_loss(end_label, end_label_mask, end_result, end_result_mask)                            # [b]
                    loss += tf.reduce_mean(input_tensor=start_loss + end_loss)
            
        return loss, predicts
    
    def get_model_fn(self):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features,
                     labels,
                     mode,
                     params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            tf.compat.v1.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            
            unique_id = features["unique_id"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            p_mask = features["p_mask"]
            segment_ids = features["segment_ids"]
            cls_index = features["cls_index"]
            
            if is_training:
                start_position = features["start_position"]
                end_position = features["end_position"]
                    
            else:
                start_position = None
                end_position = None
            
            loss, predicts = self._create_model(is_training, input_ids, input_mask, p_mask, segment_ids, cls_index,
                start_position, end_position)
                
            scaffold_fn = model_utils.init_from_checkpoint(FLAGS)
            
            output_spec = None
            if is_training:
                train_op, _, _ = model_utils.get_train_op(FLAGS, loss)
                output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                predictions={
                    "unique_id": unique_id,
                    "start_prob": predicts["start_prob"],
                    "start_index": predicts["start_index"],
                    "end_prob": predicts["end_prob"],
                    "end_index": predicts["end_index"]
                    }
                    
                output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)
            
            return output_spec
        
        return model_fn

class XLNetPredictProcessor(object):
    """Default predict processor for XLNet"""
    def __init__(self,
                 output_dir,
                 n_best_size,
                 start_n_top,
                 end_n_top,
                 max_answer_length,
                 tokenizer,
                 predict_tag=None,
                 for_train=False):
        """Construct XLNet predict processor"""
        self.n_best_size = n_best_size
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer
        self.for_train = for_train
        
        predict_tag = predict_tag if predict_tag else str(time.time())
        if self.for_train:
            pass
        self.output_summary = os.path.join(output_dir, "predict.{0}.summary.json".format(predict_tag))
        self.output_detail = os.path.join(output_dir, "predict.{0}.detail.json".format(predict_tag))
    
    def _write_to_json(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:  
            json.dump(data_list, file, indent=4)
    
    def _write_to_text(self,
                       data_list,
                       data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:
            for data in data_list:
                file.write("{0}\n".format(data))
    
    def process(self,
                examples,
                features,
                results):
        qas_id_to_features = {}
        unique_id_to_feature = {}
        for feature in features:
            if feature.qas_id not in qas_id_to_features:
                qas_id_to_features[feature.qas_id] = []
            
            qas_id_to_features[feature.qas_id].append(feature)
            unique_id_to_feature[feature.unique_id] = feature
        
        unique_id_to_result = {}
        for result in results:
            unique_id_to_result[result.unique_id] = result
        
        predict_summary_list = []
        predict_detail_list = []
        num_example = len(examples)
        for (example_idx, example) in enumerate(examples):
            if example_idx % 1000 == 0:
                tf.compat.v1.logging.info('Updating {0}/{1} example with predict'.format(example_idx, num_example))
            
            if example.qas_id not in qas_id_to_features:
                tf.compat.v1.logging.warning('No feature found for example: {0}'.format(example.qas_id))
                continue
            
            example_all_predicts = []
            example_features = qas_id_to_features[example.qas_id]
            for example_feature in example_features:
                if example_feature.unique_id not in unique_id_to_result:
                    tf.compat.v1.logging.warning('No result found for feature: {0}'.format(example_feature.unique_id))
                    continue
                
                example_result = unique_id_to_result[example_feature.unique_id]
                
                for i in range(self.start_n_top):
                    start_prob = example_result.start_prob[i]
                    start_index = example_result.start_index[i]
                    
                    for j in range(self.end_n_top):
                        end_prob = example_result.end_prob[i][j]
                        end_index = example_result.end_index[i][j]
                        
                        answer_length = end_index - start_index + 1
                        if end_index < start_index or answer_length > self.max_answer_length:
                            continue
                        
                        if start_index > example_feature.para_length or end_index > example_feature.para_length:
                            continue
                        
                        if start_index not in example_feature.token2doc_index:
                            continue
                        
                        example_all_predicts.append({
                            "unique_id": example_result.unique_id,
                            "start_prob": float(start_prob),
                            "start_index": int(start_index),
                            "end_prob": float(end_prob),
                            "end_index": int(end_index),
                            "predict_score": float(np.log(start_prob) + np.log(end_prob))
                        })
            
            example_all_predicts = sorted(example_all_predicts, key=lambda x: x["predict_score"], reverse=True)
            
            is_visited = set()
            example_top_predicts = []
            for example_predict in example_all_predicts:
                if len(example_top_predicts) >= self.n_best_size:
                    break
                
                example_feature = unique_id_to_feature[example_predict["unique_id"]]
                predict_start = example_feature.token2char_raw_start_index[example_predict["start_index"]]
                predict_end = example_feature.token2char_raw_end_index[example_predict["end_index"]]
                predict_text = example.paragraph_text[predict_start:predict_end + 1].strip()
                
                if predict_text in is_visited:
                    continue
                
                is_visited.add(predict_text)
                
                example_top_predicts.append({
                    "predict_text": predict_text,
                    "predict_score": example_predict["predict_score"]
                })
            
            if len(example_top_predicts) == 0:
                example_top_predicts.append({
                    "predict_text": "",
                    "predict_score": 0.0
                })           
            
            example_best_predict = example_top_predicts[0]
            
            example_question_text = example.question_text.split('<s>')[-1].strip()
            
            predict_summary_list.append({
                "qas_id": example.qas_id,
                "question_text": example_question_text,
                "label_text": example.orig_answer_text,
                "predict_text": example_best_predict["predict_text"],
                "predict_score": example_best_predict["predict_score"]
            })
                                          
            predict_detail_list.append({
                "qas_id": example.qas_id,
                "question_text": example_question_text,
                "label_text": example.orig_answer_text,
                "best_predict": example_best_predict,
                "top_predicts": example_top_predicts
            })
        self._write_to_json(predict_summary_list, self.output_summary)
        self._write_to_json(predict_detail_list, self.output_detail)
        
def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    
    np.random.seed(FLAGS.random_seed)
    
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    
    task_name = FLAGS.task_name.lower()
    data_pipeline = CoqaPipeline(
        data_dir=FLAGS.data_dir,
        task_name=task_name,
        num_turn=FLAGS.num_turn)
    
    model_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    
    model_builder = XLNetModelBuilder(
        model_config=model_config,
        use_tpu=FLAGS.use_tpu)
    
    model_fn = model_builder.get_model_fn()
    
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    tpu_config = model_utils.configure_tpu(FLAGS)
    
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=tpu_config,
        export_to_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    tokenizer = XLNetTokenizer(
        sp_model_file=FLAGS.spiece_model_file,
        lower_case=FLAGS.lower_case)
    
    example_processor = XLNetExampleProcessor(
        max_seq_length=FLAGS.max_seq_length,
        max_query_length=FLAGS.max_query_length,
        doc_stride=FLAGS.doc_stride,
        tokenizer=tokenizer)
    
    def write_meta_file(example_num, feature_num, file_name):
        data = {"num of examples": example_num, "num of features": feature_num}
        with open(file_name, "w") as writer:
            json.dump(data, writer, indent=4)
            
    def get_num_of_features(file_name):
        with open(file_name, "r") as reader:
            data = json.load(reader)
        return data["num of features"]
    
    if FLAGS.do_train:
        train_examples = []
        train_examples_from_g_coqa = []
        train_examples_from_g_quac = []
        tf.compat.v1.logging.info("***** Run training *****")
        
        if FLAGS.train_original_span:
            train_examples.extend(data_pipeline.get_train_examples())
        if FLAGS.train_generated_coqa:
            train_examples_from_g_coqa.extend(data_pipeline.get_g_coqa_examples())
        if FLAGS.train_generated_quac:
            train_examples_from_g_quac.extend(data_pipeline.get_g_quac_examples())
            
        train_record_file = None
        train_meta = None
        if FLAGS.train_original_span:
            print("Generate original span feature file...")
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_original_span.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_original_span.meta".format(task_name))
            if not os.path.exists(train_record_file):
                original_train_features = example_processor.convert_examples_to_features(train_examples)
                np.random.shuffle(original_train_features)
                example_processor.save_features_as_tfrecord(original_train_features, train_record_file)
                write_meta_file(len(train_examples), len(original_train_features), train_meta)
        
        elif FLAGS.train_original and FLAGS.train_generated_coqa and FLAGS.train_generated_quac:
            original_record_file = os.path.join(FLAGS.output_dir, "train-{0}_original.tfrecord".format(task_name))
            original_meta = os.path.join(FLAGS.output_dir, "train-{0}_original.meta".format(task_name))
            
            generated_coqa_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.tfrecord".format(task_name))
            generated_coqa_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.meta".format(task_name))
            
            generated_quac_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.tfrecord".format(task_name))
            generated_quac_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.meta".format(task_name))
            
            merged_coqa_record_file = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa.tfrecord".format(task_name))
            merged_coqa_meta = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa.meta".format(task_name))
            
            merged_quac_record_file = os.path.join(FLAGS.output_dir, "train-{0}_merged_quac.tfrecord".format(task_name))
            merged_quac_meta = os.path.join(FLAGS.output_dir, "train-{0}_merged_quac.meta".format(task_name))
            
            generated_coqa_quac_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa_quac.tfrecord".format(task_name))
            generated_coqa_quac_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa_quac.meta".format(task_name))
            
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa_quac.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa_quac.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                original_train_features = None
                generated_coqa_train_features = None
                generated_quac_train_features = None
                if not os.path.exists(original_record_file):
                    original_train_features = example_processor.convert_examples_to_features(train_examples)
                    np.random.shuffle(original_train_features)
                    example_processor.save_features_as_tfrecord(original_train_features, original_record_file)
                    write_meta_file(len(train_examples), len(original_train_features), original_meta)
                
                if not os.path.exists(generated_coqa_record_file):
                    generated_coqa_train_features = example_processor.convert_examples_to_features(train_examples_from_g_coqa)
                    np.random.shuffle(generated_coqa_train_features)
                    example_processor.save_features_as_tfrecord(generated_coqa_train_features, generated_coqa_record_file)
                    write_meta_file(len(train_examples_from_g_coqa), len(generated_coqa_train_features), generated_coqa_meta)
                
                if not os.path.exists(generated_quac_record_file):
                    generated_quac_train_features = example_processor.convert_examples_to_features(train_examples_from_g_quac)
                    np.random.shuffle(generated_quac_train_features)
                    example_processor.save_features_as_tfrecord(generated_quac_train_features, generated_quac_record_file)
                    write_meta_file(len(train_examples_from_g_quac), len(generated_quac_train_features), generated_quac_meta)
                
                if not os.path.exists(merged_coqa_record_file):
                    merged_coqa_features = original_train_features + generated_coqa_train_features
                    np.random.shuffle(merged_coqa_features)
                    example_processor.save_features_as_tfrecord(merged_coqa_features, merged_coqa_record_file)
                    write_meta_file(len(train_examples)+len(train_examples_from_g_coqa), len(merged_coqa_features), merged_coqa_meta)
                    
                if not os.path.exists(merged_quac_record_file):
                    merged_quac_features = original_train_features + generated_quac_train_features
                    np.random.shuffle(merged_quac_features)
                    example_processor.save_features_as_tfrecord(merged_quac_features, merged_quac_record_file)
                    write_meta_file(len(train_examples)+len(train_examples_from_g_quac), len(merged_quac_features), merged_quac_meta)
                    
                if not os.path.exists(generated_coqa_quac_record_file):
                    generated_coqa_quac_features = generated_coqa_train_features + generated_quac_train_features
                    np.random.shuffle(generated_coqa_quac_features)
                    example_processor.save_features_as_tfrecord(generated_coqa_quac_features, generated_coqa_quac_record_file)
                    write_meta_file(len(train_examples_from_g_coqa)+len(train_examples_from_g_quac), len(generated_coqa_quac_features), generated_coqa_quac_meta)
                
                train_features = original_train_features + generated_coqa_train_features + generated_quac_train_features
                np.random.shuffle(train_features)
                example_processor.save_features_as_tfrecord(train_features, train_record_file)
                write_meta_file(len(train_examples)+len(train_examples_from_g_coqa)+len(train_examples_from_g_quac), len(train_features), train_meta)
                
        elif FLAGS.train_original and FLAGS.train_generated_coqa:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_merged_coqa.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0
                
        elif FLAGS.train_original and FLAGS.train_generated_quac:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_merged_quac.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_merged_quac.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0

        elif FLAGS.train_generated_coqa and FLAGS.train_generated_quac:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa_quac.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa_quac.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0

        elif FLAGS.train_original:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_original.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_original.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0
                
        elif FLAGS.train_generated_coqa:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0
                
        elif FLAGS.train_generated_quac:
            train_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.tfrecord".format(task_name))
            train_meta = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.meta".format(task_name))
            if not os.path.exists(train_record_file) or FLAGS.overwrite_data:
                tf.compat.v1.logging.info("NO GENERATED FILES...!!!!")
                assert 1==0
                
        else:
            assert False, "Select a train dataset"
            
        num_of_features = get_num_of_features(train_meta)
        FLAGS.train_steps = math.ceil((num_of_features * FLAGS.epochs)/(FLAGS.train_batch_size * FLAGS.num_core_per_host))
        if FLAGS.fine_tune:
            assert FLAGS.pretrained_steps is not None, "The numbert pretrained steps is required"
            FLAGS.train_steps += FLAGS.pretrained_steps
                
        tf.compat.v1.logging.info("  Num examples from original CoQA dataset = %d", len(train_examples))
        tf.compat.v1.logging.info("  Num examples from generated CoQA dataset = %d", len(train_examples_from_g_coqa))
        tf.compat.v1.logging.info("  Num examples from generated QuAC dataset = %d", len(train_examples_from_g_quac))
        tf.compat.v1.logging.info("  Num total examples = %d", len(train_examples)+len(train_examples_from_g_coqa)+len(train_examples_from_g_quac))
        tf.compat.v1.logging.info("  Num total features = %d", num_of_features)
        tf.compat.v1.logging.info("  Batch size per GPU = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Total batch size = %d", FLAGS.train_batch_size * FLAGS.num_core_per_host)
        tf.compat.v1.logging.info("  Num epochs = %d", FLAGS.epochs)
        tf.compat.v1.logging.info("  Num steps = %d", FLAGS.train_steps)
        
        train_input_fn = XLNetInputBuilder.get_input_fn(train_record_file, FLAGS.max_seq_length, True, True, FLAGS.shuffle_buffer)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)     

    if FLAGS.do_predict:
        predict_examples = None
        if FLAGS.do_predict_span:
            tf.compat.v1.logging.info("***** Run the generated CoQA (only span) prediction *****")
            predict_examples = data_pipeline.get_dev_span_examples()
        elif FLAGS.do_predict_coqa:
            tf.compat.v1.logging.info("***** Run the generated CoQA prediction *****")
            predict_examples = data_pipeline.get_g_coqa_dev_examples()
        elif FLAGS.do_predict_quac:
            tf.compat.v1.logging.info("***** Run the generated QuAC prediction *****")
            predict_examples = data_pipeline.get_g_quac_dev_examples()
        elif FLAGS.do_predict_coqa_train:
            tf.compat.v1.logging.info("***** Run the generated CoQA training set prediction *****")
            predict_examples = data_pipeline.get_g_coqa_examples()
        elif FLAGS.do_predict_quac_train:
            tf.compat.v1.logging.info("***** Run the generated QuAC training set prediction *****")
            predict_examples = data_pipeline.get_g_quac_examples()
        else:
            ("***** Run CoQA prediction *****")
            predict_examples = data_pipeline.get_dev_examples()
        
        tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        
        predict_record_file = None
        predict_pickle_file = None
        
        if FLAGS.do_predict_span:
            predict_record_file = os.path.join(FLAGS.output_dir, "dev-{0}_original_span.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "dev-{0}_original_span.pkl".format(task_name))
        elif FLAGS.do_predict_coqa:
            predict_record_file = os.path.join(FLAGS.output_dir, "dev-{0}_generated_coqa.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "dev-{0}_generated_coqa.pkl".format(task_name))
        elif FLAGS.do_predict_quac:
            predict_record_file = os.path.join(FLAGS.output_dir, "dev-{0}_generated_quac.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "dev-{0}_generated_quac.pkl".format(task_name))
        elif FLAGS.do_predict_coqa_train:
            predict_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_coqa.pkl".format(task_name))
        elif FLAGS.do_predict_quac_train:
            predict_record_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "train-{0}_generated_quac.pkl".format(task_name))
        else:
            predict_record_file = os.path.join(FLAGS.output_dir, "dev-{0}.tfrecord".format(task_name))
            predict_pickle_file = os.path.join(FLAGS.output_dir, "dev-{0}.pkl".format(task_name))
        if not os.path.exists(predict_record_file) or not os.path.exists(predict_pickle_file) or FLAGS.overwrite_data:
            predict_features = example_processor.convert_examples_to_features(predict_examples)
            example_processor.save_features_as_tfrecord(predict_features, predict_record_file)
            example_processor.save_features_as_pickle(predict_features, predict_pickle_file)
        else:
            predict_features = example_processor.load_features_from_pickle(predict_pickle_file)

        predict_input_fn = XLNetInputBuilder.get_input_fn(predict_record_file, FLAGS.max_seq_length, False, False)
        results = estimator.predict(input_fn=predict_input_fn)
        
        predict_results = [OutputResult(
            unique_id=result["unique_id"],
            start_prob=result["start_prob"].tolist(),
            start_index=result["start_index"].tolist(),
            end_prob=result["end_prob"].tolist(),
            end_index=result["end_index"].tolist()
        ) for result in results]

        predict_processor = XLNetPredictProcessor(
            output_dir=FLAGS.output_dir,
            n_best_size=FLAGS.n_best_size,
            start_n_top=FLAGS.start_n_top,
            end_n_top=FLAGS.end_n_top,
            max_answer_length=FLAGS.max_answer_length,
            tokenizer=tokenizer,
            predict_tag=FLAGS.predict_tag)
        
        predict_processor.process(predict_examples, predict_features, predict_results)
        
    if FLAGS.do_export:
        tf.compat.v1.logging.info("***** Running exporting *****")
        if not os.path.exists(FLAGS.export_dir):
            os.mkdir(FLAGS.export_dir)
        
        serving_input_fn = XLNetInputBuilder.get_serving_input_fn(FLAGS.max_seq_length)
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn, as_text=False)

if __name__ == "__main__":
    FLAGS = parse_args()
    print("Do Train:",FLAGS.do_train)
    print("Do Predict:",FLAGS.do_predict)
    print("Do Export:",FLAGS.do_export)
    print("Do Original:",FLAGS.train_original)
    print("Do Generated_CoQA:",FLAGS.train_generated_coqa)
    print("Do Generated_QuAC:",FLAGS.train_generated_quac)
    print("Do lower case:", FLAGS.lower_case)
    tf.compat.v1.app.run()
