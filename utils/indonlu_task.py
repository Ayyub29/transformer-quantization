# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from cProfile import label
import logging
from enum import Flag, auto
from functools import reduce
from operator import or_

import itertools
from unittest import result
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import EvalPrediction

from utils.utils import DotDict, Stopwatch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from utils.conlleval import conll_evaluation

# setup logger
logger = logging.getLogger('INDO_NLU')
logger.setLevel(logging.INFO)

class INDONLU_Task(Flag):
    emot = auto()
    smsa = auto()
    casa = auto()
    hoasa = auto()
    wrete = auto()
    posp = auto()
    bapos = auto()
    terma = auto()
    keps = auto()
    nergrit = auto()
    nerp = auto()
    facqa = auto()
    all = emot | smsa | casa | hoasa | wrete | posp | bapos | terma | keps | nergrit | nerp | facqa

    def __contains__(self, item):
        return (self.value & item.value) == item.value

    @classmethod
    def from_str(cls, *names):
        """Construct flag from strings."""
        assert len(names)
        return reduce(or_, map(cls.__getattr__, names))

    @classmethod
    def list_names(cls):
        """List all flags, including `all`."""
        return [m.name for m in cls]

    def iter(self):
        """List all member flags which are set (excluding `all`)."""
        for x in self.__class__.__members__.values():
            if x in self and x != self.__class__.all:
                yield x

    def iter_names(self):
        """List all member flag names which are set (excluding `all`)."""
        for x in self.iter():
            yield x.name

## NEED TO BE ADJUSTED LATER
TASK_TO_SENTENCE_KEYS = {
    INDONLU_Task.emot: ('tweet', None),
    INDONLU_Task.smsa: ('text', None),
    INDONLU_Task.casa: ('sentence', None),
    INDONLU_Task.hoasa: ('sentence', None),
    INDONLU_Task.wrete: ('premise', 'hypothesis'),
    INDONLU_Task.posp: ('tokens', None),
    INDONLU_Task.bapos: ('tokens', None),
    INDONLU_Task.terma: ('tokens', None),
    INDONLU_Task.keps: ('tokens', None),
    INDONLU_Task.nergrit: ('tokens', None),
    INDONLU_Task.nerp: ('tokens', None),
    INDONLU_Task.facqa: ('question', 'passage')
}


TASK_TO_FINAL_METRIC_INDONLU = {
    INDONLU_Task.emot: 'f1',
    INDONLU_Task.smsa: 'f1',
    INDONLU_Task.casa: 'f1',
    INDONLU_Task.hoasa: 'f1',
    INDONLU_Task.wrete: 'f1',
    INDONLU_Task.posp: 'f1',
    INDONLU_Task.bapos: 'f1',
    INDONLU_Task.terma: 'f1',
    INDONLU_Task.keps: 'f1',
    INDONLU_Task.nergrit: 'f1',
    INDONLU_Task.nerp: 'f1',
    INDONLU_Task.facqa: 'f1'
}

TASK_LABELS = {
    INDONLU_Task.emot: 'label',
    INDONLU_Task.smsa: 'label',
    INDONLU_Task.casa: 'label_ids',
    INDONLU_Task.hoasa: 'label_ids',
    INDONLU_Task.wrete: 'label',
    INDONLU_Task.posp: 'pos_tags',
    INDONLU_Task.bapos: 'pos_tags',
    INDONLU_Task.terma: 'seq_label',
    INDONLU_Task.keps: 'seq_label',
    INDONLU_Task.nergrit: 'ner_tags',
    INDONLU_Task.nerp: 'ner_tags',
    INDONLU_Task.facqa: 'seq_label'
}

TASK_LABEL2INDEX = {
    INDONLU_Task.emot: {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4},
    INDONLU_Task.smsa: {'positive': 0, 'neutral': 1, 'negative': 2}
}

TASK_INDEX2LABEL = {
    INDONLU_Task.emot: {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'},
    INDONLU_Task.smsa: {0: 'positive', 1: 'neutral', 2: 'negative'},
    INDONLU_Task.casa: {0: 'negative', 1: 'neutral', 2: 'positive'},
    INDONLU_Task.hoasa: {0: 'neg', 1: 'neut', 2: 'pos', 3: 'neg_pos'}
}

TASK_MULTILABELS = {
    INDONLU_Task.casa: ['fuel', 'machine', 'others', 'part', 'price', 'service'],
    INDONLU_Task.hoasa: ['ac', 'air_panas', 'bau', 'general', 'kebersihan', 'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
}

TASK_MULTILABEL_NUM = {
    INDONLU_Task.casa: [3,3,3,3,3,3],
    INDONLU_Task.hoasa: [4,4,4,4,4,4,4,4]
}

def load_task_data_indonlu(task: INDONLU_Task, data_dir: str):
    """
    Loading dataset task on INDONLU, including determining labels length and column name on dataset
    """
    out = DotDict()

    # download and load data
    logger.info(f'Getting {task.name} dataset ...\n')
    out.datasets = load_dataset('indonlu', task.name, cache_dir=data_dir)

    # determine number of labels
    logger.info('Determine labels ...\n')
    if task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete:  # sequence classification
        label_list = out.datasets["train"].features[TASK_LABELS[task]].names
        out.num_labels = n_labels = len(label_list)
        logger.info(f'{task.name}: {n_labels} labels -- {label_list}')
    elif task == INDONLU_Task.casa or task == INDONLU_Task.hoasa: #aspect based sentiment analysis
        out.num_labels = max(TASK_MULTILABEL_NUM[task])
        # out.num_labels = len(TASK_MULTILABELS[task])
        out.num_labels_list  = TASK_MULTILABEL_NUM[task]
        label_list = []
        for feature in out.datasets['train'].column_names:
            if (feature != 'sentence'):
                label_list = label_list + [feature]
        print(f'{task.name}: { TASK_MULTILABELS[task]} labels -- {label_list}')
    else:
        label_list = out.datasets["train"].features[TASK_LABELS[task]].feature.names
        out.num_labels = n_labels = out.datasets["train"].features[TASK_LABELS[task]].feature.num_classes
        print(f'{task.name}: {n_labels} labels -- {label_list}')

    # store sentence keys
    out.sentence1_key, out.sentence2_key = TASK_TO_SENTENCE_KEYS[task]
    return out


def make_compute_metric_fn_text(task: INDONLU_Task):
    def fn(p: EvalPrediction):
        s = Stopwatch().start()
        result = {}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        list_metric = ['f1', 'accuracy', 'precision', 'recall']
        for metric in list_metric:
            metric_loader = load_metric(metric)
            if (metric == 'accuracy'):
                value = metric_loader.compute(predictions=preds, references=p.label_ids)[metric]
            else:
                value = metric_loader.compute(predictions=preds, references=p.label_ids, average="macro")[metric]
            result[metric] = value
        return result
    return fn

def make_compute_metric_fn_word(task: INDONLU_Task):
    dataset = load_dataset('indonlu', task.name)
    label_list = dataset["train"].features[TASK_LABELS[task]].feature.names
    # print(label_list)
    def fn(p: EvalPrediction):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        for (p,l) in zip(true_predictions, true_labels):
            print(p)
            print(l)
            print()
            
        metrics = {}
        acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(true_predictions, true_labels)
        metrics["ACC"] = acc
        metrics["F1"] = tm_f1
        metrics["REC"] = tm_rec
        metrics["PRE"] = tm_pre
        return metrics

    return fn

def make_compute_metric_fn_multilable(task: INDONLU_Task):
    def fn(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        hyp = []
        for pred in preds:
            hy = []
            for item in pred:
                hy.append(np.argmax(item))
            hyp.append(hy)

        list_hyp = []
        list_label = []
        batch_size = p.label_ids.shape[0]
        num_label = len(hyp)
        for i in range(batch_size):
            hyps = []
            labels = p.label_ids[i,:]
            for j in range(num_label):
                hyps.append(hyp[j][i])
            list_hyp.append([i for i in hyps])
            list_label.append([j for j in labels])

        list_hyp = list(itertools.chain.from_iterable(list_hyp))
        list_label = list(itertools.chain.from_iterable(list_label))

        f1_micro_average = f1_score(list_hyp, y_pred=list_label, average='macro')
        # roc_auc = roc_auc_score(list_hyp, list_label, average = 'micro')
        accuracy = accuracy_score(list_hyp, list_label)
        precision = precision_score(list_hyp, list_label, average='macro')
        recall = recall_score(list_hyp, list_label, average='macro')
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall}
        return metrics
    return fn