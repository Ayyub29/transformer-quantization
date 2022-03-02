# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Flag, auto
from functools import reduce
from operator import or_

import numpy as np
from datasets import load_dataset, load_metric
from transformers import EvalPrediction

from utils.utils import DotDict


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


# TASK_N = {
#     GLUE_Task.mnli: 392702,
#     GLUE_Task.qqp: 363846,
#     GLUE_Task.qnli: 104743,
#     GLUE_Task.sst2: 67349,
#     GLUE_Task.cola: 8551,
#     GLUE_Task.stsb: 5749,
#     GLUE_Task.mrpc: 3665,
#     GLUE_Task.rte: 2490,
#     GLUE_Task.wnli: 635,
# }

def load_task_data_indonlu(task: INDONLU_Task, data_dir: str):
    out = DotDict()

    # download and load data
    logger.info(f'Getting {task.name} dataset ...\n')
    out.datasets = load_dataset('indonlu', task.name, cache_dir=data_dir)

    # determine number of labels
    logger.info('Determine labels ...\n')
    if task == INDONLU_Task.wrete:  # need to be adjusted later
        out.num_labels = 1
        logger.info(f'{task.name}: 1 label -- <Regression>')
    else:
        label_list = out.datasets["train"].features["label"].names
        out.num_labels = n_labels = len(label_list)
        logger.info(f'{task.name}: {n_labels} labels -- {label_list}')

    # store sentence keys
    out.sentence1_key, out.sentence2_key = TASK_TO_SENTENCE_KEYS[task]
    return out


def make_compute_metric_fn_indonlu(task: INDONLU_Task):
    result = {}

    def fn(p: EvalPrediction):
        logger.info(p)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task == INDONLU_Task.wrete else np.argmax(preds, axis=1) ##NEED TO BE ADJUSTED LATER
        metric_f1 = load_metric('f1')
        metric_acc = load_metric('accuracy')
        metric_prec = load_metric('precision')
        metric_rec = load_metric('recall')
        result["f1"] = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")
        result["acc"] = metric_acc.compute(predictions=preds, references=p.label_ids)
        result["prec"] = metric_prec.compute(predictions=preds, references=p.label_ids, average="macro")
        result["rec"] = metric_rec.compute(predictions=preds, references=p.label_ids, average="macro")
        return result

    return fn
