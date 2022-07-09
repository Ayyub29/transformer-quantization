# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
import resource
import torch 

from enum import Enum

from transformers import BertForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, BertConfig, BertTokenizer, PreTrainedTokenizerFast, BertForTokenClassification
from models.pretrained_bert import BertForWordClassification, BertForMultiLabelClassification
from utils.indonlu_task import INDONLU_Task, TASK_INDEX2LABEL, load_task_data_indonlu

from utils.utils import count_embedding_params, count_params, DotDict
logger = logging.getLogger('INDO_NLU')
logger.setLevel(logging.ERROR)


class HF_Models(Enum):
    # vanilla BERT
    bert_base_uncased = 'bert-base-uncased'
    bert_large_uncased = 'bert-large-uncased'
    bert_base_cased = 'bert-base-cased'

    # RoBERTa
    roberta_base = 'roberta-base'

    # Distilled vanilla models
    distilbert_base_uncased = 'distilbert-base-uncased'
    distilroberta_base = 'distilroberta-base'

    # Models optimized for runtime on mobile devices
    mobilebert_uncased = 'google/mobilebert-uncased'
    squeezebert_uncased = 'squeezebert/squeezebert-uncased'

    # ALBERT: very small set of models (optimized for memory)
    albert_base_v2 = 'albert-base-v2'
    albert_large_v2 = 'albert-large-v2'

    # IndoBERT - Indonesian BERT
    indobert_base_v1 = 'indobenchmark/indobert-base-p1'
    indobert_base_v2 = 'indobenchmark/indobert-base-p2'

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


MODEL_TO_BACKBONE_ATTR = {  # model.<backbone attr>.<layers etc.>
    HF_Models.bert_base_uncased: 'bert',
    HF_Models.bert_large_uncased: 'bert',
    HF_Models.bert_base_cased: 'bert',
    HF_Models.indobert_base_v1: 'bert',
    HF_Models.indobert_base_v2: 'bert',
    HF_Models.distilroberta_base: 'roberta',
    HF_Models.roberta_base: 'roberta',
    HF_Models.mobilebert_uncased: 'mobilebert'
}

def load_model_and_tokenizer(model_name, model_path, use_fast_tokenizer, cache_dir, attn_dropout,
                             hidden_dropout, num_labels, task: INDONLU_Task, task_data, **kw):
    """
    Loading the model and tokenizer
    """
    del kw  # unused

    out = DotDict()

    # Config
    if model_path is not None:
        model_name_or_path = model_path
    else:
        model_name_or_path = HF_Models[model_name].value  # use HF identifier

    # Creating model config
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )

    # set dropout rates
    if attn_dropout is not None:
        print(f'Setting attn dropout to {attn_dropout}')
        if hasattr(config, 'attention_probs_dropout_prob'):
            setattr(config, 'attention_probs_dropout_prob', attn_dropout)

    if hidden_dropout is not None:
        print(f'Setting hidden dropout to {hidden_dropout}')
        if hasattr(config, 'hidden_dropout_prob'):
            setattr(config, 'hidden_dropout_prob', attn_dropout)

    logger.info('HuggingFace model config:')
    print(config)
    out.config = config
    out.model_name_or_path = model_name_or_path

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        cache_dir=cache_dir,
    )
    logger.info('Tokenizer:')
    logger.info(tokenizer)
    out.tokenizer = tokenizer
    config.problem_type = None
    # Model
    if task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete:
        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
        )
    elif task == INDONLU_Task.casa or task == INDONLU_Task.hoasa:
        config.num_labels_list = task_data.num_labels_list
        model = BertForMultiLabelClassification.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        model = BertForWordClassification.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=cache_dir,
        )

    print('Model:')
    print(model)
    print()
    out.model = model

    # Parameter counts
    total_params = count_params(model)
    embedding_params = count_embedding_params(model)
    non_embedding_params = total_params - embedding_params
    print(f'Parameters (embedding): {embedding_params}')
    print(f'Parameters (non-embedding): {non_embedding_params}')
    print(f'Parameters (total): {total_params}')
    out.total_params = total_params
    out.embedding_params = embedding_params
    out.non_embedding_params = non_embedding_params

    # Additional attributes
    out.model_enum = HF_Models[model_name]
    out.backbone_attr = MODEL_TO_BACKBONE_ATTR.get(out.model_enum, None)
    return out

def checkpoint(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    print('''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 ))
    return usage 

def check_memory_and_inference_time(config, task, dataset, has_Trained):
    output_dir = config.base.output_dir if has_Trained else config.model.model_path
    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'out')
    else:
        output_dir = HF_Models[config.model.model_name].value
    tokenizer = AutoTokenizer.from_pretrained(output_dir,use_fast=True)
    dataset = load_task_data_indonlu(task,data_dir=config.indonlu.data_dir)
    print(dataset.dataset)
    
    start_memory = checkpoint("Starting Point")
    model = BertForSequenceClassification.from_pretrained(output_dir,local_files_only=True)
    model.eval()
    load_memory = checkpoint("Loading the Model")
    # Forward model
    outputs = model(subwords, labels=label)
    loss, logits = outputs[:2]
    forward_memory = checkpoint("Forwarding the Model")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_memory = checkpoint("Backwarding the Model")
    
