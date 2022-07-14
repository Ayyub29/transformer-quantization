# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from re import I
import resource
import torch 
import time
import torch.nn.functional as F
import objsize

from enum import Enum

from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig
from models.pretrained_bert import BertForWordClassification, BertForMultiLabelClassification
from models.quantized_bert import _quantize_model
from utils.indonlu_task import TASK_LABELS, TASK_MULTILABELS, INDONLU_Task, TASK_INDEX2LABEL, load_task_data_indonlu

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
                             hidden_dropout, num_labels, task: INDONLU_Task, task_data, output_dir, **kw):
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
    # print(config)
    out.config = config
    out.model_name_or_path = model_name_or_path
    # create dirpath if not exist
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(output_dir + '/config.json')
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

def Average(lst):
    return sum(lst) / len(lst)

def checkpoint(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return usage 

def word_subword_tokenize(sentence, tokenizer):
    # Add CLS token
    subwords = [tokenizer.cls_token_id]
    subword_to_word_indices = [-1] # For CLS

    # Add subwords
    for word_idx, word in enumerate(sentence):
        subword_list = tokenizer.encode(word, add_special_tokens=False)
        subword_to_word_indices += [word_idx for i in range(len(subword_list))]
        subwords += subword_list

    # Add last SEP token
    subwords += [tokenizer.sep_token_id]
    subword_to_word_indices += [-1]

    return subwords, subword_to_word_indices

def check_memory_usage(config, task, is_quantized):
    print("Checking Model..")
    output_dir = config.base.output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(output_dir,use_fast=True)
    dataset = load_task_data_indonlu(task,data_dir=config.indonlu.data_dir)
    is_text_class_task = task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete
    is_multilabel_class_task = task == INDONLU_Task.casa or task == INDONLU_Task.hoasa
    load_memory_arr = []
    forward_memory_arr = []
    backward_memory_arr = []
    forward_time_arr = []
    backward_time_arr = []

    for i in range(10):
        start_memory = checkpoint("Starting Point")

        if is_text_class_task: 
            model = BertForSequenceClassification.from_pretrained(output_dir,local_files_only=True)
        elif is_multilabel_class_task:
            model = BertForMultiLabelClassification.from_pretrained(output_dir,local_files_only=True)
        else:
            model = BertForWordClassification.from_pretrained(output_dir,local_files_only=True)

        if is_quantized:
            model = _quantize_model(config,model,task)
            
        model.eval()
        load_memory = checkpoint("Loading the Model")
        load_memory_arr.append((load_memory[2] - start_memory[2])/1024.0)

        sentence = dataset.datasets['train'][i][dataset.sentence1_key]
        if is_multilabel_class_task:
            label_id = []
            for feature in TASK_MULTILABELS[task]:
                label_id.append(dataset.datasets['train'][i][feature])
            label = [label_id]
            subwords = tokenizer.encode(sentence)
            subwords = torch.LongTensor(subwords).view(1, -1)
            label = torch.LongTensor(label)
        elif is_text_class_task:
            label = [dataset.datasets['train'][i][TASK_LABELS[task]]]
            subwords = tokenizer.encode(sentence)
            subwords = torch.LongTensor(subwords).view(1, -1)
            label = torch.LongTensor(label)
        else:
            # text = tokenizer(sentence, truncation=True,max_length= config.data.max_seq_length, is_split_into_words=True)
            subwords, subword_to_word_indices = word_subword_tokenize(sentence, tokenizer)

            subwords = torch.LongTensor(subwords).view(1, -1)
            subword_to_word_indices = torch.LongTensor(subword_to_word_indices).view(1, -1)
            label = [dataset.datasets['train'][i][TASK_LABELS[task]]]
            label = torch.LongTensor(label)

        # print(label.size(), subwords.size())
        dataset_memory = checkpoint("Loading the Dataset")

        #Forward
        start_time = time.time()
        if is_multilabel_class_task or is_text_class_task:
            outputs = model(subwords, labels=label)
        else:
            outputs = model(subwords, subword_to_word_indices, labels=label)
        loss, logits = outputs[:2]
        forward_memory = checkpoint("Forwarding the Model")
        forward_memory_arr.append((forward_memory[2] - dataset_memory[2])/1024.0)
        forward_time = time.time() - start_time
        forward_time_arr.append(forward_time)
        time_after_forward = time.time()

        #Backward
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_memory = checkpoint("Backwarding the Model")
        backward_memory_arr.append((backward_memory[2]- dataset_memory[2])/1024.0)
        backward_time = time.time() - time_after_forward
        backward_time_arr.append(backward_time)
        print(f'data {i+1}')
        print(f'Memory Used: Load {load_memory[2]/1024.0 - (start_memory[2]/1024.0)} mb | Forward {forward_memory[2]/1024.0 - (dataset_memory[2]/1024.0)} mb | Backward {backward_memory[2]/1024.0 - (dataset_memory[2]/1024.0)} mb')
        print(f'Time: Forward {forward_time} s | Backward {backward_time} s')
        print()
    print("Average: ")
    print(f'Memory Used: Load {Average(load_memory_arr)} mb | Forward {Average(forward_memory_arr)} mb | Backward {Average(backward_memory_arr)} mb  ')
    print(f'Time: Forward {Average(forward_time_arr)} s | Backward {Average(backward_time_arr)} s')
    print()

def check_inference_time(config, task, is_quantized):
    print("Checking Model..")
    output_dir = config.base.output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(output_dir,use_fast=True)
    dataset = load_task_data_indonlu(task,data_dir=config.indonlu.data_dir)
    is_text_class_task = task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete
    is_multilabel_class_task = task == INDONLU_Task.casa or task == INDONLU_Task.hoasa
    forward_time_arr = []

    if is_text_class_task: 
        model = BertForSequenceClassification.from_pretrained(output_dir,local_files_only=True)
    elif is_multilabel_class_task:
        model = BertForMultiLabelClassification.from_pretrained(output_dir,local_files_only=True)
    else:
        model = BertForWordClassification.from_pretrained(output_dir,local_files_only=True)

    if is_quantized:
        model = _quantize_model(config,model,task)
            
    model.eval()

    for i in range(10):
        sentence = dataset.datasets['validation'][i][dataset.sentence1_key]
        if is_multilabel_class_task:
            subwords = tokenizer.encode(sentence)
            subwords = torch.LongTensor(subwords).view(1, -1)
        elif is_text_class_task:
            subwords = tokenizer.encode(sentence)
            subwords = torch.LongTensor(subwords).view(1, -1)
        else:
            subwords, subword_to_word_indices = word_subword_tokenize(sentence, tokenizer)

            subwords = torch.LongTensor(subwords).view(1, -1)
            subword_to_word_indices = torch.LongTensor(subword_to_word_indices).view(1, -1)

        #Forward
        start_time = time.time()
        if is_multilabel_class_task or is_text_class_task:
            outputs = model(subwords)
        else:
            outputs = model(subwords, subword_to_word_indices)
        logits = outputs[0]
        
        forward_time = time.time() - start_time
        forward_time_arr.append(forward_time)

        if (is_text_class_task):
            index = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
            print(f'Text: {sentence} | Label : {TASK_INDEX2LABEL[task][index]} ({F.softmax(logits, dim=-1).squeeze()[index] * 100:.3f}%)')
        elif is_multilabel_class_task:
            index = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]
            print(f'Text: {sentence}')
            for i, label in enumerate(index):
                print(f'Label `{TASK_MULTILABELS[task][i]}` : {TASK_INDEX2LABEL[task][label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')
        else:
            preds = torch.topk(logits, k=1, dim=-1)[1].squeeze().numpy()
            labels = [TASK_INDEX2LABEL[task][preds[i]] for i in range(len(preds))]
            for idx,word in enumerate(sentence):
                print(f'{word} | {labels[idx]}')
        print(f'Time: {forward_time} s ')
        print()
    print("Average: ")
    print(f'Time: Forward {Average(forward_time_arr)} s ')
    print()

def print_size_of_model(model):
    print(model)
    model_size = 0
    print(list(model.named_parameters())) 
    for param in model.parameters():
        # print(param.data, param.data.dtype, param.numel())
        if param.data.dtype == torch.float32:
            size = 4
        if param.data.dtype == torch.float16:
            size = 2
        if param.data.dtype == torch.int8:
            size = 1
        model_size += param.numel() * size

    print(model_size)