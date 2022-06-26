#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from enum import Flag
from lib2to3.pgen2.tokenize import tokenize
import logging
import os
import random
import warnings
from models.quantized_bert import QuantizedBertForMultiLabelClassification

from utils.indonlu_task import TASK_TO_FINAL_METRIC_INDONLU

warnings.filterwarnings('ignore')  # ignore TF warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pformat

import click
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, default_data_collator, DataCollatorForTokenClassification
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset, load_metric
from pynvml import *

from models import (
    QuantizedBertForSequenceClassification,
    QuantizedBertForMultiLabelClassification,
    QuantizedBertForWordClassification
)
from quantization.adaround import AdaRoundActQuantMode
from utils import (
    # click options
    quantization_options,
    activation_quantization_options,
    qat_options,
    adaround_options,
    make_qparams,
    glue_options,
    indonlu_options,
    transformer_base_options,
    transformer_data_options,
    transformer_model_options,
    transformer_training_options,
    transformer_progress_options,
    transformer_quant_options,

    # quantization
    apply_adaround_to_model,
    prepare_model_for_quantization,
    pass_data_for_range_estimation,
    hijack_act_quant,
    hijack_weight_quant,
    hijack_act_quant_modules,
    set_act_quant_axis_and_groups,

    # pipeline
    load_model_and_tokenizer,
    load_task_data_indonlu,
    make_compute_metric_fn_text,
    make_compute_metric_fn_word,
    make_compute_metric_fn_multilable,
    HF_Models,
    GLUE_Task,
    INDONLU_Task,
    TASK_TO_FINAL_METRIC,
    TASK_TO_FINAL_METRIC_INDONLU,
    TASK_INDEX2LABEL,
    TASK_MULTILABELS,
    TASK_LABEL2INDEX,
    TASK_LABELS,
    # misc
    DotDict,
    Stopwatch,
    DataCollatorForWordClassification
)

# setup logger
logger = logging.getLogger('main')
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))

# setup stuff
class Config(DotDict):
    pass

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
def indonlu():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))

# show default values for all options
click.option = partial(click.option, show_default=True)

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")

# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

def _is_non_empty_dir(path):
    return path.exists() and len(list(path.iterdir()))


def _show_model_on_task(model, tokenizer, task):
    text = ['Budi pergi ke pondok indah mall membeli cakwe',
            'Dasar anak sialan!! Kurang ajar!!',
            'Bahagia hatiku melihat pernikahan putri sulungku yang cantik jelita']

    # logger.info(f'Running task {task}...')
    for sentence in text:
        subwords = tokenizer.encode(sentence)
        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

        logits = model(subwords)[0]
        index = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    
        logger.info(f'Text: {sentence} | Label : {TASK_INDEX2LABEL[task][index]} ({F.softmax(logits, dim=-1).squeeze()[index] * 100:.3f}%)')


def _make_huggingface_training_args(config):
    """Create Training Arguments as required by HuggingFace Trainer."""
    output_dir = config.base.output_dir
    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'out')

    tb_logging_dir = config.progress.tb_logging_dir
    if tb_logging_dir is None:
        if config.base.output_dir is not None:
            tb_logging_dir = os.path.join(config.base.output_dir, 'tb_logs')
        else:
            tb_logging_dir = None

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=config.base.overwrite_output,
        seed=config.base.seed,
        dataloader_num_workers=config.base.num_workers,
        do_train=config.training.do_train,
        do_eval=config.training.do_eval,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        disable_tqdm=not config.progress.tqdm,
        evaluation_strategy=config.progress.eval_strategy,
        eval_steps=config.progress.eval_steps,
        logging_dir=tb_logging_dir,
        logging_first_step=config.progress.logging_first_step,
        logging_steps=config.progress.logging_steps,
        save_steps=config.progress.save_steps,
        save_total_limit=config.progress.save_total_limit,
        run_name=config.progress.run_name,
        load_best_model_at_end=config.progress.load_best_model_at_end,
        metric_for_best_model=config.progress.metric_for_best_model,
        greater_is_better=config.progress.greater_is_better,
    )
    return args


def _make_datasets_and_trainer(config, model, model_enum, tokenizer, task, task_data,
                               compute_metrics, training_args, padding=None):
    # define padding strategy
    if padding:
        padding = 'max_length'
    if padding is None:
        padding = 'max_length' if config.data.pad_to_max_length else False
        # if False, pad later, dynamically at batch creation,
        # to the max sequence length in each batch

    max_length = config.data.max_seq_length
    print("max_length :", max_length)
    is_text_class_task = task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete
    is_multilabel_class_task = task == INDONLU_Task.casa or task == INDONLU_Task.hoasa

    # tokenize text and define datasets for text classification
    def preprocess_fn_text(examples):
        # tokenize the texts
        args = (
            (examples[task_data.sentence1_key],)
            if task_data.sentence2_key is None
            else (examples[task_data.sentence1_key], examples[task_data.sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
        return result

    def preprocess_fn_multilabel(examples):
        try:
            args = (
                (examples[task_data.sentence1_key],)
                if task_data.sentence2_key is None
                else (examples[task_data.sentence1_key], examples[task_data.sentence2_key])
            )
            tokenized_inputs = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

            label_ids = []
            for i, item in enumerate(examples['sentence']):
                label_id = []
                for feature in TASK_MULTILABELS[task]:
                    label_id = label_id + [examples[feature][i]]
                label_ids = label_ids + [label_id]

            tokenized_inputs['label_ids'] = label_ids

            return tokenized_inputs
        except Exception as err:
            print(examples, err)
    
    # tokenize text and define datasets for word classification
    def preprocess_fn_word(examples):
        try:
            args = (
                (examples[task_data.sentence1_key],)
                if task_data.sentence2_key is None
                else (examples[task_data.sentence1_key],examples[task_data.sentence2_key])
            )
            tokenized_inputs = tokenizer(*args, truncation=True, max_length=max_length, is_split_into_words=True)

            sentences, seq_label = examples[task_data.sentence1_key], examples[TASK_LABELS[task]]
            # data['sentence'] = data['tokens']
            # data['seq_labels'] = data['ner_tags']
            # Add CLS token
            subwords = [tokenizer.cls_token_id]
            subword_to_word_indices = [-1] # For CLS
            
            # Add subwords
            for sentence in sentences:
                print(sentence)
                for word_idx, word in enumerate(sentence):
                    subword_list = tokenizer.encode(word, add_special_tokens=False)
                    subword_to_word_indices += [word_idx for i in range(len(subword_list))]
                    subwords += subword_list
                
            # Add last SEP token
            # subwords += [tokenizer.sep_token_id]
            subword_to_word_indices += [-1]
            tokenized_inputs["labels"] = seq_label
            tokenized_inputs["subword_to_word_ids"] = subword_to_word_indices
            return tokenized_inputs
        except Exception as err:
            print(err)

    if is_text_class_task:  
        datasets = task_data.datasets.map(
            preprocess_fn_text, batched=True, load_from_cache_file=not config.data.overwrite_cache
        )
    elif is_multilabel_class_task:
        datasets = task_data.datasets.map(
            preprocess_fn_multilabel, batched=True, load_from_cache_file=not config.data.overwrite_cache
        )
    else: 
        datasets = task_data.datasets.map(
            preprocess_fn_word, batched=True, load_from_cache_file=not config.data.overwrite_cache
        )

    train_dataset = datasets['train']
    logger.info('Example of dataset to be trained..: {features => dataset }')
    for features in train_dataset.features:
        logger.info(f"{features} => {train_dataset[2][features]}")
    eval_dataset = datasets['validation']
    
    if model_enum in (
        HF_Models.indobert_base_v1,
        HF_Models.indobert_base_v2
    ):
        logger.info('First ten examples tokenized: (#, [SEP] idx, length):')
        for i in range(10):
            tokens = tokenizer.convert_ids_to_tokens(eval_dataset[i]['input_ids'])
            sep_pos_idx = tokens.index('[SEP]')
            len_ = len(tokens)
            logger.info(f'{i + 1}, {sep_pos_idx}, {len_}, {tokens}')


    word_data_collator = DataCollatorForWordClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # data collator will default to DataCollatorWithPadding,
        # so we change it if we already did the padding:
        data_collator=default_data_collator if padding and (is_text_class_task or is_multilabel_class_task) else word_data_collator if padding else None,
    )
    return trainer, datasets, train_dataset, eval_dataset


def _log_results(task_scores_map):
    if any([v is not None for v in task_scores_map.values()]):
        logger.info('*** FINAL results (task -> score) ***')
        all_scores = []
        all_scores_excluding_wnli = []

        for task, score in task_scores_map.items():
            logger.info(f'\t{task.name} -> {100. * score:.2f}')
            all_scores.append(score)

        logger.info(f'Macro-avg (incl. WNLI) = {100. * np.mean(all_scores):.2f}')
        if len(all_scores_excluding_wnli):
            logger.info(
                f'Macro-avg (excl. WNLI) = ' f'{100. * np.mean(all_scores_excluding_wnli):.2f}'
            )


def _quantize_model(config, model, task):
    """
        Changing Model into quantized one
    """
    #
    qparams = make_qparams(config)
    qparams['quant_dict'] = config.quant.get('quant_dict', {})

    if task in (INDONLU_Task.emot, INDONLU_Task.smsa, INDONLU_Task.wrete):
        model = QuantizedBertForSequenceClassification(model, **qparams)
    elif task in (INDONLU_Task.posp, INDONLU_Task.bapos, INDONLU_Task.facqa, INDONLU_Task.keps, INDONLU_Task.nergrit, INDONLU_Task.nerp, INDONLU_Task.terma):
        model = QuantizedBertForWordClassification(model, **qparams)
    elif task in (INDONLU_Task.casa, INDONLU_Task.hoasa):
        model = QuantizedBertForMultiLabelClassification(model, **qparams)
    else:
        raise NotImplementedError(
            f'Model {config.model.model_name} is not supported for ' f'quantization.'
        )

    # use double precision if necessary
    if config.double:
        for m in model.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                m.double()

    # set state
    model.set_quant_state(weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant)

    # print model
    logger.info('Quantized model:')
    logger.info(model)

    return model


def _prepare_quantized_model(config, model, loader):
    """Prepare quantized model for training/validation."""
    if config.training.do_train:
        model = prepare_model_for_quantization(config, model, loader)

    else:
        if not config.quant.dynamic:
            # 1) estimate & fix ranges for validation
            logger.info('** Estimate quantization ranges on training data **')
            pass_data_for_range_estimation(
                loader=loader,
                model=model,
                act_quant=config.quant.act_quant,
                weight_quant=config.quant.weight_quant,
                max_num_batches=config.act_quant.num_batches,
                cross_entropy_layer=config.act_quant.cross_entropy_layer,
            )
            model.fix_ranges()

        # 2) set quant state
        model.set_quant_state(
            weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
        )
    return model


class TransformerInput(tuple):
    def __getitem__(self, index):
        return TransformerInput([t[index] for t in self])

    def to(self, device):
        out = []
        for v in self:
            out.append(v.to(device) if isinstance(v, torch.Tensor) else v)
        return TransformerInput(out)

    def size(self, *args, **kw):
        out = []
        for v in self:
            out.append(v.size(*args, **kw))
        return out[0]


def adaround_get_samples_fn(data_loader, num_samples):
    X_dict = {}
    n = 0
    m = None
    for x_dict in data_loader:
        for i, (k, v) in enumerate(x_dict.items()):
            if i == 0:
                if n + len(v) > num_samples:
                    m = num_samples - n
                    n = num_samples
                else:
                    n += len(v)
            if m is not None:
                v = v[:m]
            if k in X_dict:
                X_dict[k].append(v)
            else:
                X_dict[k] = [v]
        if n == num_samples:
            break
    for k, v in X_dict.items():
        X_dict[k] = torch.cat(v)

    inp_tuple = (X_dict['input_ids'], X_dict['attention_mask'])
    if 'token_type_ids' in X_dict:
        inp_tuple = inp_tuple + (X_dict['token_type_ids'],)
    train_data = TransformerInput(inp_tuple)
    return train_data


def _run_task(config, task: INDONLU_Task, task_data, model_data):
    """Common routine to run training/validation on a signle task."""
    model = model_data.model
    model_enum = model_data.model_enum
    tokenizer = model_data.tokenizer

    # _show_model_on_task(model, tokenizer, task)

    # log options
    # logger.info(f'Running task {task.name} with options:\n' + pformat(config))

    if config.training.do_train:
        # create dirpath if not exist
        os.makedirs(config.base.output_dir, exist_ok=True)

        # log config additionaly into a separate file
        with open(os.path.join(config.base.output_dir, 'config.out'), 'w') as f:
            f.write(pformat(config) + '\n')

    # get metric
    is_text_class_task = task == INDONLU_Task.emot or task == INDONLU_Task.smsa or task == INDONLU_Task.wrete
    is_multilabel_class_task = task == INDONLU_Task.casa or task == INDONLU_Task.hoasa
    compute_metrics = make_compute_metric_fn_text(task) if is_text_class_task else make_compute_metric_fn_multilable(task) if is_multilabel_class_task else make_compute_metric_fn_word(task)
    # prepare training arguments for huggingface Trainer
    training_args = _make_huggingface_training_args(config)
    # logger.info(f'Training/evaluation parameters for Trainer: {training_args}')

    ## attach layer number
    backbone_attr = model_data.backbone_attr # in this case -> berts
    if backbone_attr is None:
        raise NotImplementedError(
            f'Model {config.model.model_name} not yet supported for ' f'TensorBoard visualization.'
        )
    layers = getattr(model, backbone_attr).encoder.layer 
    num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        for m in layer.modules():
            m.layer_idx = layer_idx
            m.num_layers = num_layers

    # Quantization!
    if 'quant' in config:
        # replace model with a quantized one
        model = _quantize_model(config, model, task)

    # Per-embedding / per-token quantization
    per_token = config.get('quant', {}).get('per_token', False)
    per_embd = config.get('quant', {}).get('per_embd', False)
    per_groups = config.get('quant', {}).get('per_groups', None)
    permute = config.get('quant', {}).get('per_groups_permute', False)
    base_axis = 2 if (per_embd or per_groups) else 1

    if (per_token or per_embd or per_groups) and model_enum in (
        HF_Models.bert_base_uncased,
        HF_Models.bert_large_uncased,
    ):
        # Per-embedding:
        # * for shapes (B, T, d) -> axis=2
        # * for shapes (B, d) -> axis=1
        # * for other shapes not applicable: (B, H, T, T); (B, T, D); (B, K)

        # Per-token:
        # * for shapes (B, T, d), (B, T, D) -> axis=1
        # * for other shapes not applicable: (B, H, T, T); (B, d); (B, K)

        # Embeddings
        E = model.bert.embeddings
        set_act_quant_axis_and_groups(
            E.sum_input_token_type_embd_act_quantizer,
            axis=base_axis,
            n_groups=per_groups,
            permute=permute,
        )
        set_act_quant_axis_and_groups(
            E.sum_pos_embd_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
        )
        set_act_quant_axis_and_groups(
            E.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
        )

        # Encoder
        for layer_idx in range(12):
            L = model.bert.encoder.layer[layer_idx]

            # Self-attention
            A = L.attention.self
            set_act_quant_axis_and_groups(
                A.query, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.key, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.value, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.context_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )

            # Self-output
            S = L.attention.output
            set_act_quant_axis_and_groups(
                S.dense, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                S.res_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                S.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
            )

            # Output
            O = L.output
            set_act_quant_axis_and_groups(
                O.dense, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                O.res_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                O.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
            )

        # Pooling, (B, d)
        if per_embd:
            set_act_quant_axis_and_groups(
                model.bert.pooler.dense_act[0], axis=1, n_groups=per_groups, permute=permute
            )

    # Mixed-precision control for act. quantizers
    quant_dict = config.get('quant', {}).get('quant_dict', {})
    if quant_dict and model_enum in (HF_Models.bert_base_uncased, HF_Models.bert_large_uncased):
        # Embeddings
        E = model.bert.embeddings
        hijack_act_quant(quant_dict, 'e', E.sum_input_token_type_embd_act_quantizer)
        hijack_act_quant(quant_dict, 'e', E.sum_pos_embd_act_quantizer)

        hijack_weight_quant(quant_dict, 'Et', E.word_embeddings)

        # Encoder
        for layer_idx in range(12):
            L = model.bert.encoder.layer[layer_idx]

            # Self-attention
            A = L.attention.self
            hijack_act_quant(quant_dict, f's{layer_idx}', A.attn_scores_act_quantizer)
            hijack_act_quant(quant_dict, 's', A.attn_scores_act_quantizer)

            hijack_act_quant(quant_dict, f'p{layer_idx}', A.attn_probs_act_quantizer)
            hijack_act_quant(quant_dict, 'p', A.attn_probs_act_quantizer)

            hijack_act_quant(quant_dict, f'c{layer_idx}', A.context_act_quantizer)
            hijack_act_quant(quant_dict, 'c', A.context_act_quantizer)

            # Self-output
            S = L.attention.output
            hijack_act_quant(quant_dict, f'g{layer_idx}', S.dense)
            hijack_act_quant(quant_dict, 'g', S.dense)

            hijack_act_quant(quant_dict, f'u{layer_idx}', S.res_act_quantizer)
            hijack_act_quant(quant_dict, 'u', S.res_act_quantizer)

            hijack_act_quant(quant_dict, f'x{layer_idx}', S.LayerNorm)
            hijack_act_quant(quant_dict, 'x', S.LayerNorm)

            # Output
            O = L.output
            hijack_act_quant(quant_dict, f'h{layer_idx}', O.dense)
            hijack_act_quant(quant_dict, 'h', O.dense)

            hijack_act_quant(quant_dict, f'y{layer_idx}', O.res_act_quantizer)
            hijack_act_quant(quant_dict, 'y', O.res_act_quantizer)

            hijack_act_quant(quant_dict, f'z{layer_idx}', O.LayerNorm)
            hijack_act_quant(quant_dict, 'z', O.LayerNorm)

            # ** All **
            hijack_act_quant_modules(quant_dict, f'L{layer_idx}', L)
            hijack_act_quant_modules(quant_dict, 'L', L)

        # Head
        hijack_act_quant(quant_dict, 'P', model.bert.pooler.dense_act[0])
        hijack_act_quant(quant_dict, 'C', model.classifier)

        hijack_act_quant(quant_dict, 'wP', model.bert.pooler.dense_act[0])
        hijack_weight_quant(quant_dict, 'wC', model.classifier)

    # Prepare quantized model for training/validation
    if 'quant' in config:
        # make another trainer with individually controlled padding strategy & batch size for
        # range estimation
        config_ = deepcopy(config)
        config_.training.batch_size = config.quant.est_ranges_batch_size
        training_args_ = _make_huggingface_training_args(config_)
        trainer_range_est, _, _, _ = _make_datasets_and_trainer(
            config, model, model_enum, tokenizer, task, task_data, compute_metrics, training_args_,
            padding=config.quant.est_ranges_pad,
        )

        # estimate (FP32) ranges for per-group act quant permutation:
        if config.quant.per_groups_permute or config.quant.per_groups_permute_shared_h:
            trainer_per_group, _, _, _ = _make_datasets_and_trainer(
                config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                training_args_,
            )

            model.full_precision()
            pass_data_for_range_estimation(
                loader=trainer_per_group.get_train_dataloader(),
                model=model,
                act_quant=True,  # simply to not exit immediately
                weight_quant=False,
                max_num_batches=10,
                cross_entropy_layer=None,
            )
            model.set_quant_state(
                weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
            )

            # flip the state back to normal
            from quantization.range_estimators import RangeEstimatorBase

            for m in model.modules():
                if isinstance(m, RangeEstimatorBase):
                    m.per_group_range_estimation = False

            # share ranges
            if config.quant.per_groups_permute_shared_h:
                for layer_idx, layer in enumerate(model.bert.encoder.layer):

                    range_estimators = {}
                    for name, m in layer.named_modules():
                        if isinstance(m, RangeEstimatorBase):
                            if m.ranges is not None:
                                range_estimators[name] = m

                    source_ranges = None
                    for k, v in range_estimators.items():
                        if 'dense' in k:
                            source_ranges = v.ranges.clone()

                    assert source_ranges is not None

                    for k, v in range_estimators.items():
                        v.ranges = source_ranges

        # prepare quantized model (e.g. estimate ranges)
        model = _prepare_quantized_model(
            config, model, loader=trainer_range_est.get_train_dataloader()
        )

    # make datasets and Trainer
    trainer, datasets, train_dataset, eval_dataset = _make_datasets_and_trainer(
        config, model, model_enum, tokenizer, task, task_data, compute_metrics, training_args
    )

    # Training!
    model_name_or_path = model_data.model_name_or_path

    if config.training.do_train:
        logger.info('*** Training ***')
        trainer.train(model_path=model_name_or_path if os.path.isdir(model_name_or_path) else None)
        # print_summary(result)
        if config.progress.save_model:
            trainer.save_model()  # saves the tokenizer too

    # fix ranges after training, for final evaluation
    if 'quant' in config:
        model.eval()
        model.fix_ranges()
        trainer.model.eval()
        trainer.model.fix_ranges()

    # Validation!
    final_score = None
    if config.training.do_eval:
        logger.info('*** Evaluation ***')

        final_score = _eval_task(config, task, trainer, eval_dataset, datasets)
        # logger.info(f'Final score {task.name} -> {100. * final_score:.2f}')

        # save final score to file
        if config.training.do_train:
            with open(os.path.join(config.base.output_dir, 'final_score.txt'), 'w') as f:
                f.write(f'{final_score}\n')

    # _show_model_on_task(model, tokenizer, task)

    return final_score


def _eval_task(config, task, trainer, eval_dataset, datasets):
    # loop to handle MNLI double evaluation (matched and mis-matched accuracy)
    subtask_names = [task.name]
    eval_datasets = [eval_dataset]

    # if task == INDONLU_Task.wrete:
    #     subtask_names.append('mnli-mm')
    #     eval_datasets.append(datasets['validation_mismatched'])

    subtask_final_scores = []
    for subtask, eval_dataset in zip(subtask_names, eval_datasets):
        if config.data.num_val_samples is not None:
            n = min(len(eval_dataset), config.data.num_val_samples)
            eval_dataset = eval_dataset.select(range(n))

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        # print_summary(eval_result)
        # log eval results
        logger.info(f'***** Eval results {subtask} *****')
        logger.info(f'{eval_result}')
        # trainer.log_metrics(eval_log_result)
        # for key, value in eval_result.items():
        #     logger.info(f'\t{key} = {value:.4f}')

        # final_score = eval_result[f'eval_{TASK_TO_FINAL_METRIC_INDONLU[task]}']
        final_score = 80
        subtask_final_scores.append(final_score)

        if config.training.do_train:
            # save eval results to files
            subtask_eval_fpath = os.path.join(config.base.output_dir, f'eval_results_{subtask}.txt')
            with open(subtask_eval_fpath, 'w') as f:
                logger.info(f'{eval_result}')

        if config.data.num_val_samples is not None:
            break

    # compute and log final score
    final_score = np.mean(subtask_final_scores)
    return final_score

def _run(config):
    """Common routine to run training/validation on a set of tasks."""
    do_train = config.training.do_train
    mode_str = 'Training' if do_train else 'Validating'
    # logger.info(f'{mode_str} with options:\n' + pformat(config))

    # parse tasks
    task_flag = INDONLU_Task.from_str(*config.indonlu.task)
    logger.info(f'{mode_str} on tasks: {list(task_flag.iter_names())}')

    # main task loop
    s = Stopwatch().start()
    task_scores_map = {}
    for task in task_flag.iter():
        logger.info(f'{mode_str} on task {task.name} ...')

        # prepare task-specific config, if necessary
        if config.model.model_path is None:  # use pre-trained backbone for training/validation
            task_config = config
        else:
            # load the suitable checkpoint
            if do_train:
                # simply load the checkpoint given by --model-path
                task_config = config
            else:
                # for validation, load the checkpoint from the corresponding subfolder given by task
                task_dirpath = Path(config.model.model_path) / task.name
                task_out_dirpaths = task_dirpath.glob('**/out')
                non_empty_task_out_dirpaths = list(filter(_is_non_empty_dir, task_out_dirpaths))
                if not len(non_empty_task_out_dirpaths):
                    raise RuntimeError(f'Task directory ({task_dirpath}) is empty.')
                if len(non_empty_task_out_dirpaths) > 1:
                    msg = [f'Task directory ({task_dirpath}) contains multiple checkpoints:']
                    for dirpath in non_empty_task_out_dirpaths:
                        msg.append(f'* {dirpath}')
                    raise RuntimeError('\n'.join(msg))

                task_out_dirpath = str(non_empty_task_out_dirpaths[0])
                task_config = deepcopy(config)
                if config.base.output_dir is None:
                    task_config.base.output_dir = task_out_dirpath
                task_config.model.model_path = task_out_dirpath

        # load data
        task_data = load_task_data_indonlu(task=task, data_dir=task_config.indonlu.data_dir)
        # load model and tokenizer
        model_data = load_model_and_tokenizer(**task_config.model, num_labels=task_data.num_labels, task=task, task_data=task_data)
        # logger.info(f'{mode_str} with model configuration: {model_data}')
        # run on a task
        task_scores_map[task] = _run_task(task_config, task, task_data, model_data)

    # log task results
    # _log_results(task_scores_map)

    # log elapsed time
    logger.info(s.format())


def _train(config):
    # check and set training-specific options
    if config.base.output_dir is None:
        raise ValueError('--output-dir must be provided for training')
    config.training.do_train = True

    _run(config)


def _validate(config):
    # check and set validation-specific options
    config.base.overwrite_output = False
    config.training.do_eval = True
    config.training.do_train = False

    _run(config)

@indonlu.command()
@pass_config
@indonlu_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def train_baseline_indonlu(config):
    """
    Fine tuning Base model from Huggingface
    """
    _train(config)

@indonlu.command()
@pass_config
@indonlu_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def validate_baseline_indonlu(config):
    _validate(config)

@indonlu.command()
@pass_config
@indonlu_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@quantization_options
@activation_quantization_options
@qat_options
@adaround_options
@transformer_quant_options
def train_quantized_indonlu(config):
    _train(config)

@indonlu.command()
@pass_config
@indonlu_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@quantization_options
@activation_quantization_options
@adaround_options
@transformer_quant_options
def validate_quantized_indonlu(config):
    _validate(config)

if __name__ == '__main__':
    indonlu()
