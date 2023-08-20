# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import numbers
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel)

from models.code_representation.codebert.CodeBertModel import CodeBertModel
from models.code_representation.codebert.ConcatClsModel import ConcatClsModel
from models.code_representation.codebert.ConcatInlineModel import ConcatInlineModel
from models.evaluation.evaluation import evaluate_result, evaluatelog_result

logger = logging.getLogger(__name__)
max_feature_length = []
max_code_length = []


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 num_features
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.num_features = num_features


def codebert(js, tokenizer, args):
    num_features = []
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['label'], num_features)


def concat_inline(js, tokenizer, args):
    code = ' '.join(js['code_no_comment'].split())
    global max_code_length
    global max_feature_length
    features = ''
    features += "The code contains {} lines and ".format(round(js["lc"], 2))
    features += "its complexity metrics values are {}, {} and {}.".format(round(js["pi"], 2), round(js["ma"], 2),
                                                                          round(js["ml"], 2))
    features += " The nested block depth is {} ".format(round(js["nbd"], 2))
    features += "and the difficulty of this code is {}".format(round(js["d"], 2))
    features += "Maintainability score is {} ".format(round(js["mi"], 2))
    features += "and this method calls {} number of methods while its readability and effort metrics values are {} " \
                "and {}".format(round(js["fo"], 2), round(js["r"], 2), round(js["e"], 2))

    feature_tokens = tokenizer.tokenize(features)

    max_feature_length.append(len(feature_tokens))

    feature_tokens = feature_tokens[:128]

    code_tokens = tokenizer.tokenize(code)

    max_code_length.append(len(code_tokens))

    code_tokens = code_tokens[: args.block_size - len(feature_tokens) - 3]

    source_tokens = [tokenizer.cls_token] + feature_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    num_features = []
    return InputFeatures(source_tokens, source_ids, js['label'], num_features)


def concat_cls(js, tokenizer, args):
    num_features = [round(js["lc"], 2), round(js["pi"], 2), round(js["ma"], 2), round(js["ml"], 2), round(js["nbd"], 2),
                    round(js["d"], 2), round(js["mi"], 2), round(js["fo"], 2), round(js["r"], 2), round(js["e"], 2)]
    code = ' '.join(js['code_no_comment'].split())
    code_tokens = tokenizer.tokenize(code)
    code_tokens = code_tokens[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(source_tokens, source_ids, js['label'], num_features)


# creating features based on args value for different architectures
def convert_examples_to_features(js, tokenizer, args):
    model_arch = args.model_arch
    if model_arch == 'CodeBERT':
        return codebert(js, tokenizer, args)
    elif model_arch == 'ConcatInline':
        return concat_inline(js, tokenizer, args)
    elif model_arch == 'ConcatCLS':
        return concat_cls(js, tokenizer, args)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
            self.examples[i].num_features)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer, lr, epoch, batch_size, fine_tune):
    training_phase = ""
    if not fine_tune:
        training_phase = "Training"
        for param in model.encoder.base_model.parameters():
            param.requires_grad = False
    else:
        training_phase = "Fine Tuning"
        for param in model.encoder.base_model.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total param {}".format(total_params))
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=batch_size, num_workers=4, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    logger.info("***** Running {} *****".format(training_phase))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epoch)
    logger.info("  batch size = %d", batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_f1 = 0.0
    model.zero_grad()
    losses_eval = []
    losses_train = []

    last_loss = 100
    patience = 3
    trigger_times = 0

    for idx in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []

        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            num_features = batch[2].to(args.device)
            model.train()
            loss, logits = model(input_ids=inputs, num_features=num_features, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(np.mean(losses), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        results = evaluate(args, model, tokenizer)
        log_result(results, logger)
        losses_eval.append(results["eval_loss"])
        losses_train.append(np.mean(losses))

        if not fine_tune:
            if results["eval_loss"] > last_loss:
                trigger_times += 1
                logger.info('  Trigger Times Increased to %d', trigger_times)

                if trigger_times >= patience:
                    logger.info('  Early stopping! Training being finished!')
                    return model
            else:
                logger.info('  Trigger times reset to 0')
                trigger_times = 0

            last_loss = results["eval_loss"]

        # Save model checkpoint
        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            logger.info("  " + "*" * 20)
            logger.info("  Best F1:%s", round(best_f1, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            # add arg parameters
            output_dir = os.path.join(output_dir, model_filename(args))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

    return model


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        num_features = batch[2].to(args.device)
        with torch.no_grad():
            # lm_loss, logit = model(inputs, label)
            lm_loss, logit = model(input_ids=inputs, num_features=num_features, labels=label)

            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    probs = logits

    result = evaluate_result(labels, preds, probs)
    result["eval_loss"] = float(perplexity)
    return result


def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    test_loss = 0.0
    nb_test_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        num_features = batch[2].to(args.device)
        with torch.no_grad():
            # lm_loss, logit = model(inputs, label)
            lm_loss, logit = model(input_ids=inputs, num_features=num_features, labels=label)

            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

            test_loss += lm_loss.mean().item()
        nb_test_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    probs = logits
    with open(os.path.join(args.output_dir, "predictions-{}.txt".format(model_filename(args))), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            f.write(str(pred) + '\n')

    test_loss = test_loss / nb_test_steps
    perplexity = torch.tensor(test_loss)
    evaluatelog_result(labels, preds, probs, model_filename(args).replace(".bin", ""), logger)

    result = {
        "test_loss": float(perplexity),
    }

    log_result(result, logger)


def log_result(result, logger):
    for key, value in sorted(result.items()):
        if isinstance(value, numbers.Number):
            value = round(value, 4)
        logger.info("  %s = %s", key, value)


def model_filename(args):
    # return '{}'.format(
    #     '{}-bs:{}-tb:{}-eb:{}.bin'.format(args.model_arch, args.block_size, args.train_batch_size,
    #                                       args.eval_batch_size))
    return '{}'.format('{}.bin'.format(args.model_arch))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_arch", default="ConcatInline", type=str, required=True,
                        help="model type for training")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")
    parser.add_argument('--num_finetune_epochs', type=int, default=5,
                        help="num_finetune_epochs")
    parser.add_argument("--train_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--finetune_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--finetune_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    if not os.path.exists("log"):
        os.makedirs("log")
    file_handler = logging.FileHandler(os.path.join("log", "log-{}.txt".format(model_filename(args))))
    logger.addHandler(file_handler)

    # Set seed
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 4
    config.output_hidden_states = True

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    # choosing between different architectures based on the arg value
    #################################
    model_arch = args.model_arch
    if model_arch == 'CodeBERT':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = CodeBertModel(model, config, tokenizer, args)
    elif model_arch == 'ConcatInline':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = ConcatInlineModel(model, config, tokenizer, args)
    elif model_arch == 'ConcatCLS':
        model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, add_pooling_layer=False)
        model = ConcatClsModel(encoder=model, config=config, tokenizer=tokenizer, args=args)
    ##################################

    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        # model = train(args, train_dataset, model, tokenizer, lr=10e-5, epoch=20, batch_size=32, fine_tune=False)
        # train(args, train_dataset, model, tokenizer, lr=2e-5, epoch=5, batch_size=4, fine_tune=True)

        model = train(args, train_dataset, model, tokenizer, lr=args.train_learning_rate, epoch=args.num_train_epochs,
                      batch_size=args.train_batch_size, fine_tune=False)
        train(args, train_dataset, model, tokenizer, lr=args.finetune_learning_rate, epoch=args.num_finetune_epochs,
              batch_size=args.finetune_batch_size, fine_tune=True)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        output_dir = os.path.join(output_dir, model_filename(args))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        log_result(result, logger)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        output_dir = os.path.join(output_dir, model_filename(args))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
