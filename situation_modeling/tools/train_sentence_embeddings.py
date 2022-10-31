from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,SentenceEvaluator
from sentence_transformers.models.tokenizer import WhitespaceTokenizer
from sentence_transformers.model_card_templates import ModelCardTemplate
from tqdm.autonotebook import trange
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download
from sentence_transformers.models import Transformer,LSTM
from situation_modeling import initialize_config,get_config
from optparse import OptionParser,OptionGroup,Values
import logging
from datetime import datetime
import sys
import os
import gzip
import json
import csv
import wandb
import math
import time
import numpy as np
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn,Tensor
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything

from ..utils import (
    setup_wandb,
    set_seed,
    save_wandb,
    shuffle_lines
)

from situation_modeling.utils.model_analysis import situation_analysis,babi_analysis
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from torch.optim import Optimizer

util_logger = logging.getLogger('train_sentence_embeddings')


### SPECIAL TOKENIZER

### modified the english stop words from original tokenizer, a little too much pruning out of words for our experiments
ENGLISH_STOP_WORDS = [
    '!', '"', "''", "``", '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
    '^', '_', '`',  '{', '|', '}', '~', 'a', 'the','an'
]

class CustomTokenizer(WhitespaceTokenizer):
    def __init__(
            self,
            vocab: Iterable[str] = [],
            stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
            do_lower_case: bool = False
        ):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

class CustomTransformer(Transformer):
    """Slight modified Transformer model from sentence transformers which allows for 
    using decoder-decoder models like `T5`. 
    """

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            'input_ids': features['input_ids'],
            'attention_mask': features['attention_mask']
        }

        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        ### to allow for t5 encoder
        if hasattr(self.auto_model,'decoder'):
            output_states = self.auto_model.encoder(**trans_features)
        else: 
            output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({
            'token_embeddings': output_tokens,
            'cls_token_embeddings': cls_tokens,
            'attention_mask': features['attention_mask']
        })

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

class CustomLSTM(LSTM):
    def forward(self, features):
        token_embeddings = features['token_embeddings']
        sentence_lengths = torch.clamp(features['sentence_lengths'], min=1)

        ## see issue here: https://github.com/UKPLab/sentence-transformers/pull/933/commits/11f1f10dba9c48c9da51207ec2f129ba959e3031
        #packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths, batch_first=True, enforce_sorted=False)
        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed = self.encoder(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0]
        features.update(
            {'token_embeddings': unpack}
        )
        return features

    
    
# class CustomLoss(nn.Module):
#     @torch.no_grad()
#     def run_eval(self,dataloader,split='dev',out_file=None):
#         """Runs an evaluation given a dataloader 

#         :param dataloader: the input dataloader 
#         """
#         dataloader.collate_fn = self.model.smart_batching_collate
        
#         num_correct = 0.
#         total_predictions = 0.
#         global_loss = torch.zeros(1)
#         util_logger.info(f'Running evaluation for {split}.....')
#         counter = 0
#         out_list = []

#         for (features,labels,guids) in tqdm(dataloader):
#             loss_value,probs = self(features, labels)
#             correct = torch.sum(labels.view(-1) == torch.argmax(probs,dim=1).view(-1)).cpu().tolist()
#             total = labels.view(-1).size()[0]

#             num_correct += correct
#             total_predictions += total
#             global_loss += loss_value.cpu()
#             counter += 1

#             # if ofile:
#             #     out_list.append({"guid"})

#         util_logger.info(
#             f'acc={num_correct/total_predictions}, loss={global_loss.tolist()[0]}'
#         )

#         ## print out (if set to)


#         return {
#             "acc"  : num_correct / total_predictions,
#             "loss" : global_loss.tolist()[0] / counter, 
#         }
    
    
class CustomSoftmaxLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(CustomSoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        util_logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)

        ###
        prob_outputs = output.softmax(dim=-1)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss,prob_outputs
        return reps, prob_outputs

    @torch.no_grad()
    def run_eval(self,dataloader,split='dev',out_file=None):
        """Runs an evaluation given a dataloader 

        :param dataloader: the input dataloader 
        """
        dataloader.collate_fn = self.model.smart_batching_collate
        
        num_correct = 0.
        total_predictions = 0.
        global_loss = torch.zeros(1)
        util_logger.info(f'Running evaluation for {split}.....')
        counter = 0
        outputs = []
        

        for (features,labels,print_data) in tqdm(dataloader):
            loss_value,probs = self(features, labels)

            label_predictions = torch.argmax(probs,dim=1).view(-1)
            gold_labels = labels.view(-1)
            correct = torch.sum(gold_labels == label_predictions).cpu().tolist()
            total = labels.view(-1).size()[0]

            num_correct += correct
            total_predictions += total
            global_loss += loss_value.cpu()
            counter += 1

            ### store outputs for printing to file 
            if out_file is not None:
                guids = print_data["guids"]
                orig_pairs = print_data["orig_pairs"]
                
                for idx in range(total):
                    outputs.append({
                        "guid"        : guids[idx],
                        "story"       : orig_pairs[idx][0],
                        "proposition" : orig_pairs[idx][1],
                        "gold"        : REVERSE_LABELS[gold_labels[idx].cpu().item()],
                        "predicted"   : REVERSE_LABELS[label_predictions[idx].cpu().item()],
                    })

        util_logger.info(
            f'acc={num_correct/total_predictions}, total_predictions={total_predictions}, loss={global_loss.tolist()[0]}'
        )
        if out_file is not None:
            with open(out_file,'w') as results_out:
                for instance in outputs:
                    results_out.write(json.dumps(instance))
                    results_out.write('\n')

        return {
            "acc"  : num_correct / total_predictions,
            "loss" : global_loss.tolist()[0] / counter, 
        }

# class BiLinearSoftmaxLoss(CustomLoss):
#     def __init__(self,
#                  model: SentenceTransformer,
#                  sentence_embedding_dimension: int,
#                  num_labels: int,
#                  concatenation_sent_rep: bool = True,
#                  concatenation_sent_difference: bool = True,
#                  concatenation_sent_multiplication: bool = False,
#                  loss_fct: Callable = nn.CrossEntropyLoss()):
#         super(BiLinearSoftmaxLoss, self).__init__()
#         self.model = model
#         self.num_labels = num_labels

#         self.classifier = nn.Bilinear(
#             sentence_embedding_dimension,
#             sentence_embedding_dimension,
#             num_labels
#         )
#         self.loss_fct = loss_fct

#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
#         rep_a, rep_b = reps
#         output = self.classifier(rep_a,rep_b)

#         ###
#         prob_outputs = output.softmax(dim=-1)

#         if labels is not None:
#             loss = self.loss_fct(output, labels.view(-1))
#             return loss,prob_outputs
#         return reps, prob_outputs

class CustomSentenceTransformer(SentenceTransformer):
    """Custom sentence transformer model used to log some training details 

    """
    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []
        guids = []
        meta_data = {}
        pairs = []

        for example in batch:
            sentence_pair = []
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
                sentence_pair.append(text)

            labels.append(example.label)
            guids.append(example.guid)
            pairs.append(tuple(sentence_pair))
            
        labels = torch.tensor(labels).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        meta_data["guids"] = guids
        meta_data["orig_pairs"] = pairs

        return sentence_features, labels, meta_data #guids

    
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            use_wandb=False,
            log_interval=100,
            dev_dataloader=None,
            patience=1000000000,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. 
             It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size 
            from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, 
           warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal 
            learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """
        ##Add info to model card
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(
                ModelCardTemplate.get_train_objective_info(dataloader, loss)
            )

        info_loss_functions = "\n\n".join(
            [text for text in info_loss_functions]
        )

        info_fit_parameters = json.dumps({
            "evaluator": fullname(evaluator),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "optimizer_class": str(optimizer_class),
            "optimizer_params": optimizer_params,
            "weight_decay": weight_decay,
            "evaluation_steps": evaluation_steps,
            "max_grad_norm": max_grad_norm
        }, indent=4, sort_keys=True)

        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.\
          replace("{LOSS_FUNCTIONS}", info_loss_functions).\
          replace("{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate
        if dev_dataloader:
            dev_dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999
        self.best_loss  = 9999999
        non_improvement = 0
        
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)


        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False


        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):

            training_steps = 0
            loss_counts = 0
            global_train_loss = torch.zeros(1)
            num_correct = 0.
            total_predictions = 0.

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            tepoch = trange(steps_per_epoch,smoothing=0.05)
            data_iterator = data_iterators[0]
            util_logger.info(f"starting epoch {epoch}, number of steps={steps_per_epoch}")
            util_logger.info(f'size of training set: {len(data_iterator)}')

            loss_model = loss_models[0]
            optimizer = optimizers[0]
            scheduler = schedulers[0]
            #data_iterator = data_iterators[0]

            for _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                #for train_idx in range(num_train_objectives):
                # loss_model = loss_models[train_idx][0]
                # optimizer = optimizers[train_idx][0]
                # scheduler = schedulers[train_idx][0]
                # data_iterator = data_iterators[train_idx][0]

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[0])
                    data_iterators[0] = data_iterator
                    data = next(data_iterator)

                features, labels, _ = data


                if use_amp:
                    raise ValueError(
                        'Shut off in this version'
                    )
                    # with autocast():
                    #     loss_value = loss_model(features, labels)
                    
                    # scale_before_step = scaler.get_scale()
                    # scaler.scale(loss_value).backward()
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    # scaler.step(optimizer)
                    # scaler.update()

                    # skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    #loss_value = loss_model(features, labels)
                    loss_value,probs = loss_model(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()

                ## record for later 
                global_train_loss += loss_value.cpu()
                loss_counts += 1.
                        
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                ## collect batch accuracy info
                correct = torch.sum(labels.view(-1) == torch.argmax(probs,dim=1).view(-1)).cpu().tolist()
                total = labels.view(-1).size()[0]
                num_correct += correct
                total_predictions += total
                accuracy = correct / total

                ### current training accuracy
                tepoch.set_postfix(
                    train_loss=loss_value.cpu().tolist(),
                    acc=100. * accuracy,
                )
                
                ### logging indent
                if use_wandb:
                    current_lr = scheduler.get_last_lr()[0]
                    ## log training loss 
                    if training_steps % log_interval == 0:
                        wandb.log({
                            "step" : epoch,
                            "learning_rate" : current_lr,
                            "train_loss" : loss_value,
                            "step_map" : epoch,
                        })
                    else:
                        wandb.log({
                            "step" : epoch,
                            "learning_rate" : current_lr,
                            "step_map" : epoch,
                        })

            ### eval indent
            if dev_dataloader:
                results = loss_model.run_eval(dev_dataloader)
                if use_wandb:
                    wandb.log({
                        "val_acc" : results["acc"],
                        "val_loss" : results["loss"],
                        "avg_train_loss" : global_train_loss / loss_counts,
                        "train_acc" : num_correct / total_predictions if total_predictions > 0 else 0.,
                        "step_map" : epoch,
                    })
                if results["acc"] > self.best_score:
                    self.best_score = results["acc"]
                    self.best_loss = results["loss"]
                    torch.save(loss_model.state_dict(),os.path.join(output_path,"best.pt"))
                    non_improvement = 0
                else:
                    non_improvement += 1
                    if non_improvement >= patience:
                        util_logger.info('Losing patience, stopping')
                        break

            #self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

            # with trange(steps_per_epoch) as tepoch:
            # # for instance_number in trange(steps_per_epoch,
            # #                                   desc="Iteration",
            # #                                   smoothing=0.05,
            # #                                   disable=not show_progress_bar):
            #     for instance_number in tepoch:
            #         tepoch.set_description(f"Epoch {epoch}")
                    
            #         for train_idx in range(num_train_objectives):
            #             loss_model = loss_models[train_idx]
            #             optimizer = optimizers[train_idx]
            #             scheduler = schedulers[train_idx]
            #             data_iterator = data_iterators[train_idx]

            #             try:
            #                 data = next(data_iterator)
            #             except StopIteration:
            #                 data_iterator = iter(dataloaders[train_idx])
            #                 data_iterators[train_idx] = data_iterator
            #                 data = next(data_iterator)

            #             features, labels = data

            #             loss_value,probs = loss_model(features, labels)
            #             loss_value.backward()
            #             torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
            #             optimizer.step()

            #             global_train_loss += loss_value.cpu()
            #             loss_counts += 1.

            #             optimizer.zero_grad()

            #             # if not skip_scheduler:
            #             if not skip_scheduler:
            #                 scheduler.step()

            #             ## compute step accuracy
            #             correct = torch.sum(labels.view(-1) == torch.argmax(
            #                 probs,dim=1).view(-1)).cpu().tolist()

            #             total = labels.view(-1).size()[0]
            #             num_correct += correct
            #             total_predictions += total
            #             accuracy = correct / total

            #             if use_wandb: # and instance_number % log_interval == 0:
            #                 wandb.log({"step" : epoch})
            #                 current_lr = scheduler.get_last_lr()[0]
            #                 wandb.log({"learning_rate" : current_lr})
            #                 if instance_number % log_interval == 0:
            #                     wandb.log({"train_loss" : loss_value})

            #             if self.best_score != -9999999: 
            #                 tepoch.set_postfix(
            #                     train_loss=loss_value.cpu().tolist(),
            #                     acc=100. * accuracy,
            #                     val_acc=self.best_score
            #                 )
            #             else:
            #                 tepoch.set_postfix(
            #                     train_loss=loss_value.cpu().tolist(),
            #                     acc=100. * accuracy
            #                 )
            #             #time.sleep(0.1)

            #         training_steps += 1
            #         global_step += 1

            # ### log average loss
            # if use_wandb:
            #     wandb.log({"avg_train_loss" : global_train_loss / loss_counts})
            #     wandb.log({"train_acc" : num_correct / total_predictions if total_predictions > 0 else 0.})

            # ### iterate through validation if provided
            # if dev_dataloader:
            #     results = loss_model.run_eval(dev_dataloader)
            #     if use_wandb:
            #         wandb.log({"val_acc" : results["acc"]})
            #         wandb.log({"val_loss" : results["loss"]})
            #     if results["acc"] > self.best_score:
            #         self.best_score = results["acc"]
            #         self.best_loss = results["loss"]
            #         torch.save(loss_model.state_dict(),os.path.join(output_path,"best.pt"))
            #     else:
            #         non_improvement += 1
            #         if non_improvement >= patience:
            #             util_logger.info('Losing patience, stopping')
            #             break
                    
            #     ## turn back on training mode (just in case)
            #     for loss_model in loss_models:
            #         loss_model.zero_grad()
            #         loss_model.train()

        ###
        return {
            "best_dev_accuracy" : self.best_score,
            "best_dev_loss"     : self.best_loss,
            "training_accuracy" : num_correct / total_predictions,
            #"avg_train_loss"    : (global_train_loss / loss_counts).cpu().tolist(),
        }


def params(config):
    """Params for this module

    :param config: the global configuration
    """
    group = OptionGroup(config,"situation_modeling.tools.train_sentence_embeddings",
                            "Parameters for running and training models")

    group.set_conflict_handler("resolve")

    from ..runner import params as r_params
    r_params(config)

    group.add_option("--sentence_bert_data",
                         dest="sentence_bert_data",
                         default='',
                         type=str,
                         help="The sentence bert data [default='']")

    group.add_option("--log_interval",
                         dest="log_interval",
                         default=100,
                         type=int,
                         help="The amount of logging to do [default=100]")

    group.add_option("--representation_eval",
                         dest="representation_eval",
                         action='store_true',
                         default=False,
                         help="Run representation learning eval on sts (sanity checking purporses) [default=False]")

    group.add_option("--no_future",
                         dest="no_future",
                         action='store_true',
                         default=False,
                         help="Do not include future stuff [default=False]")

    group.add_option("--no_story",
                         dest="no_story",
                         action='store_true',
                         default=False,
                         help="No story baseline, replace story with random string [default=False]")

    group.add_option("--check_truncation",
                         dest="check_truncation",
                         action='store_true',
                         default=False,
                         help="do a check of truncation [default=False]")

    group.add_option("--warmup_steps",
                         dest="warmup_steps",
                         default=0.1,
                         type=float,
                         help="warmnup steps [default=0.1]")

    group.add_option("--embedding_model",
                         dest="embedding_model",
                         default="transformer",
                         type=str,
                         help="The type of embedding model to use [default='transformer']")

    group.add_option("--breakpoint_symbol",
                         dest="breakpoint_symbol",
                         default="#",
                         type=str,
                         help="The type of symbol to use as a breakpoint [default='#']")

    config.add_option_group(group)

SITUATION_LABELS = {
    0.0    : 0,
    1.0    : 1,
    0.5    : 2,
    "0.0"  : 0,
    "1.0"  : 1,
    "0.5"  : 2,
    "yes"  : 1,
    "no"   : 0,
    "maybe": 2,
}

REVERSE_LABELS = {
    1  : "yes",
    0  : "no",
    2  : "maybe",
    1. : "yes",
    0. : "no",
    2. : "maybe",    
}

class CustomInputExample(InputExample):
    pass


def build_data(config,split,tokenizer=None):
    """Build data for a specific split 

    :param config: the global configuration 
    param split: the data split 
    """
    if split == "train":
        data_path = os.path.join(config.data_dir,f"{config.train_name}.jsonl")
    elif split == "dev":
        data_path = os.path.join(config.data_dir,f"{config.dev_name}.jsonl")
    elif split == "test":
        data_path = os.path.join(config.data_dir,f"{config.test_name}.jsonl")
    else:
        raise ValueError(
            f'Unknown split name: {split}'
        )
    printed = 0
    sample = []
    lens_s = []
    lens_p = []

    # randomly shuffle lines of data
    if config.max_data != 100000000:
        util_logger.info('Shuffling data')
        shuffled_data = shuffle_lines(data_path)
    else:
        with open(data_path) as data:
            shuffled_data = [line for line in data]

    for m,line in enumerate(shuffled_data):
        json_line = json.loads(line)
        texts = json_line["texts"]
        prop_lists = json_line["prop_lists"]
        labels = json_line["outputs"]
        if split == "train" and m <=5:
            util_logger.info(f'TRAIN INSTANCE\n-------------\n \t{json_line}')

        for z,text in enumerate(texts):
            before = [t+"." if "." not in t else t for t in texts[:z+1]]
            after = [t+"." if "." not in t else t for t in texts[z+1:]]
            if config.no_future: 
                full_text = f"{' '.join(before)}"
            else:
                full_text = f"{' '.join(before)} {config.breakpoint_symbol} {' '.join(after)}"

            #### no_story baseline, jsut prediction just based on fact 
            if config.no_story:
                full_text = "empty"
                
            for k,prop in enumerate(prop_lists[z]):
                if split == "train" and printed > config.max_data: continue
                out_label = labels[z][k]
                sample.append(
                    InputExample(
                        #texts=[text,prop],
                        texts=[full_text,prop],
                        label=SITUATION_LABELS[out_label],
                        guid=f"instance_{m}_text_{z}_prop_{k}",
                    ))
                if split == "train" and m <= 5:
                    util_logger.info(f'\t\t [{full_text},{prop}],label={out_label}={SITUATION_LABELS[out_label]}')
                printed += 1

                ## check truncation
                if tokenizer is not None:
                    lens_s.append(len(tokenizer.tokenize(full_text)))
                    lens_p.append(len(tokenizer.tokenize(prop)))
                else:
                    lens_s.append(len(full_text.split()))
                    lens_p.append(len(prop.split()))
                        
    ###
    util_logger.info(
        f'number of sample instances: {len(sample)},split={split}'
    )

    ## truncation information 
    if lens_s and lens_p:
        greater_s = [l for l in lens_s if l+2 >= config.max_seq_length]
        greater_p = [l for l in lens_p if l+2 >= config.max_seq_length]
        util_logger.info(
            f'stories: max_length={np.max(lens_s)},avg={np.mean(lens_s)},truncated={len(greater_s)} / {len(lens_s)}'
        )
        util_logger.info(
            f'stories: max_length={np.max(lens_p)},avg={np.mean(lens_p)},truncated={len(greater_p)} / {len(lens_p)}'
        )
    return sample

def load_modules(config,num_labels=3):
    """Loads the sentence transformer model, possibly from file (if available)

    :param config: global configuration 
    """
    ##
    existing = config.load_existing
    
    if existing:
        ## load the previous config
        old_config = config
        util_logger.info('Loading existing model config')
        with open(os.path.join(os.path.dirname(existing),"model_config.json")) as base_config:
            config = Values(json.loads(base_config.read()))
        if old_config.max_seq_length > config.max_seq_length:
            config.max_seq_length = old_config.max_seq_length
    
    ### encoder 
    ##pooler
    if config.embedding_model == "transformer":

        word_embedding_model = CustomTransformer(
            config.base_model,
            max_seq_length=config.max_seq_length
        )
        
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        ## sentence transformer
        model = CustomSentenceTransformer(
            modules=[word_embedding_model,pooling_model],
        )
        ### train loss function
        #lclass = BiLinearSoftmaxLoss if config.loss_type == 'bilinear' else CustomSoftmaxLoss
        #util_logger.info(f'loss type: {config.loss_type}')
    #train_loss = lclass(

    elif config.embedding_model == "bilstm":

        word_embedding_model = models.WordEmbeddings.from_text_file(
            'glove.6B.300d.txt.gz',
            tokenizer=CustomTokenizer() ##<--- special tokenzier to avoid getting rid of certain things 
        )
        #if config.embedding_model == "bilstm": 
        lstm = CustomLSTM(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            hidden_dim=1024
        )
        pooling_model = models.Pooling(
            lstm.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=True
        )
        model = CustomSentenceTransformer(
            modules=[word_embedding_model, lstm, pooling_model]
        )
        # else:
        #     pooling_model = models.Pooling(
        #         word_embedding_model.get_word_embedding_dimension(),
        #         pooling_mode_mean_tokens=True,
        #         pooling_mode_cls_token=False,
        #         pooling_mode_max_tokens=False
        #     )
        #     sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
        #     dan1 = models.Dense(
        #         in_features=sent_embeddings_dimension,
        #         out_features=sent_embeddings_dimension
        #     )
        #     dan2 = models.Dense(
        #         in_features=sent_embeddings_dimension,
        #         out_features=sent_embeddings_dimension
        #     )
        #     model = CustomSentenceTransformer(
        #         modules=[word_embedding_model, pooling_model, dan1, dan2]
        #     )
    else:
        raise ValueError(
            f'Unknown embedding model! {config.embedding_model}'
        )

    train_loss = CustomSoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=num_labels,
    )
    if existing:
        util_logger.info(f'Loading existing model...path={existing}')
        train_loss.load_state_dict(torch.load(existing))
        train_loss.to(model._target_device)

    ### backup new model config 
    elif config.output_dir:
        with open(os.path.join(config.output_dir,"model_config.json"),'w') as mconfig:
            mconfig.write(json.dumps(config.__dict__,indent=4))

    return (train_loss,model)



def main(argv):
    """Main execution point, will train a sentence transformer model 

    :param argv: the cli input 
    
    taken from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli.py
    """
    ## set up 
    config = initialize_config(argv,params)
    setup_wandb(config)
    seed_everything(config.seed, workers=False)

    if config.wandb_project or config.wandb_entity:
        wandb.init()

    ########
    # DATA #
    ########
    train_dataloader = None; dev_dataloader = None; test_dataloader = None
    ## if possible, load tokenizer to monitor truncate
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    except:
        tokenizer = None 
    ###
    
    ## train dataloader
    if not config.no_training: 
        train_dataloader = DataLoader(
            build_data(config,split='train',tokenizer=tokenizer),
            shuffle=True,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
        )

    if not config.no_training or config.dev_eval: 
        ### dev dataloader
        dev_dataloader = DataLoader(
            build_data(config,split='dev',tokenizer=tokenizer),
            shuffle=False,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
        )
    if config.test_eval:
        test_dataloader = DataLoader(
            build_data(config,split='test',tokenizer=tokenizer),
            shuffle=False,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
        )

    ####################
    # RUN THE TRAINER  #
    ####################
    metrics = {}

    if not config.no_training:

        ### load the modules 
        train_loss,model = load_modules(config)

        if config.wandb_project or config.wandb_entity:
            wandb.watch(
                model,
                log_freq=config.log_interval,
                log=None,
            )

        warmup_steps = math.ceil(
            len(train_dataloader) * config.num_train_epochs * config.warmup_steps
        )

        util_logger.info("Warmup-steps: {}".format(warmup_steps))

        train_metrics = model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.num_train_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=config.output_dir,
            use_wandb=True if config.wandb_project and config.wandb_entity \
            else False,
            dev_dataloader=dev_dataloader,
            optimizer_params={
                "lr" : config.learning_rate,
            },
            patience=config.patience if config.early_stopping else 1000000000000,
        )
        
        ###
        metrics.update(train_metrics)
        config.load_existing = os.path.join(config.output_dir,"best.pt")

    ### do dev or test run 
    if config.dev_eval or config.test_eval or config.train_eval:

        ## load the model again (possible from existing best model)
        util_logger.info('Loading existing model..')
        train_loss,model = load_modules(config)
        train_loss.eval()

        for split,dataloader,switched_on,split_name in [
                ("train",train_dataloader,config.train_eval,config.train_name),
                ("dev",dev_dataloader,config.dev_eval,config.dev_name),
                ("test",test_dataloader,config.test_eval,config.test_name),
            ]:
            if dataloader is None or switched_on is False: continue
            util_logger.info(f'running eval on {split}')
            out_file = None
            if config.print_output:
                out_file = os.path.join(config.output_dir,f"{split_name}_eval.jsonl")

            results = train_loss.run_eval(dataloader,out_file=out_file)
            metrics[f"{split}_eval"] = results["acc"]
            metrics[f"{split}_loss"] = results["loss"]

    ### print metrics
    util_logger.info(metrics)

    more_metrics = situation_analysis(config,convert=True)
    if "babi" in config.wandb_data:
        frame = babi_analysis(config,convert=True)
        more_metrics.update(frame)
    if isinstance(more_metrics,dict): 
        metrics.update(more_metrics)
        
    if config.output_dir:
        util_logger.info('Printing the metrics output...')
        with open(os.path.join(config.output_dir,"metrics.json"),'w') as mout:
            mout.write(json.dumps(metrics,indent=4))

if __name__ == "__main__":
    main(sys.argv[1:])
