import os
import torch
from torchtyping import TensorType
from torch import Tensor,nn
from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from torch.autograd import Variable

from ..evaluation import round_acc_eval, close_acc_eval, band_acc_eval
from..utils.gen_utils import probs_to_bins, util_logger
from .loss_base import ScoreLoss
from .soft_logic_utils import compute_logic_loss
from .logic_utils import compute_constraint_loss

def _run_eval(
        pred_labels,
        labels,
        prop_mask,
        eval_type = "by_prop"
    ):
    """Computes accuracy metric over labels.

    :param pred_labels: the prediced situation probabilities as labels 
    :param labels: the target labels
    :param prop_mask: the proposition mask
    """
    if eval_type == "match_rows":
        correct_rows: TensorType["B", "T"] = torch.all(torch.eq(labels, pred_labels),dim=2)
        row_mask: TensorType["B", "T"] = torch.any(prop_mask,dim=2)
        
        # apply mask
        masked_correct: TensorType["B", "T"] = torch.logical_and(correct_rows,row_mask)

        num_correct_rows = masked_correct.count_nonzero()
        num_unmasked_rows = row_mask.count_nonzero()

        num_correct = num_correct_rows
        num_total = num_unmasked_rows
        acc = num_correct_rows / num_unmasked_rows

    elif eval_type=="by_prop":
        
        correct_preds: TensorType["B", "T", "P"]  = (pred_labels == labels)
        
        # apply mask to leave only non-masked correct predictions
        masked_correct: TensorType["B", "T", "P"] = torch.logical_and(prop_mask.bool(), correct_preds)
        
        num_correct = torch.sum(masked_correct)
        num_total = torch.sum(prop_mask)
        acc = num_correct / num_total

    else:
        raise ValueError(f"Unknown acc eval type {eval_type}")

    # return num correct and num total as these are used to calculate global metrics
    # and not just batch level averages
    return acc, num_correct, num_total


class MultiClassCELoss(ScoreLoss):
    """
    Multi-class cross-entropy loss using softmax over labels. Assuming probabilities are mapped
    to discrete labels (T/F/?).
    """
    def __init__(self, config):
        super(MultiClassCELoss, self).__init__(config)
        assert(config.class_output), "MultiClassCELoss must be used with --class_output flag!"

        # for backward compatibility when loading old configs where default was bce acc eval
        # https://github.com/yakazimir/situation_modeling/issues/62
        if not config.acc_eval_type in ["match_rows","by_prop"]:
            util_logger.warn(
                f"Setting invalid config.acc_eval_type from {config.acc_eval_type} to 'by_prop' for use with MultiClassCELoss."
            )
            config.acc_eval_type = "by_prop"
        assert(config.acc_eval_type in ["match_rows","by_prop"]),\
          f"MultiClassCELoss must be used with --config.acc_eval_type in ['match_rows','by_prop'], is {config.acc_eval_type}!"
        
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        """The main forward method
        """
        prop_mask: TensorType["B", "T", "P"] = features["prop_mask"]
        scores: TensorType["B", "T", "P", "O"] = features["scores"]
        B, T, P, O = scores.shape

        # softmax
        pred_probs: TensorType["B", "T", "P", "O"] = scores.softmax(dim=-1)

        NP = int(B*T*P)
        scores_flat: TensorType["NP","O"] = scores.view(-1).reshape(NP,O)
        
        pred_labels: TensorType["B", "T", "P"] = scores.argmax(dim=-1)

        loss = None
        acc  = None

        if labels is not None:
            labels_long = labels.to(torch.long)
            labels_flat = labels_long.view(-1)
            
            # mask loss
            unmasked_loss = self.loss_fn(scores_flat,labels_flat)
            masked_loss = unmasked_loss.reshape(B, T, P) * prop_mask

            #loss = torch.mean(masked_loss)
            psum = prop_mask.sum()

            if torch.is_nonzero(psum): 
                loss = masked_loss.sum() / prop_mask.sum()
            else:
                loss = torch.mean(masked_loss)*0.
            
            ## get an accuracy measurement (moved to global function above)
            ## important: only want to compute if we have `labels`
            acc, n_correct, n_total = _run_eval(
                pred_labels,
                labels_long,
                prop_mask,
                eval_type=self.config.acc_eval_type
            )

        ## items to carry over for printing (if provided) 
        print_out = {} if "print_out" not in features else features["print_out"]
        data_stats = {} if "data_stats" not in features else features["data_stats"]

        output_dict = {
            "loss"      : loss,
            "metrics"   : {
                f"{self.config.acc_eval_type}_acc"     : acc,
                f"{self.config.acc_eval_type}_correct" : n_correct,
                f"{self.config.acc_eval_type}_total"   : n_total,
            },
            "outputs"   : pred_labels, 
            "print_out" : print_out,
            "labels"    : labels,
            "pred_probs": pred_probs,
            "data_stats": data_stats,
            "prop_mask": prop_mask
        }
        return output_dict

    @classmethod
    def from_config(cls,config):
        return cls(config)
    
    
class JointMultiCEGenerationLoss(MultiClassCELoss):
    """Loss function for training a multi-task situation classifier and 
    generation model.
    """

    def __init__(self,config):
        super().__init__(config)
        if config.class_param + config.gen_param != 1.0:
            raise ValueError(
                'Interpolation parameters do not sum to 1.0!'
            )


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        """The main forward method
        """
        ### encodings for text inputs and propositions
        out_dict = super().forward(features,labels)
    
        ### add interpolated loss 
        if (labels is not None) and (self.config.gen_param > 0.0) and ("gen_loss" in features):
            orig_loss = out_dict["loss"]

            out_dict["full_losses"] = {}
            out_dict["full_losses"]["sloss"] = orig_loss
            out_dict["full_losses"]["gloss"] = features["gen_loss"]

            interp_loss = (self.config.class_param*orig_loss) + (self.config.gen_param*features["gen_loss"])
            out_dict["loss"] = interp_loss

        return out_dict

class LogicalRelaxationLoss(MultiClassCELoss):
    """Multi-task loss that supports generation (cross-entropy) loss, joint proposition cross-entropy and soft logic constraint loss
    """

    def __init__(self, config):
        """Initialiates loss function that can contain an additional logical relaxation term 

        :param config: the global configuration
        """
        super(MultiClassCELoss, self).__init__(config)
        assert(config.class_output),\
          "MultiClassCELoss must be used with --class_output flag!"

        # for backward compatibility when loading old configs where default was bce acc eval
        # https://github.com/yakazimir/situation_modeling/issues/62
        if not config.acc_eval_type in ["match_rows","by_prop"]:
            util_logger.warn(
                f"Setting invalid config.acc_eval_type from {config.acc_eval_type} to 'by_prop' for use with MultiClassCELoss."
            )
            config.acc_eval_type = "by_prop"

        assert(config.acc_eval_type in ["match_rows","by_prop"]),\
          f"LogicalRelaxationLoss must be used with --config.acc_eval_type in ['match_rows','by_prop'], is {config.acc_eval_type}!"

        ### uses nllloss and manual log softmax 
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.logic_weight = Variable(
            torch.ones(1)*config.logic_loss_weight,
            requires_grad=False
        )
        self.gen_weight = Variable(
            torch.ones(1)*config.gen_param,
            requires_grad=False
        )
        self.main_weight = Variable(
            torch.ones(1)*config.main_loss_weight,
            requires_grad=False
        )

        ## tnorm stuff 
        self.prod_tnorm         = config.prod_norm
        self.consistency_warmup = config.consistency_warmup
        self.generation_warmup  = config.generation_warmup
        self.main_warmup        = config.main_warmup

        self.excluded_rules = set([r.strip() for r in config.exclude_rules.split(';')])
        
        self.logger.info(
            f'main loss weight: {config.main_loss_weight}, prod-norm={self.prod_tnorm}, logic loss: {config.logic_loss_weight}, excluded rules: {str(self.excluded_rules)}, consistency_warmup: {self.consistency_warmup}, gen weight={self.gen_weight}'
        )
        if self.config.uncertainty_loss:
            self.log_vars = torch.nn.Parameter(torch.zeros((4)))

    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        """The main forward method
        """
        prop_mask: TensorType["B", "T", "P"] = features["prop_mask"]
        scores: TensorType["B", "T", "P", "O"] = features["scores"]
        B, T, P, O = scores.shape

        # softmax
        pred_probs: TensorType["B", "T", "P", "O"] = scores.softmax(dim=-1)

        NP = int(B*T*P)
        scores_flat: TensorType["NP","O"] = scores.view(-1).reshape(NP,O)
        log_scores: TensorType["NP","O"] = scores_flat.log_softmax(dim=-1)

        pred_labels: TensorType["B", "T", "P"] = scores.argmax(dim=-1)

        loss = None
        acc  = None
        batch_violations = None
        full_losses = {}
        
        ## try to reduce logging for training, still having memory issues
        metrics = features["metrics"] if "metrics" in features else {}

        if labels is not None:
            meta = {} if "meta" not in features else features["meta"]
            curr_epoch = 10000000 if "epoch" not in meta else meta["epoch"]

            labels_long = labels.to(torch.long)
            labels_flat = labels_long.view(-1)

            ## get an accuracy measurement (moved to global function above)
            ## important: only want to compute if we have `labels`
            acc, n_correct, n_total = _run_eval(
                pred_labels,
                labels_long,
                prop_mask,
                eval_type=self.config.acc_eval_type
            )

            # main CE loss 
            unmasked_loss = self.loss_fn(log_scores,labels_flat)
            masked_loss = unmasked_loss.reshape(B, T, P) * prop_mask
            #main_loss = torch.mean(masked_loss) #torch.mean(masked_loss) #<--- changed this to sum

            ### does that prop_mask.sum is non-zero

            ### label dropout: randomly mask %x of labels
            #torch.rand(100).uniform_() <= 0.9)*1.
            if self.config.label_dropout > 0. and features["evaluate"] is False:
                amount = 1. - self.config.label_dropout
                if "epoch" in meta:
                    cutoff = 1. if curr_epoch == 0 \
                      else 1. - (curr_epoch / self.config.num_train_epochs)
                    amount = amount*cutoff

                ldrop = ((torch.randn((B,T,P)).uniform_() <= amount)*1.).to(log_scores.device)
                prop_mask = prop_mask*ldrop

                metrics["label_dropout"] = torch.tensor([amount])

            ### computation of main loss 
            if self.config.loss_aggr == "mean":
                psum = prop_mask.sum()
                if torch.is_nonzero(psum): 
                    main_loss = masked_loss.sum() / psum #prop_mask.sum()
                else:
                    main_loss = torch.mean(masked_loss)*0.
            elif self.config.loss_aggr == "sum":
                main_loss = torch.mean(masked_loss)
            else:
                raise ValueError(
                    f'Unknown loss aggreegatin type: {self.config.loss_aggr}'
                )
                
            full_losses["sloss"] = main_loss
            gen_probs = None if "gen_probs" not in features else features["gen_probs"]

            ### v2 with optional model counting
            if self.config.logic_type == "wmc" and curr_epoch < self.consistency_warmup:
                constraint_loss = torch.zeros(1).to(log_scores.device)
                total_constraints = 0
                violations = 0.
                global_violations = 0
            else: 
                constraint_loss,\
                total_constraints,\
                batch_violations,\
                global_violation = \
                compute_constraint_loss(
                    features["constraints"],
                    pred_probs,
                    log_scores,
                    pred_labels,
                    self.excluded_rules,
                    self.config,
                    gen_probs
                )

            # generation loss (computed elsewhere using the base model)
            gen_loss = features["gen_loss"] \
              if "gen_loss" in features \
              else torch.zeros(1).to(log_scores.device)

            #################
            ### loss weight terms 
            gen_weight = self.gen_weight.to(log_scores.device) \
              if (curr_epoch >= self.generation_warmup and "gen_loss" in features) \
              else torch.zeros(1).to(log_scores.device)
              
            ### turn off the gen loss after a certain number of epochs (special case, should generalize to other losses)
            if curr_epoch >= self.config.gen_max:
                gen_weight = (torch.zeros(1)*self.config.gen_min).to(log_scores.device)

            logic_weight = self.logic_weight.to(log_scores.device) \
              if (curr_epoch >= self.consistency_warmup and total_constraints > 0) \
              else torch.zeros(1).to(log_scores.device)

            main_weight = self.main_weight.to(log_scores.device) \
              if (curr_epoch >= self.main_warmup) \
              else torch.zeros(1).to(log_scores.device)

            ### qa loss
            qa_loss = features["qa_loss"] if "qa_loss" in features \
              else torch.zeros(1).to(log_scores.device)
            qa_weight = (torch.ones(1)*self.config.qa_loss_weight).to(log_scores.device) \
              if (curr_epoch >= self.config.qa_warmup and "qa_loss" in features) \
              else torch.zeros(1).to(log_scores.device)

            ### weighting uses uncertainty
            ## see: https://github.com/Hui-Li/multi-task-learning-example-PyTorch
            ## https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681/12
            if self.config.uncertainty_loss:
                log_vars = self.log_vars.to(log_scores.device)

                precision0 = torch.exp(-log_vars[0])
                precision1 = torch.exp(-log_vars[1])
                precision2 = torch.exp(-log_vars[2])
                precision3 = torch.exp(-log_vars[3])

                loss1 = (precision0*main_loss + log_vars[0]) \
                  if torch.is_nonzero(main_weight) \
                  else main_weight
                loss2 = (precision1*constraint_loss + log_vars[1]) \
                  if torch.is_nonzero(logic_weight) and torch.is_nonzero(constraint_loss) \
                  else logic_weight
                loss3 = (precision2*qa_loss + log_vars[2]) \
                  if torch.is_nonzero(qa_weight) \
                  else qa_weight
                loss4 = (precision3*gen_loss + log_vars[3]) \
                  if torch.is_nonzero(gen_weight) \
                  else gen_weight

                zero_tensor = torch.zeros(1).to(log_scores.device)
                if loss3 < 0.: 
                    loss3 = torch.max(zero_tensor,loss3)
                if loss2 < 0.: 
                    loss2 = torch.max(zero_tensor,loss2)

                #loss = loss1+loss2+loss3+loss4
                #loss = torch.max(zero_tensor,loss1+loss2+loss3+loss4)
                ## somtimes goes negative: see https://stackoverflow.com/questions/68806330/negative-loss-when-trying-to-implement-aleatoric-uncertainty-estimation-accordin
                loss = loss1+loss2+loss3+loss4

            ### manual weighting 
            else: 
                ### aggregation
                loss = (main_weight*main_loss)+\
                  (gen_weight*gen_loss)+\
                  (logic_weight*constraint_loss)+\
                  (qa_weight*qa_loss)


            ### log
            if "gen_loss" in features and (torch.is_nonzero(gen_weight) or features["evaluate"]):
                full_losses["gloss"] = features["gen_loss"]
            if total_constraints > 0 and (torch.is_nonzero(logic_weight) or self.config.log_all):
                full_losses["logic_loss"] = constraint_loss
            if "qa_loss" in features and (torch.is_nonzero(qa_weight) or features["evaluate"] is True):
                full_losses["qa_loss"] = features["qa_loss"]
                                              
        ## items to carry over for printing (if provided) 3
        print_out  = {} if "print_out" not in features else features["print_out"]
        data_stats = {} if "data_stats" not in features else features["data_stats"]

        
        if features["evaluate"] is True:
            metrics.update({
                f"{self.config.acc_eval_type}_acc"     : acc,
                f"{self.config.acc_eval_type}_correct" : n_correct,
                f"{self.config.acc_eval_type}_total"   : n_total
            })
            
            output_dict = {
                "loss"       : loss,
                "metrics"    : metrics,
                "outputs"    : pred_labels, 
                "print_out"  : print_out,
                "labels"     : labels,
                "pred_probs" : pred_probs,
                "data_stats" : data_stats,
                "full_losses": full_losses,
            }
            if gen_probs is not None:
                output_dict["gen_probs"] = gen_probs
            if batch_violations is not None:
                output_dict["metrics"][f"global_constraint_violations"] = torch.tensor(batch_violations).float()
                output_dict["metrics"][f"rho"] = torch.tensor(global_violation).float()
        else:
            metrics.update({f"{self.config.acc_eval_type}_acc" : acc})
            
            output_dict = {
                "loss"       : loss,
                "metrics"    : metrics,
                "full_losses": full_losses,
            }
            if batch_violations is not None:
                output_dict["metrics"][f"global_constraint_violations"] = torch.tensor(batch_violations).float()
                output_dict["metrics"][f"rho"] = torch.tensor(global_violation).float()

        return output_dict

    # def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
    #     """The main forward method
    #     """
    #     prop_mask: TensorType["B", "T", "P"] = features["prop_mask"]
    #     scores: TensorType["B", "T", "P", "O"] = features["scores"]
    #     B, T, P, O = scores.shape

    #     # softmax
    #     pred_probs: TensorType["B", "T", "P", "O"] = scores.softmax(dim=-1)

    #     NP = int(B*T*P)
    #     scores_flat: TensorType["NP","O"] = scores.view(-1).reshape(NP,O)
    #     log_scores: TensorType["NP","O"] = scores_flat.log_softmax(dim=-1)

    #     pred_labels: TensorType["B", "T", "P"] = scores.argmax(dim=-1)

    #     loss = None
    #     acc  = None
    #     batch_violations = None
    #     full_losses = {}

    #     if labels is not None:

    #         labels_long = labels.to(torch.long)
    #         labels_flat = labels_long.view(-1)

    #         ## get an accuracy measurement (moved to global function above)
    #         ## important: only want to compute if we have `labels`
    #         acc, n_correct, n_total = _run_eval(
    #             pred_labels,
    #             labels_long,
    #             prop_mask,
    #             eval_type=self.config.acc_eval_type
    #         )

    #         # mask loss
    #         unmasked_loss = self.loss_fn(log_scores,labels_flat)
    #         masked_loss = unmasked_loss.reshape(B, T, P) * prop_mask
    #         main_loss = torch.mean(masked_loss) #torch.mean(masked_loss) #<--- changed this to sum

    #         full_losses["sloss"] = main_loss

    #         ### second attempt, using ideas and tricks from: https://github.com/utahnlp/neural-logic
    #         constraint_loss = torch.zeros(1).to(log_scores.device)
    #         total_constraints = 0.0
    #         batch_violations = 0.0
    #         actionable_loss = 0

    #         ### constraints
    #         meta = {} if "meta" not in features else features["meta"]
    #         curr_epoch = 10000000 if "epoch" not in meta else meta["epoch"]
    #         global_violation = 0.
                
    #         for (name,operator,left,right) in features["constraints"]:
    #             if name in self.excluded_rules: continue

    #             ### implication
    #             if operator == "implication" or operator == "biconditional":
    #                 left_score = torch.ones(1).to(log_scores.device)
    #                 right_score = torch.ones(1).to(log_scores.device)

    #                 left_bools  = [None]*len(left)
    #                 right_bools = [None]*len(right)

    #                 for w,(l_t,l_i,l_b,l_label) in enumerate(left):
    #                     left_score *= pred_probs[l_b][l_t][l_i][l_label]
    #                     left_bools[w] = (pred_labels[l_b][l_t][l_i].detach().item() == l_label)
                        
    #                 for z,(r_t,r_i,r_b,r_label) in enumerate(right):
    #                     right_score *= pred_probs[r_b][r_t][r_i][r_label]
    #                     right_bools[z] = (pred_labels[r_b][r_t][r_i].detach().item() == r_label)

    #                 ### compute symbolically whether constraint is satisfied
    #                 matched = (not all(left_bools)) or all(right_bools)

    #                 ### debugging 
    #                 # print(name)
    #                 # print(operator)
    #                 # print([(features["print_out"]["prop_lists"][l_b][l_t][l_i],l_label) for (l_t,l_i,l_b,l_label) in left])
    #                 # print([(features["print_out"]["prop_lists"][r_b][r_t][r_i],r_label) for (r_t,r_i,r_b,r_label) in right])
    #                 # print("----------------")

    #                 ### `s_prod` t-norm
    #                 if self.prod_tnorm == "s_prod":
    #                     elements = (1 - left_score + left_score*right_score) + 0.00001
    #                     if operator == "biconditional":
    #                         elements *= (1 - right_score + left_score*right_score) + 0.00001
    #                         matched = matched and ((not all(right_bools)) or all(left_bools))
    #                 else:
    #                     division   = right_score / (left_score+0.001)
    #                     ones_tensor = torch.ones(division.shape).to(log_scores.device)
    #                     elements = torch.min(ones_tensor,division)
                        
    #                     if operator == "biconditional":
    #                         division2 = left_score / (right_score+0.001)
    #                         elements *= torch.min(ones_tensor,division2)
    #                         matched = matched and ((not all(right_bools)) or all(left_bools))

    #                 ## sum not strictly needed here
    #                 constraint_loss += -1*torch.sum(torch.log(elements))
    #                 total_constraints += 1.

    #                 ### compute constraint symnolically
    #                 if matched is False:
    #                     batch_violations += 1.
    #                     global_violation = 1.

    #             else:
    #                 raise ValueError(
    #                     f'Unknown or unsupported operator: {operator}'
    #                 )
    #         if total_constraints > 0.:
    #             constraint_loss = self.logic_weight.to(log_scores.device)*constraint_loss
    #             full_losses["logic_loss"] = constraint_loss
    #             batch_violations = batch_violations / total_constraints

    #         ### joint loss computation
    #         #loss = main_loss
    #         if "gen_loss" in features:
    #             full_losses["gloss"] = features["gen_loss"]
                
    #         if not self.no_constraint_loss and curr_epoch >= self.consistency_warmup:
    #             if "gen_loss" in features and curr_epoch >= self.generation_warmup: 
    #                 loss = (self.main_weight.to(log_scores.device)*main_loss)+\
    #                   (self.logic_weight.to(log_scores.device)*constraint_loss)+\
    #                   (self.gen_weight.to(log_scores.device)*features["gen_loss"])
    #             else:
    #                 loss = (self.main_weight.to(log_scores.device)*main_loss)+\
    #                   (self.logic_weight.to(log_scores.device)*constraint_loss)
    #         else:
    #             if "gen_loss" in features and curr_epoch >= self.generation_warmup:
    #                 loss = (self.main_weight.to(log_scores.device)*main_loss)+\
    #                   (self.gen_weight.to(log_scores.device)*features["gen_loss"])
    #             else: 
    #                 loss = main_loss


            
    #         # loss = (self.main_weight.to(log_scores.device)*main_loss)+\
    #         #   (self.logic_weight.to(log_scores.device)*constraint_loss)+\
    #         ## generation loss
    #         # if (labels is not None) and "gen_loss" in features: # and curr_epoch >= self.generation_warmup:
    #         #     full_losses["gloss"] = features["gen_loss"]
    #         #     loss += (self.gen_weight.to(log_scores.device)*features["gen_loss"])

    #     ## items to carry over for printing (if provided) 
    #     print_out = {} if "print_out" not in features else features["print_out"]
    #     data_stats = {} if "data_stats" not in features else features["data_stats"]
    #     #meta = {} if "meta" not in features else features["meta"]

    #     output_dict = {
    #         "loss"      : loss,
    #         "metrics"   : {
    #             f"{self.config.acc_eval_type}_acc" : acc,
    #             f"{self.config.acc_eval_type}_correct" : n_correct,
    #             f"{self.config.acc_eval_type}_total" : n_total
    #         },
    #         "outputs"   : pred_labels, 
    #         "print_out" : print_out,
    #         "labels"    : labels,
    #         "pred_probs": pred_probs,
    #         "data_stats": data_stats,
    #         "full_losses": full_losses,
    #     }
    #     if batch_violations is not None:
    #         output_dict["metrics"][f"global_constraint_violations"] = torch.tensor(batch_violations).float()
    #         output_dict["metrics"][f"rho"] = torch.tensor(global_violation).float()

    #         # output_dict["metrics"][f"ratio_informative_losses"] = torch.tensor(
    #         #     actionable_loss/len(features["constraints"])).float()

    #     return output_dict
