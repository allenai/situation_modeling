#####
import torch

def compute_logic_loss(
        constraints,
        pred_probs,
        log_scores,
        pred_labels,
        prod_tnorm='r_prod',
        excluded_rules=set(),
    ):
    """Computes soft logic losses 

    :param constraints: the list of constraints 
    :param pred_probs: predictive probabilities 
    :type constraints: list of tuples 

    :notes: adapted from: https://github.com/utahnlp/neural-logic/blob/634b676f86c751361e96e077c67bc21f67c3189e/chunking/loss.py#L108
    """
    constraint_loss = torch.zeros(1).to(log_scores.device)
    total_constraints = 0.0

    batch_violations = 0.0
    global_violation = 0

    for (name,operator,left,right) in constraints:
        if name in excluded_rules: continue

        if operator == "implication" or operator == "biconditional":

            left_score = torch.ones(1).to(log_scores.device)
            right_score = torch.ones(1).to(log_scores.device)

            left_bools  = [None]*len(left)
            right_bools = [None]*len(right)

            for w,(l_t,l_i,l_b,l_label) in enumerate(left):
                left_score *= pred_probs[l_b][l_t][l_i][l_label]
                left_bools[w] = (pred_labels[l_b][l_t][l_i].detach().item() == l_label)
            for z,(r_t,r_i,r_b,r_label) in enumerate(right):
                right_score *= pred_probs[r_b][r_t][r_i][r_label]
                right_bools[z] = (pred_labels[r_b][r_t][r_i].detach().item() == r_label)

            ## check symbolically is constraint is satisfied 
            matched = (not all(left_bools)) or all(right_bools)

            if prod_tnorm == "s_prod":
                elements = (1 - left_score + left_score*right_score) + 0.00001
                if operator == "biconditional":
                    elements *= (1 - right_score + left_score*right_score) + 0.00001
                    matched = matched and ((not all(right_bools)) or all(left_bools))
            else:
                division   = right_score / (left_score+0.001)
                ones_tensor = torch.ones(division.shape).to(log_scores.device)
                elements = torch.min(ones_tensor,division)
                        
                if operator == "biconditional":
                    division2 = left_score / (right_score+0.001)
                    elements *= torch.min(ones_tensor,division2)
                    matched = matched and ((not all(right_bools)) or all(left_bools))

            constraint_loss += -1*torch.sum(torch.log(elements))
            total_constraints += 1.

            if matched is False:
                batch_violations += 1.
                global_violation = 1.
                
        else:
            raise ValueError(
                f'Operation not implemented: {operator}'
            )

        # #elif operator == "disjunction":
        # #    pass

    violations = batch_violations / total_constraints if total_constraints > 0. else None
    return (constraint_loss,total_constraints,violations,global_violation)
