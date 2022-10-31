import torch
from pysdd.sdd import SddManager, Vtree, WmcManager


def compute_soft_logic_loss(
        constraints,
        pred_probs,
        log_scores,
        pred_labels,
        aggr_type,
        prod_tnorm='r_prod',
        excluded_rules=set(),
        gen_probs=None,
    ):
    """Computes soft logic losses 

    :param constraints: the list of constraints 
    :param pred_probs: predictive probabilities 
    :type constraints: list of tuples 

    :notes: adapted from: https://github.com/utahnlp/neural-logic/blob/634b676f86c751361e96e077c67bc21f67c3189e/chunking/loss.py#L108
    """
    constraint_loss = torch.zeros(1).to(log_scores.device)
    total_to_count = torch.zeros(1).to(log_scores.device)
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

            l = -1*torch.sum(torch.log(elements))
            constraint_loss += l
            total_constraints += 1.
            if torch.is_nonzero(l):
                total_to_count += 1.

            if matched is False:
                batch_violations += 1.
                global_violation = 1.
                

        else:
            raise ValueError(
                f'Operation not implemented: {operator}'
            )

        # #elif operator == "disjunction":
        # #    pass
    if aggr_type == "mean" and torch.is_nonzero(total_to_count): 
        constraint_loss = constraint_loss / total_to_count
        
    violations = batch_violations / total_constraints if total_constraints > 0. else None
    return (constraint_loss,total_constraints,violations,global_violation)

POL_MAP = {
    0 : 0,
    1 : 2,
}

def get_probs(sdd,lit_probs,rmap,gen_probs=None):
    """Formula for turning circuit representation into pytorch loss 

    :see: https://github.com/pylon-lib/pylon/blob/master/pylon/circuit_solver.py and 
        https://github.com/lucadiliello/semantic-loss-pytorch/blob/0afa10791f878e8bb7e5cb7fd99a3a1dc0ec2679/semantic_loss_pytorch/py3psdd/sdd.py

    :Param sdd: the compiled sdd representation 
    :param lit_probs: the softmax probabilities associated with model prediction 
    :param rmap: pointer from variables to locations in lit_probs
    """
    if sdd.is_false():
        return 0.0
    elif sdd.is_true():
        return 1.0   
    elif sdd.is_literal() and sdd.literal > 0:
        if len(rmap[sdd.literal]) == 4:
            return lit_probs[rmap[sdd.literal][2]]\
              [rmap[sdd.literal][0]]\
              [rmap[sdd.literal][1]]\
              [rmap[sdd.literal][-1]]

        ### generation 
        return gen_probs[rmap[sdd.literal][0]]
    elif sdd.is_literal() and sdd.literal < 0:
        ## same as above
        if len(rmap[-sdd.literal]) == 4:
            return 1.0 - lit_probs[rmap[-sdd.literal][2]]\
              [rmap[-sdd.literal][0]]\
              [rmap[-sdd.literal][1]]\
              [rmap[-sdd.literal][-1]]

        ### generation 
        return 1.0 - gen_probs[rmap[-sdd.literal][0]]
    elif sdd.is_decision():
        p = 0.0
        for prime, sub in sdd.elements():
            p += get_probs(prime, lit_probs,rmap,gen_probs) * get_probs(sub, lit_probs,rmap,gen_probs)
        return p

def _count_models(
        formula,
        var_map,
        reverse_map,
        probs,
        pred_labels,
        gen_probs
    ):
    """Runs the main model counting algorithm. Starts by 
    constructing a CNF formula then weightning propositions 
    by model probabilities. 
    
    :param formula: CNF formula 
    :var_map: the map from variables in formula to pointers in prob
    :param reverse_map: a reversed version of map
    :param probs: the model label probabilities 
    :param pred_labels: The predicted labels for computing symbolic constraint score
    """

    num_vars = len(var_map)

    mgr = SddManager(var_count=num_vars)
    cnf_formula = None
    global_violation = 0
    total_violations = 0

    ### create base cnf formula
    for (name,clause) in formula:
        new_formula = None

        ### check if clause is satisfied 
        # satisfies = any([
        #     pred_labels[reverse_map[v][2]]\
        #     [reverse_map[v][0]]\
        #     [reverse_map[v][1]].detach().item() == POL_MAP[p] \
        #     for p,v in clause
        # ])
        # if satisfies is False:
        #     global_violation = 1
        #     total_violations += 1

        for j in range(0,len(clause),2):
            next_items = [mgr.literal(i) if p == 1 else -mgr.literal(i) for p,i in clause[j:j+2]]
            if len(next_items) == 1:
                new_formula = mgr.disjoin(new_formula,next_items[0])
            else: 
                nd = mgr.disjoin(*next_items)
                new_formula = nd if new_formula is None else mgr.disjoin(new_formula,nd)

        cnf_formula = new_formula \
          if cnf_formula is None \
          else mgr.conjoin(cnf_formula,new_formula)

    ### weight the formula
    #wmc = cnf_formula.wmc(log_mode=False)
    ### ordinary wmc disattached from pytorch 
    # for (tid,pid,bid,label),var_id in var_map.items():
    #     prob = probs[bid][tid][pid][label].item()
    #     wmc.set_literal_weight(mgr.literal(var_id),prob)
    #     wmc.set_literal_weight(-mgr.literal(var_id),1.0 - prob)
    # p = torch.tensor(wmc.propagate(),requires_grad=True)
    p = get_probs(cnf_formula,probs,reverse_map,gen_probs)

    return (p,global_violation,total_violations)


def count_models(
        constraints,
        pred_probs,
        log_scores,
        pred_labels,
        aggr_type,
        excluded_rules=set(),
        gen_probs=None,
    ):
    """Does weighted model counting on a target CNF formula 


    :param constraints: constraint as CNF 
    :param pred_probs: the model predicted probabilities 
    :param log_scores: if needed, log probability scores 
    :param aggr_type: the type of aggregation to do when computing final loss
    :param excluded_rules: rules to exclude 
    """
    closses = {}
    global_violations = 0
    batch_violations = 0
    total_constraints = 0

    closs = torch.zeros(1).to(log_scores.device)
    total = torch.zeros(1).to(log_scores.device)

    for b,(cnf_formula,var_map) in enumerate(constraints.constraint_formula):
        formulas = [(name,f) for (name,f) in cnf_formula if name not in excluded_rules]
        reverse_map = {v:k for k,v in var_map.items()}

        ### exclude/ablate rules if desired 
        if not formulas: continue

        constraint_prob,single_violation,violations = _count_models(
            formulas,
            var_map,
            reverse_map,
            pred_probs,
            pred_labels,
            gen_probs
        )
        
        closs += -1*torch.log(constraint_prob)
        total += 1.

        total_constraints += len(formulas)
        global_violations = 1. if single_violation == 1. else global_violations
        batch_violations += violations

    # if closses:
    #     probs = torch.tensor([v for v in closses.values()]).to(log_scores.device)
    #     closs = -torch.log(torch.mean(probs)).to(log_scores.device)
    if aggr_type == "mean" and total_constraints > 0.: 
        closs = closs / total
        
    ###
    violations = batch_violations / total_constraints if total_constraints > 0. else None
    return (closs,total_constraints,violations,global_violations)

        

def compute_constraint_loss(
        constraints,
        pred_probs,
        log_scores,
        pred_labels,
        excluded_rules,
        config,
        gen_probs
    ):
    """Computes a loss over the provided constraints using either 
    fuzzy or weighted model counting semantics 

    :param constraints: the list of target constraints 
    :param pred_probs: the probabilities associated with prediction 
    :param log_scores: the log probability scores 
    :param pred_labels: the label predictions made
    :param config: the global configuration 
    """
    if config.logic_type == "fuzzy":
        return compute_soft_logic_loss(
            constraints,
            pred_probs,
            log_scores,
            pred_labels,
            config.logic_aggr,
            config.prod_norm,
            excluded_rules,
            gen_probs=gen_probs,
        )
    elif config.logic_type == "wmc":
        return count_models(
            constraints,
            pred_probs,
            log_scores,
            pred_labels,
            config.logic_aggr,
            excluded_rules,
            gen_probs=gen_probs
        )
    raise ValueError(
        f"Unknown logic type: {config.logic_type}"
    )
