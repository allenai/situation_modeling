import os
import sys
import re
import json
import logging
from collections import defaultdict
import itertools
import numpy as np
from ..readers.constraints import parse_basic_implications_conjunctions

util_logger = logging.getLogger('situation_modeling.utils.model_analysis')

orig_map = {
        "2" : 1.0,
        "0" : 0.0,
        "1" : 0.5,
}
output_map = {
    "yes"   : "2",
    "no"    : "0",
    "maybe" : "1",
    1.    : "2",
    0.    : "0",
    0.5   : "1"
}

def text_to_text_analysis(config):
    metrics = {}
    for split,out_file,fname in [
            ("dev",f"{config.dev_name}_eval.jsonl",config.dev_name),
            ("test",f"{config.test_name}_eval.jsonl",config.test_name),
        ]:
        has_texts = False
        orig_list = []
        orig_path = os.path.join(config.data_dir,f"{fname}.jsonl")
        correct_length_gen = defaultdict(int)
        correct_length_gen_totals = defaultdict(int)
        total = 0.
        correct = 0.

        if not os.path.isfile(orig_path):
            util_logger.warning(f'Cannot find corresponding original file: {orig_path}')
        else:
            with open(orig_path) as orig_data:
                for line in orig_data:
                    line = line.strip()
                    json_line = json.loads(line)
                    if "texts" in json_line: has_texts = True 
                    orig_list.append(json_line)

        full_path = os.path.join(config.output_dir,out_file)
        if not os.path.isfile(full_path):
            util_logger.warning(f'Attempting to check output, file not found: {full_path}')
            continue

        ### no analysis to do 
        if has_texts is False:
            util_logger.warning(f'No story problems founding, return nothing ')
            return {}

        ###
        with open(full_path) as output_data:
            for k,line in enumerate(output_data):
                line = line.strip()
                json_line = json.loads(line)
                matched = (json_line["answer"].strip() == json_line["gen_out"].strip())
                total += 1.
                if matched: correct += 1 

                orig_json = orig_list[k]
                ### cases with a text
                
                if "texts" in orig_json:
                    story_length = len(orig_json["texts"])
                    correct_length_gen_totals[story_length] += 1.
                    if matched: 
                        correct_length_gen[story_length] += 1.

        
        ###
        metrics[f"{split}_em_acc"] = correct / total if total > 0. else 0.
        if correct_length_gen and len(correct_length_gen) > 1:
            scores = []
            for length,num_correct in correct_length_gen.items():
                #scores.append(num_correct / correct_length_gen_totals[length])
                score = num_correct / correct_length_gen_totals[length] if  num_correct > 0 else 0.
                metrics[f"{split}_gen_{length}_acc"] = score
                scores.append(score)
            metrics[f"{split}_gen_avg_acc"] = np.mean(scores)
            
    return metrics

### add probs slot 
    
def load_flat(path,orig_list,config,metrics,split):
    ### transformers sentence transformer files into ordinary situation state format
    new_template = []
    for instance in orig_list:
        new_json = {}
        new_json["guid"]   = instance["id"] if "id" in instance else instance["guid"]
        new_json["texts"]  = ' '.join(instance["texts"])
        new_json["events"] = [
            [
            t,
            [p for p in instance["prop_lists"][k]],
            [None          for l in instance["outputs"][k]],
            [output_map[l] for l in instance["outputs"][k]]]
            for k,t in enumerate(instance["texts"])
            ]
        new_template.append(new_json)
        
    original_total   = 0.
    original_correct = 0.
    with open(path) as output:
        for line in output:
            json_line = json.loads(line)
            identifier = json_line["guid"] 
            
            story = json_line["story"]
            story_fragment = [s.replace(".","") for s in story.split(". ")]
            ### pointers to original 
            _,n,_,t,_,p = identifier.split("_")
            n = int(n); t = int(t); p = int(p)
            prop = json_line["proposition"]
            predicted = output_map[json_line["predicted"]]
            gold      = output_map[json_line["gold"]]

            original_total += 1.
            if predicted == gold:
                original_correct += 1.
            # ### check that text matches
            # ### check for hash
            # has_hash = [j for j,i in enumerate(story_fragment) if "#" in i]
            # if hash_has:
            #     assert len(has_hash) == 1
            #assert story_fragment == orig_list[n]["texts"][:t+1],(story_fragment,orig_list[n]["texts"][:t+1])

            new_output = new_template[n]
            
            ## check that propositions align 
            assert prop == new_output["events"][t][1][p]
            new_template[n]["events"][t][2][p] = predicted
            assert new_output["events"][t][3][p] == gold


    raw_score = original_correct / original_total if original_total > 0. else 0.
    metrics[f"raw_{split}_acc"] = raw_score
    metrics[f"raw_{split}_total"] = original_total
    ### print out
    new_format_copy = os.path.join(config.output_dir,f"{config.dev_name}_eval_reformat.jsonl")
    with open(new_format_copy,'w') as copy_out:
        for item in new_template:
            copy_out.write(json.dumps(item))
            copy_out.write('\n')
    return new_template


def situation_analysis(config,convert=False):
    """Does post-hoc analysis on the situation models 

    :param config: the global configuration 
    :param convert: convert a flat prediction model file 
    """
    metrics = {}
    excluded_rules = set([r.strip() for r in config.exclude_rules.split(';')])

    ### check for output files and does additional analysis 
    for split,out_file,fname in [
            ("dev",f"{config.dev_name}_eval.jsonl",config.dev_name),
            ("test",f"{config.test_name}_eval.jsonl",config.test_name),
        ]:

        ### original file
        orig_list = []
        orig_path = os.path.join(config.data_dir,f"{fname}.jsonl")
        if not os.path.isfile(orig_path):
            util_logger.warning(f'Cannot find corresponding original file: {orig_path}')
        else:
            with open(orig_path) as orig_data:
                for line in orig_data:
                    line = line.strip()
                    json_line = json.loads(line)
                    orig_list.append(json_line)

        ### correspponding results file 
        full_path = os.path.join(config.output_dir,out_file)
        if not os.path.isfile(full_path):
            util_logger.warning(f'Attempting to check output, file not found: {full_path}')
            continue

        len_totals = defaultdict(int)
        len_correct = defaultdict(int)
        len_correct_swap = defaultdict(int)
        total = 0.
        correct = 0.
        swap_correct = 0.
        num_swapped = 0
        good_swaps = 0.
        global_story_consistency = []
        avg_global_story_consistency = []
        avg_constraint_violations = []
        global_has_correct = []
        correct_inferences = 0.
        total_inferences = 0.
        total_generations = 0.
        correct_generations = 0.

        constraint_type_counts = defaultdict(int)
        constraint_type_correct = defaultdict(int)
        correct_length_gen = defaultdict(int)
        correct_length_gen_totals = defaultdict(int)

        ### generation
        gen_correct = 0.
        gen_total = 0.


        ##
        my_data = []
        if convert is False:
            with open(full_path) as output_data:
                for k,line in enumerate(output_data):
                    line = line.strip()
                    json_line = json.loads(line)
                    my_data.append(json_line)
        else:
            my_data = load_flat(full_path,orig_list,config,metrics,split)
            
        if True: 
            for k,json_line in enumerate(my_data):
                #line = line.strip()
                #json_line = json.loads(line)
                events = json_line["events"]
                prop_lists = [i[1] for i in events]
                story_length = len(events)

                answer_prop = None

                ### generation accuracy (if there)
                if "answers" in json_line and "gen_answers" in json_line:
                    gen_total += 1.
                    answer_prop = json_line["gen_answers"][0]
                    correct_length_gen_totals[story_length] += 1
                    if json_line["answers"][0] == json_line["gen_answers"][0]:
                        gen_correct += 1.
                        correct_length_gen[story_length] += 1.

                ### proposition accuracy, also by story length
                #for event,plist,predicted,gold in events:
                for e in events:
                    event = e[0]
                    plist = e[1]
                    predicted = e[2]
                    gold = e[3]
                    for l,p in enumerate(plist):
                        predicted_label = predicted[l]
                        gold_label = gold[l]
                        total += 1
                        len_totals[story_length] += 1.
                        if predicted_label == gold_label:
                            correct += 1.
                            len_correct[story_length] += 1.

                        swapped_pred = predicted_label
                        if answer_prop and p == answer_prop and predicted_label != 1 and predicted_label != 2:
                            swapped_pred = 2
                            num_swapped += 1.
                            if gold_label == 2:
                                good_swaps += 1.
                                
                        if swapped_pred == gold_label:
                            swap_correct += 1.
                            len_correct_swap[story_length] += 1.

                #### check constraint consistency
                if orig_list:
                    orig_json = orig_list[k]

                    #assert json_line["guid"] == orig_list[k]["id"],"file mismatch"
                    assert prop_lists == orig_json["prop_lists"]
                    gold = [[orig_map[str(v)] for v in e[3]] for e in events]
                    predicted = [[orig_map[str(v)] for v in e[2]] for e in events]
                    flat_gold = list(itertools.chain(*gold))
                    flat_predicted = list(itertools.chain(*predicted))

                    global_consistency = 0
                    wrong_constraint = 0
                    total_constraints = 0
                    has_correct = 0

                    # if hasattr(config,"no_constraint_loss") and config.no_constraint_loss is True:
                    #     raw_constraints = []
                    # else:
                    raw_constraints = orig_json.get("constraints",[])

                    for constraint in parse_basic_implications_conjunctions(raw_constraints):
                        constraint_name = constraint.name
                        #print(constraint_name)
                        constraint_type_counts[constraint_name] += 1

                        #### no skipping over constraints 
                        #if constraint_name == "deriv-loc" or constraint_name == "pronoun":
                        #   continue 
                        #if constraint_name in excluded_rules: pass #continue

                        left = constraint.arg1
                        right = constraint.arg2
                        ctype = constraint.operator_type
                        total_constraints += 1

                        left_results = []
                        right_results = []
                        left_text  = []
                        right_text = []
                        left_gold  = []
                        right_gold = []

                        for prop in left:
                            label = 0. if "~" in prop else 1.
                            pointer = prop.replace("~","").strip()
                            j,indx = pointer.split("_")
                            j = int(j)
                            indx = int(indx)
                            prediction = predicted[j][indx]
                            
                            left_results.append(prediction == label)
                            left_text.append(prop_lists[j][indx])
                            left_bool = True if label == 1. else False
                            left_gold.append(left_bool)

                        for prop in right:
                            label = 0. if "~" in prop else 1.
                            pointer = prop.replace("~","").strip()
                            j,indx = pointer.split("_")
                            j = int(j)
                            indx = int(indx)
                            prediction = predicted[j][indx]
                            right_results.append(prediction == label)
                            right_text.append(prop_lists[j][indx])
                            right_bool = True if label == 1. else False
                            right_gold.append(right_bool)

                        ###
                        if constraint.name == "proof":
                            total_inferences += 1.
                            if all(left_results) and all(right_results):
                                correct_inferences += 1

                        satisfied = (not all(left_results)) or all(right_results)
                        if ctype == "biconditional":
                            satisfied = satisfied and ((not all(right_results)) or all(left_results))

                        if satisfied is False:
                            global_consistency = 1
                            wrong_constraint += 1
                        else:
                            #if all(left_results):
                            constraint_type_correct[constraint_name] += 1.

                        if ctype == "biconditional" and (all(left_results) and all(right_results)):
                            has_correct = 1
                        elif all(left_gold):
                            has_correct = 1
                    ####
                    if total_constraints > 0: 
                        global_story_consistency.append(global_consistency)
                        avg_constraint_violations.append(
                            wrong_constraint / total_constraints
                        )
                        global_has_correct.append(has_correct)

                        ####                         
                        
                    
        ### log metrics

        metrics[f"{split}_prop_acc"] = correct / total if total > 0. else 0.
        metrics[f"{split}_total"] = total
        if total_inferences > 0.:
            metrics[f"{split}_inference_acc"] = correct_inferences / total_inferences

        ### length accuracy
        length_scores = []
        for length in len_totals:
            score = len_correct[length] / len_totals[length] if len_totals[length] > 0 else 0.
            metrics[f"{split}_{length}_prop_acc"] = score
            length_scores.append(score)
        metrics[f"{split}_avg_prop_acc"] = np.mean(length_scores)

        ### constraint metrics
        if global_story_consistency:
            metrics[f"{split}_avg_violations"] = np.mean(avg_constraint_violations) #np.mean(avg_constaint_violations)
            metrics[f"{split}_global_violations"] = np.mean(global_story_consistency)
            metrics[f"{split}_conditional_violations"] = np.sum(global_story_consistency) / np.sum(global_has_correct) \
              if np.sum(global_has_correct) > 0 else 0.

        ###
        if constraint_type_counts:
            for cname,overall_count in constraint_type_counts.items():
                if not overall_count: continue 
                metrics[f"{split}_{cname}"] = constraint_type_correct[cname] / overall_count

        ###
        if gen_total > 0:
            metrics[f"{split}_gen_em"] = gen_correct / gen_total
            scores = []
            for length,num_correct in correct_length_gen.items():
                score = num_correct / correct_length_gen_totals[length] if  num_correct > 0 else 0.
                metrics[f"{split}_gen_{length}_acc"] = score
                scores.append(score)
            metrics[f"{split}_gen_avg_acc"] = np.mean(scores)

        ### swaps
        if num_swapped > 0:
            metrics[f"{split}_num_swaps"] = num_swapped
            metrics[f"{split}_good_swaps"] = good_swaps
            metrics[f"{split}_swap_prop_acc"] = swap_correct / total
            for length in len_totals:
                score = len_correct_swap[length] / len_totals[length] if len_totals[length] > 0 else 0.
                metrics[f"{split}_{length}_swap_prop_acc"] = score
            
            
    return metrics

def correction_analysis(config):
    metrics =  {} 
    for split,out_file,fname in [
            ("dev",f"{config.dev_name}_eval.jsonl",config.dev_name),
            ("test",f"{config.test_name}_eval.jsonl",config.test_name),
        ]:
        full_path = os.path.join(config.output_dir,out_file)
        if not os.path.isfile(full_path):
            util_logger.warning('correct output not found')
            continue

        len_totals = defaultdict(int)
        len_correct = defaultdict(int)
        len_correct_swap = defaultdict(int)
        total = 0.
        correct = 0.
        swap_correct = 0.
        num_swapped = 0
        good_swaps = 0.
        global_story_consistency = []
        avg_global_story_consistency = []
        avg_constraint_violations = []
        global_has_correct = []
        correct_inferences = 0.
        total_inferences = 0.
        total_generations = 0.
        correct_generations = 0.

        constraint_type_counts = defaultdict(int)
        constraint_type_correct = defaultdict(int)
        correct_length_gen = defaultdict(int)
        correct_length_gen_totals = defaultdict(int)

        ### generation
        gen_correct = 0.
        gen_total = 0.

        num_corrected = 0.
        with open(full_path) as to_correct: 
            for k,line in enumerate(to_correct):
                line = line.strip()
                json_line = json.loads(line)
                events = json_line["events"]
                prop_lists = [i[1] for i in events]
                story_length = len(events)

                answer_prop = None

                ### global answers
                if "answers" in json_line and "gen_answers" in json_line:
                    answer_prop = json_line["gen_answers"][0]
                for e in events:
                    event = e[0]
                    plist = e[1]
                    predicted = e[2]
                    gold = e[3]
                    for l,p in enumerate(plist):
                        predicted_label = predicted[l]
                        gold_label = gold[l]
                        total += 1
                        len_totals[story_length] += 1.
                        swapped_pred = predicted_label

                        ## swap to true if the the model predicts anything other than unknown=`1`
                        if answer_prop and p == answer_prop and predicted_label != 1 and predicted_label != 2:
                            swapped_pred = 2
                            num_swapped += 1.
                            if gold_label == 2:
                                good_swaps += 1.

                        if predicted_label == gold_label:
                            correct += 1.
                            len_correct[story_length] += 1.
                        if swapped_pred == gold_label:
                            swap_correct += 1.
                            len_correct_swap[story_length] += 1.
                                                                        
        print(correct / total)
        print(num_swapped)
        print(good_swaps)
        print(swap_correct / total)
                                
                            
                            
        
def babi_analysis(config,convert=False):
    """Do analysis on babi output

    :param config: the global configuration 
    """
    metrics = {}
    for split,out_file,fname in [
            ("dev",f"{config.dev_name}_eval.jsonl",config.dev_name),
        ]:

        ### original file
        orig_list = []
        orig_path = os.path.join(config.data_dir,f"{fname}.jsonl")
        
        if not os.path.isfile(orig_path):
            util_logger.warning(f'Cannot find corresponding original file: {orig_path}')
        else:
            with open(orig_path) as orig_data:
                for line in orig_data:
                    line = line.strip()
                    json_line = json.loads(line)
                    orig_list.append(json_line)

        full_path = os.path.join(config.output_dir,out_file)
        if not os.path.isfile(full_path):
            util_logger.warning(f'Attempting to check output, file not found: {full_path}')
            continue

        my_data = []
        if convert is False:
            with open(full_path) as output_data:
                for k,line in enumerate(output_data):
                    line = line.strip()
                    json_line = json.loads(line)
                    my_data.append(json_line)
        else:
            my_data = load_flat(full_path,orig_list,config,metrics,split)

        movement_path_distance = defaultdict(int)
        movement_distance_correct = defaultdict(int)
        poss_path_distance = defaultdict(int)
        poss_distance_correct = defaultdict(int)

        for k,json_line in enumerate(my_data):
            orig_json = orig_list[k]

            
            ##################################
            # TRACKING MOVEMENT PERFORMANCE  #
            ##################################
            if "meta" in orig_json and "movement_paths" in orig_json["meta"]:
                mov_paths = orig_json["meta"]["movement_paths"]

                for (start,end,actor,location) in mov_paths:
                    length = 0
                    distance = 0

                    for j in range(start,end):
                        ### corresponding prediction
                        props = [
                            (orig_json["outputs"][j][u],p,u) for u,p in enumerate(orig_json["prop_lists"][j]) \
                            if actor in p and location in p 
                        ]
                        if not props or len(props) > 1:
                            util_logger.warning(f'movement path at {k} seems incorrect')
                            continue
                        target_label,target_prop,target_index = props[0]
                        assert target_label == 1.0
                        output_prop = json_line["events"][j][1][target_index]
                        assert target_prop == output_prop
                        predicted_label = int(json_line["events"][j][2][target_index])

                        assert int(json_line["events"][j][3][target_index]) == 2

                        if predicted_label == 2.0:
                            movement_distance_correct[distance] += 1

                        movement_path_distance[distance] += 1
                        distance += 1

            ###################################
            # TRACKING POSESSION PERFORMANCE  #
            ###################################

            if "meta" in orig_json and 'possession_paths' in orig_json["meta"]:
                pos_paths = orig_json["meta"]['possession_paths']
                for (start,end,actor,obj) in pos_paths:
                    length = 0
                    distance = 0

                    for j in range(start,end):
                        ### corresponding prediction
                        props = [
                            (orig_json["outputs"][j][u],p,u) for u,p in enumerate(orig_json["prop_lists"][j]) \
                            if actor in p and obj in p 
                        ]
                        if not props or len(props) > 1:
                            util_logger.warning(f'movement path at {k} seems incorrect')
                            continue
                        target_label,target_prop,target_index = props[0]
                        assert target_label == 1.0
                        output_prop = json_line["events"][j][1][target_index]
                        assert target_prop == output_prop
                        predicted_label = int(json_line["events"][j][2][target_index])

                        assert int(json_line["events"][j][3][target_index]) == 2

                        if predicted_label == 2.0:
                            poss_distance_correct[distance] += 1
                        poss_path_distance[distance] += 1
                        distance += 1

        #print(movement_path_distance)
        #print(movement_distance_correct)
        for distance in movement_path_distance:
            total = movement_path_distance[distance]
            correct = movement_distance_correct[distance]
            metrics[f"{split}_movement_{distance}_total"] = total
            metrics[f"{split}_movement_{distance}_correct"] = correct

        for distance in poss_distance_correct:
            total = poss_path_distance[distance]
            correct = poss_distance_correct[distance]
            metrics[f"{split}_poss_{distance}_total"] = total
            metrics[f"{split}_poss_{distance}_correct"] = correct

        return metrics


if __name__ == "__main__":
    from situation_modeling import get_config
    config = get_config('runner')

    ### babi analysis
    config.output_dir = "_runs/babi_500_analysis/baseline"
    config.data_dir = "_runs/babi_500_analysis/baseline"
    config.dev_name = "dev"
    metrics = situation_analysis(config,convert=True)
    frame = babi_analysis(config,convert=True)
    metrics.update(frame)
    print(json.dumps(metrics,indent=4,sort_keys=True))


    ### correction analysis
    # config.output_dir = "_runs/correction"
    # config.data_dir = "_runs/correction"
    # config.dev_name = "correction_dev"
    # config.no_constraint_loss = True
    # metrics = situation_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))
    

    ### STANDARD EXAMPLE 
    # config.output_dir = "_runs/v10_ex_output"
    # config.data_dir = "_runs/v10_ex_output"
    # config.dev_name = "mix_dev"
    # config.test_name = "ood_dev"
    # metrics = situation_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))

    ### newest babi example
    # config.output_dir = "_runs/arc_analysis"
    # config.data_dir = "_runs/arc_analysis"
    # config.dev_name = "dev"
    # metrics = situation_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))

    ### 
    # config.output_dir = "_runs/bilstm_trip_baseline"
    # config.data_dir = "_runs/bilstm_trip_baseline"
    # config.dev_name = "dev"
    # metrics = situation_analysis(config,convert=True)
    # print(json.dumps(metrics,indent=4,sort_keys=True))

    # config.output_dir = "_runs/transformer_trip_baseline/second"
    # config.data_dir = "_runs/transformer_trip_baseline/second"
    # config.dev_name = "dev"
    # metrics = situation_analysis(config,convert=True)
    # print(json.dumps(metrics,indent=4,sort_keys=True))
    

    # config.output_dir = "_runs/obqa_analysis"
    # config.data_dir = "_runs/obqa_analysis"
    # config.dev_name = "dev"
    # metrics = situation_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))


    # config.output_dir = "_runs/arc_challenge_prop_model/"
    # config.data_dir = "_runs/arc_challenge_prop_model/"
    # config.dev_name = "dev"
    # metrics = situation_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))


    ### SENTENCE TRANSFORMER EXAMPLE
    # config.output_dir = "_runs/v10_ex_output_flat"
    # config.data_dir = "_runs/v10_ex_output_flat"
    # config.dev_name = "mix_dev"
    # config.test_name = "ood_dev"
    # metrics = situation_analysis(config,convert=True)
    #print(json.dumps(metrics,indent=4,sort_keys=True))


    #print(metrics) 
    #print(parse_basic_implications_conjunctions)

    ## gneration output example
    # config.output_dir = "_runs/gen_output"
    # config.data_dir = "_runs/gen_output"
    # config.dev_name = "mix_dev"
    # config.test_name = "ood_dev"
    # metrics = text_to_text_analysis(config)
    # print(json.dumps(metrics,indent=4,sort_keys=True))

    # config.output_dir = "_runs/with_qa"
    # config.data_dir = "_runs/with_qa"
    # config.dev_name = "mix_dev"
    # config.test_name = "ood_dev"
    # metrics = situation_analysis(config)
    # metrics = text_to_text_analysis(config)
    #print(json.dumps(metrics,indent=4,sort_keys=True))
