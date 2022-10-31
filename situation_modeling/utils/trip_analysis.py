from os import truncate
from pathlib import Path
from typing import List
import inflect
import json
import numpy as np
import os
import pandas as pd
import logging

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

util_logger = logging.getLogger('situation_modeling.utils.trip_analysis')

REPO_ROOT = Path(__file__).parents[2]

ROBERTA_RESULTS = {
    "dev": str(REPO_ROOT / "situation_modeling/utils/trip_eval/results_cloze_implausible_explanations_consistency_dev.jsonl"),
    "test": str(REPO_ROOT / "situation_modeling/utils/trip_eval/results_cloze_explanations_consistency_test.jsonl")
}

### copied from https://github.com/sled-group/Verifiable-Coherent-NLU/blob/main/www/dataset/ann.py
# to avoid reliance on external repo

att_to_idx = {'h_location': 0, 
              'conscious': 1, 
              'wearing': 2, 
              'h_wet': 3, 
              'hygiene': 4, 
              'location': 5, 
              'exist': 6, 
              'clean': 7, 
              'power': 8, 
              'functional': 9, 
              'pieces': 10, 
              'wet': 11, 
              'open': 12, 
              'temperature': 13, 
              'solid': 14, 
              'contain': 15, 
              'running': 16, 
              'moveable': 17, 
              'mixed': 18, 
              'edible': 19}
idx_to_att = {v: k for k,v in att_to_idx.items()}
human_atts = ['h_location', 'conscious', 'wearing', 'h_wet', 'hygiene']
att_to_num_classes = {
    "h_location": 3,
    "conscious": 9,
    "wearing": 9,
    "h_wet": 9,
    "hygiene": 9,
    "location": 9,
    "exist": 9,
    "clean": 9,
    "power": 9,
    "functional": 9,
    "pieces": 9,
    "wet": 9,
    "open": 9,
    "temperature": 9,
    "solid": 9,
    "contain": 9,
    "running": 9,
    "moveable": 9,
    "mixed": 9,
    "edible": 9
}

# This is valid for pre -> post, pre, and post (since 2 means true -> true and also just true)
att_default_values = {'h_location': 0, 'conscious': 2, 'wearing': 0, 'h_wet': 0, 'hygiene': 0, 'location': 0, 'exist': 2, 'clean': 0, 'power': 0, 'functional': 2, 'pieces': 0, 'wet': 0, 'open': 0, 'temperature': 0, 'solid': 0, 'contain': 0, 'running': 0, 'moveable': 2, 'mixed': 0, 'edible': 0}

# Naive way to check if a given entity name is a human
def is_human(entity):
  return (entity[0].isupper() and entity != 'TV')


att_types = {'h_location': 'h_location', 
                    'conscious': 'default', 
                    'wearing': 'default', 
                    'h_wet': 'default', 
                    'hygiene': 'default', 
                    'location': 'location', 
                    'exist': 'default', 
                    'clean': 'default',
                    'power': 'default', 
                    'functional': 'default', 
                    'pieces': 'default', 
                    'wet': 'default', 
                    'open': 'default', 
                    'temperature': 'default', 
                    'solid': 'default', 
                    'contain': 'default', 
                    'running': 'default', 
                    'moveable': 'default', 
                    'mixed': 'default', 
                    'edible': 'default'}

att_types = {'h_location': 'h_location', 
                    'conscious': 'default', 
                    'wearing': 'default', 
                    'h_wet': 'default', 
                    'hygiene': 'default', 
                    'location': 'location', 
                    'exist': 'default', 
                    'clean': 'default',
                    'power': 'default', 
                    'functional': 'default', 
                    'pieces': 'default', 
                    'wet': 'default', 
                    'open': 'default', 
                    'temperature': 'default', 
                    'solid': 'default', 
                    'contain': 'default', 
                    'running': 'default', 
                    'moveable': 'default', 
                    'mixed': 'default', 
                    'edible': 'default'}
att_change_dir = {'h_location': {0: 'does not move to a new location', 1: 'disappears', 2: 'moves somewhere new'},
            'location': {0: 'does not move to a new location', 1: 'disappears', 2: 'is picked up', 3: 'is put down', 4: 'is put on', 5: 'is removed', 6: 'is put into a container', 7: 'is taken out of a container', 8: 'moved somewhere new'},
            'default': {0: (-1,-1), 1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (0, 1), 5: (-1, 0), 6: (-1, 1), 7: (0, -1), 8: (1, -1)}}
att_change_dir_bw = {'default': {(-1, -1): 0, (0, 0): 1, (1, 1): 2, (1, 0): 3, (0, 1): 4, (-1, 0): 5, (-1, 1): 6, (0, -1): 7, (1, -1): 8}}
att_adj = { 'conscious': ('unconscious', 'conscious'),
            'wearing': ('undressed', 'dressed'), 
            'h_wet': ('dry', 'wet'), 
            'hygiene': ('dirty', 'clean'), 
            'exist': ('nonexistent', 'existent'), 
            'clean': ('dirty', 'clean'),
            'power': ('unpowered', 'powered'), 
            'functional': ('broken', 'functional'), 
            'pieces': ('whole', 'in pieces'), 
            'wet': ('dry', 'wet'), 
            'open': ('closed', 'open'), 
            'temperature': ('cold', 'hot'), 
            'solid': ('fluid', 'solid'), 
            'contain': ('empty', 'occupied'), 
            'running': ('turned off', 'turned on'), 
            'moveable': ('stuck', 'moveable'), 
            'mixed': ('separated', 'mixed'), 
            'edible': ('inedible', 'edible')}


loc_prop_based_on_label = True

labels_dict = {0: 'False', 1: 'Unknown', 2: 'True',
               "0": 'False', "1": 'Unknown', "2": 'True'}
labels_dict_paper = {0: 'Unknown', 1: 'False', 2: 'True'}

def read_run_output(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = [{} for _ in range(len(lines))]
    for idx, line in enumerate(lines):
        json_object = json.loads(line)
        guid = json_object['guid']
        texts = json_object['texts']
        events = json_object['events']
        questions = json_object.get('questions', [])
        data[idx]['guid'], data[idx]['texts'], data[idx]['events'] = guid, texts, events
        data[idx]['questions'] = questions
    return data


def read_ground_truth(filename):
    f = open(filename, "r")
    lines = f.readlines()
    data = [{} for _ in range(len(lines))]
    guid_to_story_map = {}
    for idx, line in enumerate(lines):
        json_object = json.loads(line)
        story_id = json_object['story_id']
        example_id = json_object['example_id']
        texts = json_object['texts']
        attr_lists = json_object['attr_lists']
        outputs = json_object['outputs']
        guid = json_object['guid']
        confl_sents = json_object['confl_sents']
        breakpoint = json_object['breakpoint']
        data[idx]['story_id'], data[idx]['example_id'], data[idx]['texts'], data[idx]['attr_lists'],\
        data[idx]['outputs'], data[idx]['guid'], data[idx]['confl_sents'], data[idx]['breakpoint'] = \
            story_id, example_id, texts, attr_lists, outputs, guid, confl_sents, breakpoint
        guid_to_story_map[guid] = json_object
    return data, guid_to_story_map

def get_prop_type(prop):
    prop_type = ""
    if " is " in (" " + prop + " ") or " are " in (" " + prop + " "):
        prop_type = 'effect'
    elif " was " in (" " + prop + " ") or " were " in (" " + prop + " "):
        prop_type = 'precondition'
    return prop_type

def get_props_metrics_result(prop_type, total_attr_f1_scores):
    props_metrics_result = {}
    props_metrics_summary_result = 0

    y_test_all_attributes, y_pred_all_attributes = [], []
    for attribute in total_attr_f1_scores.keys():
        props_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test, y_pred = [], []
        for i, prop in enumerate(total_attr_f1_scores[attribute][-3]):
            if prop[1] != prop_type:
                continue
            y_test.append(total_attr_f1_scores[attribute][-1][i])
            y_test_all_attributes.append(total_attr_f1_scores[attribute][-1][i])

            y_pred.append(total_attr_f1_scores[attribute][-2][i])
            y_pred_all_attributes.append(total_attr_f1_scores[attribute][-2][i])

            # print("Confusion Matrix for: " + attribute)
            metrics_result = get_confusion_matrix(y_test, y_pred, False)
            for metric_name, metric_val in metrics_result.items():
                props_metrics_result[attribute][metric_name] = metric_val
            props_metrics_result[attribute]['Freq'] = len(y_test)

    # print("Confusion Matrix for: " + attribute)
    metrics_result = get_confusion_matrix(y_test_all_attributes, y_pred_all_attributes, False)
    for metric_name, metric_val in metrics_result.items():
        if metric_name != "MacroF1":
            continue
        props_metrics_summary_result = metric_val


    return props_metrics_result, props_metrics_summary_result


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     thresh = cm.max() / 2.
#     import itertools
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()

def get_confusion_matrix(y_test, y_pred, plot_graph=False):
    
    res = {}
    freq = {}
    for val in y_test:
        if val not in freq:
            freq[val] = 1
        else:
            freq[val] += 1
    

    freq = {}
    
    for val in y_pred:
        if val not in freq:
            freq[val] = 1
        else:
            freq[val] += 1
    

    confusion = confusion_matrix(y_test, y_pred)
    res["confusion"] = confusion

    # importing accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    # print('\nAccuracy: {:.2f}\n'.format(acc))

    micro_f1_score = f1_score(y_test, y_pred, average='micro')
    # print('Micro F1-score: {:.2f}\n'.format(micro_f1_score))
    macro_f1_score = f1_score(y_test, y_pred, average='macro')
    res['MacroF1'] = round(macro_f1_score, 2)
    res['MicroF1'] = round(micro_f1_score, 2)
    res['Accuracy'] = round(acc, 2)
    # print('Macro F1-score: {:.2f}\n'.format(macro_f1_score))

    if plot_graph:
        cm = metrics.confusion_matrix(y_test, y_pred)
        # plot_confusion_matrix(cm, classes=['false', 'unknown', 'true'])
        # print('\nClassification Report\n')
        report = classification_report(y_test, y_pred, target_names=['false', 'unknown', 'true'])
        res["classification_report"] = report
    
    return res

def read_paper_verifiability_results_all(filename, include_loc_attributes=True):

    res = {}

    inflect_eng = inflect.engine()
    states_verifiable = True

    f = open(filename, "r")
    list_of_jsons = []
    count_stories_to_verify, count_verified_stories = 0, 0
    acc = 0
    verifiability_results = []
    total_stories, total_wrong_story_predictions, total_correct_but_inconsistent_predictions = 0, 0, 0
    effect_attr_f1_scores, precondition_attr_f1_scores, total_attr_f1_scores = {}, {}, {}
    for j in f:
        list_of_jsons = json.loads(j)
    for story in list_of_jsons:
        total_stories += 1
        # example_id = story['example_id']
        # if not example_id == "13-C0":
        #     continue
        
        # use gold conflict labels
        l_conflict, p_conflict = story['conflict_label'], story['conflict_label']

        states_verifiable = True
        found_states = False
        count_stories_to_verify += 1
        l_eff, p_eff = story['effects_label'], story['effects_pred']

        # label of the implausible story - use gold label, since we want to eval any on
        # story, not just where model was right
        story_label_string = 'story1' if story['story_label'] == 0 else 'story0'

        for sl, sp in [(l_eff, p_eff)]:
            for sl_e, sp_e in zip(sl, sp):
                attr_labels = sl[sl_e][str(l_conflict[0])]
                attr_predictions = sp[sp_e][str(p_conflict[0])]
                # print(f"attr_predictions: {attr_predictions}")
                for key, val in attr_predictions.items():
                    is_loc_prop = False
                    effect_sentence = story[story_label_string].split('\n')[l_conflict[0]]
                    axillary_verb = "are" if inflect_eng.singular_noun(sp_e) else "is"

                    if key in ['h_location', 'location']:
                        labels_dict = att_change_dir[key]
                        is_loc_prop = True
                        if loc_prop_based_on_label:
                            prop = sl_e + " " + att_change_dir[key][attr_labels[key]] if key in attr_labels else "The location of the " + sl_e + " is unknown"
                        else:
                            prop = sl_e + " " + att_change_dir[key][val]


                    else:
                        labels_dict = labels_dict_paper
                        prop = sp_e + " " + axillary_verb + " " + att_adj[key][1]

                    prediction = labels_dict[val]
                    if key in attr_labels:
                        label = labels_dict[attr_labels[key]]
                    else:
                        label = labels_dict_paper[att_default_values[key]]


                    if prediction != att_default_values[key]:
                        found_states = True

                    if key in ['h_location', 'location'] and not include_loc_attributes:
                        label = ''
                        prediction = ''

                    if label != prediction:
                        states_verifiable = False

                    if key not in total_attr_f1_scores:
                        total_attr_f1_scores[key] = [[], [], [], [], []]
                    if key not in effect_attr_f1_scores:
                        effect_attr_f1_scores[key] = [[], [], [], [], []]

                    effect_attr_f1_scores[key][0].append(story['example_id'])
                    total_attr_f1_scores[key][0].append(story['example_id'])

                    effect_attr_f1_scores[key][1].append(effect_sentence)
                    total_attr_f1_scores[key][1].append(effect_sentence)

                    effect_attr_f1_scores[key][2].append((prop, get_prop_type(prop)))
                    total_attr_f1_scores[key][2].append((prop, get_prop_type(prop)))

                    effect_attr_f1_scores[key][3].append(prediction)
                    total_attr_f1_scores[key][3].append(prediction)

                    effect_attr_f1_scores[key][4].append(label)
                    total_attr_f1_scores[key][4].append(label)

                    verifiability_results.append({'ExampleId': story['example_id'], 'SentenceType': 'effect',
                                                  'Sentence': effect_sentence,
                                                  'Prop': prop,
                                                  'Attribute': key, 
                                                  'Label': label, 
                                                  'Prediction': prediction, 
                                                  'IsCorrect': label == prediction,
                                                  "Entity": sl_e})
                    if label == prediction:
                        acc += 1

        l_prec, p_prec = story['preconditions_label'], story['preconditions_pred']

        for sl, sp in [(l_prec, p_prec)]:
            for sl_e, sp_e in zip(sl, sp):
                attr_labels = sl[sl_e][str(l_conflict[1])]
                attr_predictions = sp[sp_e][str(p_conflict[1])]

                for key, val in attr_predictions.items():
                    is_loc_prop = False
                    prec_sentence = story[story_label_string].split('\n')[l_conflict[1]]
                    axillary_verb = "were" if inflect_eng.singular_noun(sp_e) else "was"


                    if key in ['h_location', 'location']:
                        labels_dict = att_change_dir[key]
                        is_loc_prop = True
                        if loc_prop_based_on_label:
                            prop = sl_e + " " + att_change_dir[key][attr_labels[key]] if key in attr_labels else "The location of the " + sl_e + " is unknown"
                        else:
                            prop = sl_e + " " + att_change_dir[key][val]
                    else:
                        labels_dict = labels_dict_paper
                        prop = sp_e + " " + axillary_verb + " " + att_adj[key][1]


                    prediction = labels_dict[val]
                    if key in attr_labels:
                        label = labels_dict[attr_labels[key]]
                    else:
                        label = labels_dict_paper[att_default_values[key]]

                    if prediction != att_default_values[key]:
                        found_states = True

                    if key in ['h_location', 'location'] and not include_loc_attributes:
                        label = ''
                        prediction = ''

                    if label != prediction:
                        states_verifiable = False

                    if key not in total_attr_f1_scores:
                        total_attr_f1_scores[key] = [[], [], [], [], []]
                    if key not in precondition_attr_f1_scores:
                        precondition_attr_f1_scores[key] = [[], [], [], [], []]

                    precondition_attr_f1_scores[key][0].append(story['example_id'])
                    total_attr_f1_scores[key][0].append(story['example_id'])

                    precondition_attr_f1_scores[key][1].append(prec_sentence)
                    total_attr_f1_scores[key][1].append(prec_sentence)

                    precondition_attr_f1_scores[key][2].append((prop, get_prop_type(prop)))
                    total_attr_f1_scores[key][2].append((prop, get_prop_type(prop)))

                    precondition_attr_f1_scores[key][3].append(prediction)
                    total_attr_f1_scores[key][3].append(prediction)

                    precondition_attr_f1_scores[key][4].append(label)
                    total_attr_f1_scores[key][4].append(label)


                    verifiability_results.append({'ExampleId': story['example_id'], 'SentenceType': 'precondition',
                                                  'Sentence': prec_sentence,
                                                  'Prop': prop,
                                                  'Attribute': key,
                                                  'Label': label,
                                                  'Prediction': prediction, 
                                                  'IsCorrect': label == prediction,
                                                  "Entity": sl_e
                                                 
                                                  
                                                  })
                    if label == prediction:
                        acc += 1

        verifiable = states_verifiable and found_states
        if verifiable:
            count_verified_stories += 1

    
    effect_metrics_result = {}
    for attribute in effect_attr_f1_scores:
        effect_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = effect_attr_f1_scores[attribute][-1]
        y_pred = effect_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            effect_metrics_result[attribute][metric_name] = metric_val
        effect_metrics_result[attribute]['Freq'] = len(effect_attr_f1_scores[attribute][0])

    precondition_metrics_result = {}

    for attribute in sorted(precondition_attr_f1_scores):
        precondition_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = precondition_attr_f1_scores[attribute][-1]
        y_pred = precondition_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            precondition_metrics_result[attribute][metric_name] = metric_val
        precondition_metrics_result[attribute]['Freq'] = len(precondition_attr_f1_scores[attribute][0])


    # print("total attributes: ")
    # print()
    total_metrics_result = {}
    for attribute in sorted(total_attr_f1_scores):
        total_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = total_attr_f1_scores[attribute][-1]
        y_pred = total_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            total_metrics_result[attribute][metric_name] = metric_val
        total_metrics_result[attribute]['Freq'] = len(total_attr_f1_scores[attribute][0])



    res["acc"] = (acc / len(verifiability_results))
    res["total_stories"] = total_stories
    res["wrong_predictions"] = total_wrong_story_predictions
    # + "(" +
        #   str(round(total_wrong_story_predictions / total_stories * 100, 1)) + "%)")
    res["correct but inconsistent predictions"] = total_correct_but_inconsistent_predictions

    # print("consistent predictions: " + str(count_stories_to_verify) +
        #   "(" + str(round(count_stories_to_verify / total_stories * 100, 1)) + "%)")
    verified_stories_percent = round(count_verified_stories / total_stories * 100, 1)


    res["total_props"] = len(verifiability_results)
    example_id_propositions_freq = {}
    for instance in verifiability_results:
        if instance['ExampleId'] not in example_id_propositions_freq:
            example_id_propositions_freq[instance['ExampleId']] = 0
        example_id_propositions_freq[instance['ExampleId']] += 1
    avg_props = round(sum(example_id_propositions_freq.values()) / len(example_id_propositions_freq), 3)
    res["avg_props_per_story"] = avg_props

    precondition_props_metrics_result, precondition_props_metrics_summary_result = get_props_metrics_result("precondition", total_attr_f1_scores)
    effect_props_metrics_result, effect_props_metrics_summary_result = get_props_metrics_result("effect", total_attr_f1_scores)

    res.update( {

        "verifiability_results": verifiability_results,
        "verifiability_pct": round((count_verified_stories / count_stories_to_verify), 3),
        "verified_stories_count": count_verified_stories,
        "effect_metrics_result": effect_metrics_result,
        "precondition_metrics_result": precondition_metrics_result,
        "total_metrics_result": total_metrics_result,
        "precondition_props_metrics_result": precondition_props_metrics_result,
        "effect_props_metrics_result": effect_props_metrics_result,
        "precondition_props_metrics_summary_result": precondition_props_metrics_summary_result,
        "effect_props_metrics_summary_result": effect_props_metrics_summary_result

    })
    return res

def eval_location_props(label_prop: str, entity: str, attribute: str, bp_event_output: List):
    """
    Evaluate location propositions (h_location/location) by taking argmax of True labels across all relevant classes.



    Args:
        label_prop (str): Proposition corresponding to the correct class
        attribute (str): attribute type - (h_location/location)
        bp_event_output (List): Breakpoint model output for this timestep's event.
    """
    event_props = bp_event_output[1]
    prop_idx = event_props.index(label_prop)

    all_prop_values = list(att_change_dir[attribute].values())
    all_location_props = [f"{entity} {value}" for value in all_prop_values]
    assert(label_prop in all_location_props)

    pred_probs = []
    for p in all_location_props:
        assert(p in event_props)
        prop_idx = event_props.index(p)
        pred_probs.append((p, bp_event_output[4][prop_idx]))
    
    prediction, _ = max(pred_probs,key=lambda x: x[1])

    return prediction


    

def model_eval_verifiability(evaluated_file_name, 
                            output_file_name, 
                            paper_verifiability_results, 
                            paper_consistent_examples, 
                            include_loc_attributes=True,
                            classify_locs=True):

    res = {}

    res["verifiable_consistent_stories"] = []

    ground_truth, guid_to_story = read_ground_truth(evaluated_file_name)
    model_output = read_run_output(output_file_name)
    count_verifiable_consistent_stories = 0
    count_total_consistent_stories = 0
    effect_attr_f1_scores, precondition_attr_f1_scores, total_attr_f1_scores = {}, {}, {}

    for story in model_output:
        # skip plausibility instances
        if len(story["questions"]) > 0:
            if "$plaus" in story["questions"][0]:
                continue

        is_verifiable = True
        story_guid = story['guid']
        if len(guid_to_story[story_guid]['confl_sents']) == 0:
            continue
        example_id, confl_sents, attr_lists = guid_to_story[story_guid]['example_id'], \
                                              [guid_to_story[story_guid]['confl_sents'][-1],
                                               guid_to_story[story_guid]['breakpoint']], \
                                              guid_to_story[story_guid]['attr_lists']

        if example_id not in paper_consistent_examples:
            continue

        count_total_consistent_stories += 1
        effect_sentence, precondition_sentence = story['events'][confl_sents[0]], story['events'][confl_sents[1]]

        example_effect_sentences = []
        example_precondition_sentences = []
        for r in paper_verifiability_results:
            if r['ExampleId'] == example_id and r['SentenceType'] == 'effect':
                example_effect_sentences.append(r)
            elif r['ExampleId'] == example_id and r['SentenceType'] == 'precondition':
                example_precondition_sentences.append(r)

        for sentence in example_effect_sentences:
            sentence['ModelPrediction'] = 'Unknown'
            sentence['ModelLabel'] = 'Unknown'
            sentence['IsModelCorrect'] = sentence['Label'] == sentence['ModelPrediction']

            if sentence['Sentence'] == effect_sentence[0] and sentence['Prop'] in effect_sentence[1]:
                prop_idx = effect_sentence[1].index(sentence['Prop'])
                prediction = labels_dict[effect_sentence[2][prop_idx]]
                label = labels_dict[effect_sentence[3][prop_idx]]
                is_model_correct = label == prediction
                sentence['ModelPrediction'] = prediction
                sentence['ModelLabel'] = label
                sentence['IsModelCorrect'] = is_model_correct

                if sentence['Attribute'] not in total_attr_f1_scores:
                    total_attr_f1_scores[sentence['Attribute']] = [[], [], [], [], []]
                if sentence['Attribute'] not in effect_attr_f1_scores:
                    effect_attr_f1_scores[sentence['Attribute']] = [[], [], [], [], []]
                    
                if sentence['Attribute'] in ['h_location', 'location'] and classify_locs:
                    # run classification by taking argmax of True labels across all relevant classes
                    label = sentence['Prop']
                    prediction = eval_location_props(label_prop=label, 
                                        entity=sentence['Entity'],
                                        attribute=sentence['Attribute'],
                                        bp_event_output=effect_sentence)
                    is_model_correct = label == prediction
                    sentence['ModelPrediction'] = prediction
                    sentence['ModelLabel'] = label
                    sentence['IsModelCorrect'] = is_model_correct

                if sentence['Attribute'] in ['h_location', 'location'] and not include_loc_attributes:
                    label, prediction = '', ''
                    sentence['ModelLabel'], sentence['ModelPrediction'] = label, prediction
                    sentence['IsModelCorrect'] = True

                if label != prediction:
                    is_verifiable = False

                effect_attr_f1_scores[sentence['Attribute']][0].append(sentence['ExampleId'])
                total_attr_f1_scores[sentence['Attribute']][0].append(sentence['ExampleId'])

                effect_attr_f1_scores[sentence['Attribute']][1].append(effect_sentence[0])
                total_attr_f1_scores[sentence['Attribute']][1].append(effect_sentence[0])

                effect_attr_f1_scores[sentence['Attribute']][2].append((sentence['Prop'], get_prop_type(sentence['Prop'])))
                total_attr_f1_scores[sentence['Attribute']][2].append((sentence['Prop'], get_prop_type(sentence['Prop'])))

                effect_attr_f1_scores[sentence['Attribute']][3].append(sentence['ModelPrediction'])
                total_attr_f1_scores[sentence['Attribute']][3].append(sentence['ModelPrediction'])

                effect_attr_f1_scores[sentence['Attribute']][4].append(sentence['ModelLabel'])
                total_attr_f1_scores[sentence['Attribute']][4].append(sentence['ModelLabel'])

        for sentence in example_precondition_sentences:
            sentence['ModelPrediction'] = 'Unknown'
            sentence['ModelLabel'] = 'Unknown'
            sentence['IsModelCorrect'] = sentence['Label'] == sentence['ModelPrediction']

            if sentence['Sentence'] == precondition_sentence[0] and sentence['Prop'] in precondition_sentence[1]:
                prop_idx = precondition_sentence[1].index(sentence['Prop'])
                prediction = labels_dict[precondition_sentence[2][prop_idx]]
                label = labels_dict[precondition_sentence[3][prop_idx]]
                is_model_correct = label == prediction
                sentence['ModelPrediction'] = prediction
                sentence['ModelLabel'] = label
                sentence['IsModelCorrect'] = is_model_correct

                if sentence['Attribute'] in ['h_location', 'location'] and classify_locs:
                    # run classification by taking argmax of True labels across all relevant classes
                    label = sentence['Prop']
                    # print(sentence)
                    prediction = eval_location_props(label_prop=label, 
                                        entity=sentence['Entity'],
                                        attribute=sentence['Attribute'],
                                        bp_event_output=precondition_sentence)
                    is_model_correct = label == prediction
                    sentence['ModelPrediction'] = prediction
                    sentence['ModelLabel'] = label
                    sentence['IsModelCorrect'] = is_model_correct

                if sentence['Attribute'] in ['h_location', 'location'] and not include_loc_attributes:
                    label, prediction = '', ''
                    sentence['ModelLabel'], sentence['ModelPrediction'] = label, prediction
                    sentence['IsModelCorrect'] = True

                if sentence['Attribute'] not in total_attr_f1_scores:
                    total_attr_f1_scores[sentence['Attribute']] = [[], [], [], [], []]
                if sentence['Attribute'] not in precondition_attr_f1_scores:
                    precondition_attr_f1_scores[sentence['Attribute']] = [[], [], [], [], []]

                if label != prediction:
                    is_verifiable = False

                precondition_attr_f1_scores[sentence['Attribute']][0].append(sentence['ExampleId'])
                total_attr_f1_scores[sentence['Attribute']][0].append(sentence['ExampleId'])

                precondition_attr_f1_scores[sentence['Attribute']][1].append(precondition_sentence[0])
                total_attr_f1_scores[sentence['Attribute']][1].append(precondition_sentence[0])

                precondition_attr_f1_scores[sentence['Attribute']][2].append((sentence['Prop'], get_prop_type(sentence['Prop'])))
                total_attr_f1_scores[sentence['Attribute']][2].append((sentence['Prop'], get_prop_type(sentence['Prop'])))

                precondition_attr_f1_scores[sentence['Attribute']][3].append(sentence['ModelPrediction'])
                total_attr_f1_scores[sentence['Attribute']][3].append(sentence['ModelPrediction'])

                precondition_attr_f1_scores[sentence['Attribute']][4].append(sentence['ModelLabel'])
                total_attr_f1_scores[sentence['Attribute']][4].append(sentence['ModelLabel'])

        if is_verifiable and (example_effect_sentences or example_precondition_sentences):
            count_verifiable_consistent_stories += 1
            res["verifiable_consistent_stories"].append(example_id)

    # print("---------------------------------------------------------------")
    # print("Our Model Evaluation: ")
    res["total_consistent_stories"] = count_total_consistent_stories
    res["total_verifiable_stories"] = count_verifiable_consistent_stories
    # print("total verifiable stories: " + str(count_verifiable_consistent_stories))
    if count_total_consistent_stories > 0:
        verifiability_percents = round      (count_verifiable_consistent_stories / count_total_consistent_stories, 4)
    else:
        verifiability_percents = 0



    # print("verifiable(%): " + str(verifiability_percents))

    res["effect_attr_f1_scores"] = effect_attr_f1_scores 
    effect_metrics_result = {}
    for attribute in effect_attr_f1_scores.keys():
        effect_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = effect_attr_f1_scores[attribute][-1]
        y_pred = effect_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            effect_metrics_result[attribute][metric_name] = metric_val
        effect_metrics_result[attribute]['Freq'] = len(effect_attr_f1_scores[attribute][0])



    res["precondition_attr_f1_scores"] = precondition_attr_f1_scores 

    precondition_metrics_result = {}
    for attribute in precondition_attr_f1_scores.keys():
        precondition_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = precondition_attr_f1_scores[attribute][-1]
        y_pred = precondition_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            precondition_metrics_result[attribute][metric_name] = metric_val
        precondition_metrics_result[attribute]['Freq'] = len(precondition_attr_f1_scores[attribute][0])

    total_metrics_result = {}
    for attribute in total_attr_f1_scores.keys():
        total_metrics_result[attribute] = {'Freq': 0, 'Accuracy': 0, 'MicroF1': 0, 'MacroF1': 0}
        y_test = total_attr_f1_scores[attribute][-1]
        y_pred = total_attr_f1_scores[attribute][-2]
        # print("Confusion Matrix for: " + attribute)
        metrics_result = get_confusion_matrix(y_test, y_pred, False)
        for metric_name, metric_val in metrics_result.items():
            total_metrics_result[attribute][metric_name] = metric_val
        total_metrics_result[attribute]['Freq'] = len(total_attr_f1_scores[attribute][0])


    precondition_props_metrics_result, precondition_props_metrics_summary_result = get_props_metrics_result("precondition", total_attr_f1_scores)
    effect_props_metrics_result, effect_props_metrics_summary_result = get_props_metrics_result("effect", total_attr_f1_scores)

    # for key, val in effect_metrics_result.items():
    #     print(key, val)
    # print()
    # print()
    # for key, val in precondition_metrics_result.items():
    #     print(key, val)
    # print()
    # print()
    # for key, val in total_metrics_result.items():
    #     print(key, val)

    # print("Our Model Evaluation: ")
    # print("total consistent stories: " + str(count_total_consistent_stories))
    # print("total verifiable stories: " + str(count_verifiable_consistent_stories))
    # print("verifiable(%): " + str(
    #     int(round(count_verifiable_consistent_stories / count_total_consistent_stories, 2) * 100)))

    res.update({"paper_verifiability_results": paper_verifiability_results, 
    "verifiability_percents": verifiability_percents,
     "count_verifiable_consistent_stories": count_verifiable_consistent_stories,
     "effect_metrics_result": effect_metrics_result,
     "precondition_metrics_result": precondition_metrics_result,
     "total_metrics_result": total_metrics_result,
     "effect_props_metrics_result": effect_props_metrics_result,
     "precondition_props_metrics_summary_result": precondition_props_metrics_summary_result,
     "effect_props_metrics_summary_result": effect_props_metrics_summary_result})
    return res


def calc_instance_metrics(row, answers="answers", 
    gen_answers="gen_answers", story_id="example_id"):

    question = row["questions"][0]
    assert(("$plaus" in question) or ("$conflict" in question))

    res = {
        "example_id": row[story_id],
        "answers": row[answers],
        "gen_answers": row[gen_answers]
        }

    if "$plaus" in question:
        # plausibility instance
        res["type"] = "plausibility"
        res["correct_plaus"] = row[answers][0] == row[gen_answers][0]
    elif "$conflict" in question:
        # conflict detection
        res["type"] = "conflict"
        conflict_label = set([int(x) for x in row[answers][0].split(",")])
        conflict_pred = set() 
        try:
            conflict_pred = set([int(x) for x in row[gen_answers][0].split(",")])
        except:
            pass # if malformed prediction string, just assume it was empty
        
        tp = len(conflict_pred.intersection(conflict_label))
        fn = len(conflict_label.difference(conflict_pred))
        fp = len(conflict_pred.difference(conflict_label))
        
        res.update({
        "correct_confl": (tp == 2) & (fp == 0) & (fn == 0),
        "TP": tp,
        "FP": fp,
        "FN": fn

        })


    return res

def get_actual_example_id(row):
    """ 
    For plausibility questions, use C{index} where index is the id of 
    implausible story (from meta_question field)
    """
    # hack to make sure we can call `any`
    if np.array([pd.isna(row["question_meta"])]).any():
        return row["example_id"]
    else:
        _, example_id, _ = row["question_meta"][0]
        return example_id

def calc_trip_metrics(model_output_path: str, 
                        trip_data_path: str,
                        trip_model_results_path: str
                        ):
    """_summary_

    Args:
        model_output_path (str): _description_
        trip_model_results_path (str): _description_
        trip_data_path (str): _description_
    """
    res_df = pd.read_json(str(model_output_path), lines=True)

    data_df = pd.read_json(str(trip_data_path), lines=True)

    # get actual example_id based on original data
    data_df["actual_example_id"] = data_df.apply(get_actual_example_id, axis=1)
    res_df["example_id"] = data_df["actual_example_id"]

    # by instance metrics for task 1,2 (plaus + conflict)
    inst_res = pd.DataFrame(list(res_df.apply(calc_instance_metrics, axis=1)))

    num_stories = len(inst_res.example_id.unique())

    conf_df = inst_res[inst_res.type == "conflict"]
    plaus_df = inst_res[inst_res.type == "plausibility"]
    assert((len(conf_df) == len(plaus_df)) & (len(conf_df) == num_stories)), "Should be same number of conflict and plausibility instances as total"

    merged = pd.merge(conf_df[["example_id","correct_confl"]], plaus_df[["example_id","correct_plaus"]])
    assert(len(merged) == num_stories)

    # tasks 1 and 2
    num_plaus_correct = merged.correct_plaus.sum()
    num_consistent_correct = merged.correct_confl.sum()
    num_consistent_and_plausible = (merged[(merged.correct_confl) & (merged.correct_plaus)]).correct_confl.sum()

    ## task 3
    # get consistent ids
    const_ids = list(conf_df[conf_df.correct_confl]["example_id"])
    paper_results = read_paper_verifiability_results_all(trip_model_results_path, 
            include_loc_attributes=True)

    model_res_new = model_eval_verifiability(
        evaluated_file_name=trip_data_path,
        output_file_name=model_output_path,
        paper_verifiability_results=paper_results["verifiability_results"],
        paper_consistent_examples=const_ids,
        include_loc_attributes=True
    )
    
    num_verif_correct = model_res_new["total_verifiable_stories"]
    
    # checks that calculations line up
    assert(model_res_new["total_consistent_stories"] == num_consistent_correct)
    ver_const_ids = model_res_new["verifiable_consistent_stories"]
    ver_ids = [ex_id in ver_const_ids for ex_id in list(merged.example_id)]
    merged["verified_consistent"] = ver_ids
    assert ((merged.correct_confl) & (merged.verified_consistent)).sum() == num_verif_correct
    
    if num_consistent_and_plausible > 0:
        pct_verif_from_consistent_and_plaus = round(num_verif_correct/num_consistent_and_plausible, 2)
    else:
        pct_verif_from_consistent_and_plaus = 0



    
    
    tasks_metrics = {
        "num_stories": num_stories,
        "num_plaus_correct": num_plaus_correct,
        "num_consistent_correct": num_consistent_correct,
        "num_verif_correct": num_verif_correct,
        "num_plaus_consistent_correct": num_consistent_and_plausible,
        "pct_plausible_correct": round(num_plaus_correct/num_stories, 4),
        "pct_consistent_correct": round(num_consistent_correct/num_stories, 4),
        "pct_consistent_and_plausible_correct": round(num_consistent_and_plausible/num_stories, 4),
        "pct_verif_correct": round(num_verif_correct/num_stories, 4),
        "pct_verif_from_consistent_and_plaus": pct_verif_from_consistent_and_plaus

    }

    return tasks_metrics

def trip_post_hoc_analysis(config):
    """
    Does post-hoc analysis on the situation models 

    :param config: the global configuration  
    """
    metrics = {}

    ### check for output files and does additional analysis 
    for split,out_file,fname in [
            ("dev",f"{config.dev_name}_eval.jsonl",config.dev_name),
            ("test",f"{config.test_name}_eval.jsonl",config.test_name),
        ]:

        ### original data file
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

        # get baseline results file path
        roberta_results_path = ROBERTA_RESULTS.get(split)

        split_metrics = calc_trip_metrics(model_output_path=full_path,
                                    trip_data_path=orig_path,
                                    trip_model_results_path=roberta_results_path)
        metrics[split] = split_metrics
    
    # flatten nested dict to one level
    final_metrics = {}
    for split, value_dict in metrics.items():
        for k,v in value_dict.items():
            new_k = f"{split}_trip_metrics/{k}"
            final_metrics[new_k] = v

    return final_metrics



