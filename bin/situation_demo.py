import streamlit as st 
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import shortuuid
import wandb
sys.path.append('.')
from optparse import OptionParser,OptionGroup
from situation_modeling.register import load_external_project

from situation_modeling import (
    ModelRunner,
    initialize_config,
)


BOILER_PLATE = {
    "Enter your story" : "",
    "Enter question (optional)": "",
}

CLS_MAP = {
    0: "no",
    1: "maybe",
    2: "yes"
    }

@st.cache(allow_output_mutation=True)
def example_sets():
    return ([
        "...",
        "John went back to the school.\nFred grabbed the milk.\nThen he dropped the milk.\nMary is either in the bedroom or the office.\nBill and Fred journeyed to the office.\nAfterwards they went back to the park.\nBill took the football.\nAfter that he moved to the bathroom.\nBill took the apple.\nAfter that he went back to the bedroom.",
        "Daniel got the apple.\nJeff and Mary went back to the bedroom.\nJulie is no longer in the office.\nMary took the football.\nMary handed the football to Jeff.\nJeff passed the football to Mary.\nJulie is either in the garden or the kitchen.\nMary gave the football to Jeff.\nJeff dropped the football.\nMary is either in the garden or the school."
    ])

@st.cache(allow_output_mutation=True)
def example_propositions():
    return ([
        "...",
        "John at bathroom. John at school\nFred holds milk\nFred holds milk\nMary at office. Mary at bedroom. Mary at school\nBill at bedroom. Fred at office. Bill at office\nFred at park. Bill at park. Bill at hallway\nBill holds football\nBill at bathroom. Bill at park. football at bathroom\nBill holds apple\napple at bedroom. Bill at bedroom. football at bedroom",
        "Daniel holds apple\nJeff at bedroom. Jeff at kitchen. Mary at bedroom\nJulie at office.\nMary holds football\nMary holds football. Jeff holds football\nJeff holds football. Mary holds football\nJulie at bathroom. Julie at garden. Julie at kitchen\nMary holds football. Jeff holds football\nJeff holds football\nMary at kitchen. Mary at garden. Mary at school"
    ])

@st.cache(allow_output_mutation=True)
def example_questions():
    return ([
        "...",
        "Where is the apple?",
        "Where is the football?"
        ])


@st.cache(allow_output_mutation=True)
def build_model(config):
    model = ModelRunner.load(config,keep_old_config=True)
    return model


@st.cache(allow_output_mutation=True)
def build_config():
    from situation_modeling.runner import params
    config = initialize_config(sys.argv[1:],params)
    config.keep_old_config = True

    ### 
    config.sit_gen = False
    config.external_project = "custom_modules/babi_situation_model"

    if not config.load_existing and not config.wandb_model:
        raise ValueError(
            'Must provide model checkpoint via `--load_existing` or `--wandb_model`'
        )
    if config.external_project:
        print("LOADING EXTERNAL MODULE")
        load_external_project(config.external_project)
    return config

def run_model(
        model,
        config,
        story_text,
        proposition_list,
        question_text,
):
    """Runs the model"""
    parsed_text = [s.strip() for s in story_text.split("\n")]
    parsed_propositions = [[x.strip() for x in s.split('.') if not x.strip() == ""] for s in proposition_list.split("\n")]

    if len(parsed_propositions) == 1:
        single_prop_set = parsed_propositions[0]
        parsed_propositions = [single_prop_set for n in range(len(parsed_text))]

    input_story = {
        "texts"      : parsed_text,
        "prop_lists" : parsed_propositions
    }
    if question_text.strip() and question_text != "...":
       #input_story["question"] = [f"{} {question_text}"]
       input_story["question"] = [question_text]
       input_story["answer"] = ["unk"]

    situation_out = model.query(input_story).generate_situation_map()[0]
    print(situation_out)


    row_content = []
    simple_row_content = [] # non formatted content, can be logged to wandb table
    index = []

    history = []
    
    for k,event in enumerate(situation_out["events"]):
        #event_description,propositions,predictions,_ = event
        event_description = event[0]
        propositions = event[1]
        predictions = event[2]
        
        history.append(event_description)
        
        index.append(f"[SIT]{k},t={k}")
        
        if config.class_output:
            predictions = [CLS_MAP.get(v) for v in predictions]
            green_colors = [True if p in ["yes", "maybe"] else False for p in predictions]
        else:
            green_colors = [True if p>=0.5 else False for p in predictions]
        
        row_content.append([
            "%s" % "!! %s **%s^^ @@" % (' '.join(history[:k]),history[k]),
            '{ %s }' % ', '.join(
                [f"B(GREEN`{p}`^GREEN)=GREEN{v}^GREEN" if is_green else \
                    f"B(RED`{p}`^RED)=RED{v}^RED" \
                     for p,v,is_green in zip(propositions,predictions,green_colors)
        ])])
        simple_row_content.append([index[k],
            "%s" % (history[k]),
            '{ %s }' % ', '.join(
                [f"B({p})={v}" if is_green else \
                    f"B({p})={v}" \
                     for p,v,is_green in zip(propositions,predictions,green_colors)
        ])])

    return (parsed_text,
                parsed_propositions,
                pd.DataFrame(
                    row_content,
                    index=index,
                    columns=["event and breakpoint","belief state"]),
                pd.DataFrame(
                    simple_row_content,
                    index=index,
                    columns=["timestep","event","belief state"])
                )
                

def _process_html(raw_html):
    html_out = raw_html.replace("^RED",'</span>')
    html_out = html_out.replace("^GREEN",'</span>')
    html_out = html_out.replace("RED",'''<span style="color:red;font-family:'Courier New'">''')
    html_out = html_out.replace("GREEN",'''<span style="color:green; font-family:'Courier New'">''')
    html_out = html_out.replace("**","<b>").replace("^^","</b>")
    html_out = html_out.replace("!!",'''<p style="font-family:'Helvetica'; font-size:15px">''')
    html_out = html_out.replace("@@","</p>")

    return html_out 
    

def main():
    st.title("Breakpoint Transformer")
    ex_inputs = example_sets()
    ex_props = example_propositions()
    ex_q = example_questions()
    
    ex_s = st.selectbox("Select example:",ex_inputs,index=0)
    story_id = -1
    argument_filler = "Enter your input"
    prop_filler    = "Enter some text propositions"
    q_filler = "Ask a question"

    ##
    config = build_config()
    model = build_model(config)
    
    wandb_log = False
    dataframe = None
    
    if config.log_streamlit_wandb:
        wandb_log = True
        strmlit_wandb_run = wandb.init(job_type="eval",project="streamlit", 
                                       entity=config.wandb_entity,
                                       name=config.wandb_name,
                                       config=config)
        
        

    if ex_s != "...":
        argument_filler = ex_s
        story_id = ex_inputs.index(ex_s)
        prop_filler = ex_props[story_id]
        q_filler = ex_q[story_id]


    story_text = st.text_area(
        "Narrative text (each step in story delimited by `\\n`)",
        argument_filler,
        height=300
    )
    proposition_list = st.text_area(
        "Step propositions (each step delimited `\\n`, line propositions delimited by `. `). To track a proposition through time, enter single line.",
        prop_filler,
        height=300
        
    )
    question_box = st.text_area('Question')
    
    # setup stremlit logging if specified
    if wandb_log:
        log_check = st.checkbox('Log results to wandb')
    

    submit = st.button("Inspect Beliefs")
    
    

    if submit and (story_text != "..." or story_text != "Enter your input") and proposition_list:
        with st.spinner("Processing..."):

            ### run the model
            pd.set_option('max_colwidth', 10)
            
            text,props,dataframe, simple_df = run_model(
                model,
                config,
                story_text,
                proposition_list,
                #None,
                question_box
            )
            dataframe.style.set_properties(subset=["story breakpoint and context"], **{'width': '20px'})

            #print(dataframe.to_html())
            #st.markdown(dataframe.to_markdown())
            html_out = dataframe.to_html(
                justify='left',
                bold_rows=False,
                col_space=2
            )
            html_out = _process_html(html_out) 

            ### illustrate raw input 
            st.markdown("""**Raw model input with *breakpoints***: <p style="font-family:'Helvetica'"> %s <b>[SIT]</b>%s</p>""" %\
                         (' <b>[SIT]</b>'.join([t if l == 0 else "%s %s" % (l-1,t) for l,t in enumerate(text)]),len(text)-1),
                         unsafe_allow_html=True)

            ## illustrate propositions
            st.markdown(
            """**Breakpoint propositions**: <br> %s""" %\
                str('\n'.join(["<b>[SIT]</b>%d=%s<br>" % (k,str(["<b>PROP</b> %s" % p for p in plist]).replace("[","{").replace("]","}").replace("PROP","[PROP]")) \
                                   for k,plist in enumerate(props)])),unsafe_allow_html=True)
            
            ### main output
            st.write("<b>belief states</b>:",unsafe_allow_html=True)
            st.write(html_out,unsafe_allow_html=True)
        
            if wandb_log and log_check:
                table_uid = shortuuid.ShortUUID().random(length=8)
                print(f"Logging table {table_uid} to wandb...")
                table = wandb.Table(dataframe=simple_df)
                strmlit_wandb_run.log({f"table_{table_uid}": table,
                                       f"text_{table_uid}": story_text,
                                       f"props_{table_uid}": proposition_list})
            
if __name__ == "__main__":
    main()
