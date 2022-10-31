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
from situation_modeling import (
    ModelRunner,
    initialize_config,
)


BOILER_PLATE = {
    "Enter your story" : "",
    "Enter question (optional)": "",
}

@st.cache(allow_output_mutation=True)
def example_sets():
    return ([
        "...",
        "1. Purchase a blackboard eraser. 2. Keep the blackboard eraser in the glove box or attach it to a spot on or near the car door for easy access. 3. Use the eraser to clean the inner side of the windshield. 4. Replace after use. What is the initial location of eraser?",
    ])

@st.cache(allow_output_mutation=True)
def build_model(config):
    config.keep_old_config = True
    model = ModelRunner.load(config,keep_old_config=True)
    return model


@st.cache(allow_output_mutation=True)
def build_config():
    from situation_modeling.runner import params
    config = initialize_config(sys.argv[1:],params)

    if not config.load_existing and not config.wandb_model:
        raise ValueError(
            'Must provide model checkpoint via `--load_existing` or `--wandb_model`'
        )
    return config

def run_model(
        model,
        config,
        model_input,
):
    """Runs the model"""
    out = model.query(model_input)

    return pd.DataFrame(
        [out.outputs],
        [0],
        columns=["model output"],
    )
                

def main():
    st.title("Generic Text2Text viewer")
    ex_inputs = example_sets()
    
    ex_s = st.selectbox("Select example:",ex_inputs,index=0)
    argument_filler = "Enter your input"

    ##
    config = build_config()
    model = build_model(config)
    
    if ex_s != "...":
        argument_filler = ex_s

    text_input = st.text_area(
        "Enter some input text",
        argument_filler,
        height=300
    )

    submit = st.button("Generate Text")

    if submit and (text_input != "..." or text_input != "Enter your input"):
        with st.spinner("Processing..."):

            ### run the model
            pd.set_option('max_colwidth', 10)
            
            out_dframe = run_model(
                model,
                config,
                text_input,
            )

            st.write(out_dframe)
            
if __name__ == "__main__":
    main()
