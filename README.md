Situation Modeling
======================

Code base for situation modeling projects and other general-purpose
tools for NLU based on [**huggingface transformers**](https://github.com/huggingface/transformers).

For details about our EMNLP 2022 paper on breakpoint modeling, see
`etc/emnlp_2022_scripts` (more details forthcoming). 

## Basic setup
----------------------------

We suggest using [**conda**](https://docs.conda.io/en/latest/miniconda.html) for creating a python environment, and doing the following:
```bash
conda create -n situation_modeling python=3.8
conda activate situation_modeling ## after setting up above
pip install -r requirements.txt
```

Main run command:
```bash
./run.sh mode [options] --help
```

## Running a vanilla Text2Text model 
----------------------------

Below is an example of how to train a small **multi-angle** T5 model
on a multi-angle version of bAbi. This can be down by running the
following:
```bash
./run.sh runner \
         --output_dir _runs/babi_run/ \ #<-- output dir
         --data_dir etc/data/multi_angle_example/ \ #<-- data loc
         --dev_eval \ #<-- run dev and train eval
         --train_eval \
         --base_model "t5" \
         --model_name_or_path "t5-base" \ #<-- transformer details
         --tokenizer_name_or_path "t5-base" \
         --model_type "text2text" \ #<-- type of architecture
         --max_seq_length "200" \  #<-- max seq length
         --max_output_length "50" \ #<-- max output length
         --learning_rate "5e-5" \
         --train_batch_size 16 \
         --max_data "2000" \
         --print_output \ #<-- print output to file
         --num_train_epochs "12" \ 
         --early_stopping \ #<-- early stopping
         --patience "2"
```
To see the full list of options, run
```bash
./run.sh runner --help
```
**other models** (e.g., `t5-large`, `bart`) can be used by changing
`--model_name_or_path` and `--tokenizer_name_or_path` (this has mostly
been trained with T5)

# Data format
see example in `etc/data/multi_angle_example`:
```json
{
    "id": "5812_valid_sub_question_question_gen_0",
    "context": "$answer$ football $context$ John took the football there. $story$ Sandra picked up the milk there. John took the football there.",
    "answer": "What is John carrying?", "meta":
    {"prefix": "question:"}
}
```
`id` is the identifier, `context` is the main input, `answer` is the
main output, `meta["prefix"]` is an identifier that specifies the
*angle* (this can be left blank if just doing standard
translation). By convention the prefix `answer:` is used for cases
where exact match accuracy needs to be measured. 

# Data output
Running the code above produces two files in `--output_dir`,
`train_eval.jsonl` and `dev_eval.jsonl`, which look like the
following:
```json
{
    "id": "4240_valid_sub_question_question_gen_3",
    "context": "question: $answer$ kitchen $context$ Mary went to the
    kitchen. $story$ Jeff moved to the bedroom. Sandra went to the
    bathroom. Julie moved to the park. Jeff journeyed to the
    office. John moved to the school. Mary went to the kitchen. Sandra
    picked up the football. Sandra journeyed to the kitchen. Sandra handed the football to Mary.",
    "gen_out": "Where is the football?",
    "answer": "Where is the football?",
    "meta": {},
    "prefix": "question:"
}
```
where `gen_out` is the model translated output. 

# Loading a checkpoint in python
```python
```python
## create initial config 
from situation_modeling import ModelRunner,get_config
config = get_config("runner")

## specify target checkpoint
config.load_existing = "_runs/babi_run/epoch=6-step=223.ckpt"

## load model and checkpoint
model = ModelRunner.load(config,keep_old_config=False)

## query the model via text
out = model.query("question: $answer$ football $context$ John took the football there. $story$ Sandra picked up the milk there. John took th football there.")
print(out.outputs)
## => ['What is John carrying?']

### playing with model hyper-parameters
### can be changed (not tested!)
model.modify_config("num_beams",1)  #<-- change beam size for decoder
model.modify_config("max_output_length",2) <-- change output length
out = model.query("question: $answer$ football $context$ John took the football there. $story$ Sandra picked up the milk there. John took th football there.")
#->['What'] 
```
