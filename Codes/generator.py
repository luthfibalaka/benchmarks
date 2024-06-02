import pandas as pd
import torch
import json

from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline
from utils.csv_data_source import CsvDataSource

# Loading questions and affiliations mapping
with open("resources/questions.json") as file:
    questions = json.load(file)
with open("resources/table_affiliation_mapping.json") as file:
    affiliations = json.load(file)


def get_prompt(dataset: str, question: str):
    """Return a prompt string for direct style

    ## Parameters:
        - dataset (str): The dataset information
        - question (str): A question to be asked about the dataset
    """
    return f"""Given this dataset:
*/
{dataset}
*/
and this question:
/*
{question}
*/
Assume you have all the necessary information to respond to the question. Generate an answer for the question given the dataset satisfying the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of the question.
2. Relevance: The answer must directly provide the information requested in the question without any extraneous details."""


def get_prompt_role_play(affiliation: str, dataset: str, question: str, role: str):
    """Return a prompt string for role play style

    ## Parameters:
        - affiliation (str): The affiliation of the dataset
        - dataset (str): The dataset information
        - question (str): A question to be asked about the dataset
        - role (str): The role that the LLM play
    """
    return f"""Given this dataset:
*/
{dataset}
*/
and this question:
/*
{question}
*/
Assume you are {role} with all the necessary information to respond to the question. Generate an answer for the question given the dataset satisfying the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of the question.
2. Relevance: The answer must directly provide the information requested in the question without any extraneous details."""


# Adjust the name of the models
pipe = initialize_pipeline("models/openhermes-7b", torch.bfloat16)


def process_direct(generation_params={}):
    # Process the tables and questions with standard prompt
    csv_data_source = CsvDataSource("tables")
    benchmark = pd.DataFrame(columns=["T", "Q", "A", "E", "R"])
    for table in iter(csv_data_source):
        csv_file_name = table[0]
        print(f"Processing table {csv_file_name[:-4]}")
        dataset = "".join(table[1]).rstrip()
        for i in questions.keys():
            print(f"Processing question {i}")
            question = questions[i]["question"]
            prompt = get_prompt(dataset, question)
            conversation = [{"role": "user", "content": prompt}]
            answer = prompt_pipeline(pipe, conversation, **generation_params)[-1][
                "content"
            ]
            row = pd.DataFrame(
                {
                    "T": [csv_file_name[:-4]],
                    "Q": [question],
                    "A": [answer],
                    "E": ["unknown"],
                    "R": ["unknown"],
                }
            )
            benchmark = pd.concat([benchmark, row], ignore_index=True)
        print("Table processed!")
    return benchmark


def process_role_play(generation_params={}):
    # Process the tables and questions with role-play prompt
    csv_data_source = CsvDataSource("tables")
    benchmark = pd.DataFrame(columns=["T", "Q", "A", "E", "R"])
    for table in iter(csv_data_source):
        csv_file_name = table[0]
        print(f"Processing table {csv_file_name[:-4]}")
        dataset = "".join(table[1]).rstrip()
        affiliation = affiliations[csv_file_name[7:-4]]
        for i in questions.keys():
            print(f"Processing question {i}")
            question = questions[i]["question"]
            role = questions[i]["role"]
            prompt = get_prompt_role_play(affiliation, dataset, question, role)
            conversation = [{"role": "user", "content": prompt}]
            answer = prompt_pipeline(pipe, conversation, **generation_params)[-1][
                "content"
            ]
            row = pd.DataFrame(
                {
                    "T": [csv_file_name[:-4]],
                    "Q": [question],
                    "A": [answer],
                    "E": ["unknown"],
                    "R": ["unknown"],
                }
            )
            benchmark = pd.concat([benchmark, row], ignore_index=True)
        print("Table processed!")
    return benchmark


# Run the scenarios. For instance, this is for direct style nucleus sampling
benchmark = process_direct({"do_sample": True, "top_p": 0.95, "top_k": 0})
benchmark.to_csv("openhermes-direct-nucleus_0.95.csv", index=False)
