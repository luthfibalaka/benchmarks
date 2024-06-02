import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from pipeline.pipeline_initializer import initialize_pipeline
from pipeline.prompting_interface import prompt_pipeline
from tqdm import tqdm


# Adjust the path of the model
pipe = initialize_pipeline("models/qwen-14b", torch.bfloat16)

def get_eval_prompt(question: str, answer: str):
    """Return a prompt string for zero shot scenario
    
    ## Parameters:
        - question (str): An question about the dataset
        - answer (str): An answer for the question
    """
    return f"""Question Q:
/*
{question}
*/
Answer A:
/*
{answer}
*/
Assume that the answerer has all the necessary information to respond to question Q. Evaluate answer A based on the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of question Q.
2. Relevance: The answer must directly provide the information requested in question Q without any extraneous details.
If the answer satisfies both criteria, label it as 'good'. If it fails to meet one or both criteria, label it as 'bad'. Provide your evaluation in the following format:
- Label: [good/bad]
- Reasoning: [Provide a brief explanation for your label]"""

# Adjust the name of the benchmark to be evaluated
benchmark_name = "openhermes-direct-nucleus_0.95.csv"
evals = pd.read_csv(benchmark_name)
for i in tqdm(range(evals.shape[0])):
    table = evals["T"][i]
    question = evals["Q"][i]
    answer = evals["A"][i]
    curr_eval = evals["E"][i]
    if curr_eval == "unknown":
        prompt = get_eval_prompt(question, answer)
        conversation = [{"role": "user", "content": prompt}]
        model_output = prompt_pipeline(pipe, conversation)[-1]["content"]
        if "label: good" in model_output.lower() or "label: [good]" in model_output.lower():
            evaluation = "good"
        elif "label: bad" in model_output.lower() or "label: [bad]" in model_output.lower():
            evaluation = "bad"
        else:
            evaluation = model_output
        evals.loc[i, "E"] = evaluation
        evals.loc[i, "R"] = model_output
        evals.to_csv(benchmark_name, index=False)
