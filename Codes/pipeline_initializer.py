from transformers import pipeline


def initialize_pipeline(model_path: str, torch_dtype):
    """
    Initialize a text generation pipeline

    ### Parameters:
    - model_path (str): The path of a model and tokenizer's weights.

    ### Returns:
    - pipe (TextGenerationPipeline): The pipeline for text generation.
    """

    pipe = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype=torch_dtype
    )

    return pipe
