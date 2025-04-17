from . import hflm

MODEL_REGISTRY = {
    "hf": hflm.HFLM,
    "gpt2": hflm.GPT2,
    "bloom": hflm.BLOOM,
}


def get_model(model_type, model_path, batch_size=1, device="cuda", model_alias="model"):
    """Load and return model.

    :param model_type: str
        Type of model to load, refers to name in MODEL_REGISTRY dictionary.
    :param model_path: str
        Path to trained model directory or huggingface model name.
    :param batch_size: int
        Batch size for model.
    :param device: str
        PyTorch device for running models.
    :param device: model_alias
        Alias or short name of model
    """

    try:
        model = MODEL_REGISTRY[model_type]
    except KeyError:
        # return exception if model type not found in MODEL_REGISTRY (since we will consider more than just LM models)
        raise Exception(f"{model_type} not supported")
        # TODO: update or remove default model running.
        print("Invalid model. Running hugging face default.")
        model = MODEL_REGISTRY["hf"]

    return model(model_path, **{"batch_size": batch_size, "device": device, "model_alias": model_alias})
