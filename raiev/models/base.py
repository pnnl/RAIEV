class Model:
    def __init__(self, model_alias):
        self.model_alias = model_alias

    @property
    def description(self):
        """Returns a description of the model
        (e.g., "GPT2 is a language model implemented using the
         pretrained implementation from huggingface")
        """
        return '"model" base class'

    def predict(self, inputArgs):
        """
        Function that returns the model's prediction for the given input values passed in inputArgs.
        This function needs to be implemented for each model wrapper, otherwise it will raise an exception.
        """
        raise Exception("predict() function needs to be implemented")
