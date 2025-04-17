import transformers
import torch
import pandas as pd
from .hflmBase import BaseLM


class HFLM(BaseLM):
    def __init__(self, pretrained_path, batch_size=1, device="cuda", model_alias="hflm_model"):
        super().__init__(model_alias)
        assert isinstance(batch_size, int)
        assert isinstance(device, str)

        if device:
            self.dev = torch.device(device)
        else:
            self.dev = torch.device("cuda") if torch.cuda_is_available() else torch.device("cpu")

        self.lm = transformers.AutoModelForCausalLM.from_pretrained(pretrained_path).to(self.dev)
        self.lm.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_path)
        self.vocab_size = self.tokenizer.vocab_size

        # maintain model meta data
        self.model_path = pretrained_path
        self.batch_size_per_gpu = batch_size
        self.model_alias = model_alias

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.lm.config.n_ctx

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self.dev

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.lm(inps)[0][:, :, : self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.lm.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)


class GPT2(HFLM):
    @property
    def max_length(self):
        try:
            return self.lm.config.seq_length
        except:
            return 256

    @property
    def description(self):
        """Returns a description of the model"""
        return f"{self.model_alias} is a GPT2 language model loaded from {self.model_path}"

    def predict(self, inputArgs, tasktype=None):
        """
        Function that returns the model's prediction for the given input values passed in inputArgs.
        This function needs to be implemented for each model wrapper, otherwise it will raise an exception.
        """
        if type(inputArgs) == pd.core.frame.DataFrame:
            results = []
            for _, row in inputArgs.iterrows():
                results += self.predict(row.to_dict(), tasktype=tasktype)
            return results

        else:
            assert type(inputArgs) == dict, "Incorrect inputArgs provided"

            # create result object with metadata you want to keep from the input
            metadata_fields_to_keep = ["dataset", "id"]
            result = {k: inputArgs[k] for k in metadata_fields_to_keep}

            def convertOptionsListToString(options):
                return "\n".join([ascii_lowercase[i] + ". " + options[i] for i in range(len(options))])

            if tasktype == "multiple choice":
                # input = multiple choice style json object like
                # {'dataset':'testset1-mc', 'id': 'tg123',
                # 'question': 'What is the first letter of the alphabet?', 'options':['a','b','c']}
                optionsString = convertOptionsListToString(inputArgs["options"])
                prompt = f"Question: {question}\nOptions: {optionsString}\nAnswer: "

                # add predictions and confidence to results object
                # generate prediction using prompt above
                result[
                    "prediction"
                ] = None  # ---- TODO: implement passing prompt to model and returning generated prediction -----
                result[
                    "confidences"
                ] = None  # ---- TODO: implement passing prompt to model and returning generated confidences (note, these two lines may separate into at least 3 (1. into a pass to model and generate output, 2&3. assign output to result object) -----

            elif tasktype == "text generation":
                # input = text generation style json object like
                # {'dataset':'testset2-textgen', 'id': 'tg123',
                # 'prompt': 'The first letter of the alphabet is '}
                # generate response and add generated text to results object
                if "max_gen_tokens" not in inputArgs.keys():
                    inputArgs["max_gen_tokens"] = self.max_gen_toks

                prompt = self.tokenizer(inputArgs["prompt"], return_tensors="pt")
                prediction = self.lm.generate(
                    prompt["input_ids"].to(self.device),
                    max_length=inputArgs["max_gen_tokens"],
                    eos_token_id=self.eot_token_id,
                    pad_token_id=self.eot_token_id,
                    attention_mask=prompt["attention_mask"].to(self.device),
                    do_sample=False,
                )

                result["model_alias"] = self.model_alias
                result["pred"] = self.tok_decode(prediction[0])
                result.update(
                    {c: inputArgs[c] for c in ["id", "gold", "dataset", "max_gen_tokens"] if c in inputArgs.keys()}
                )
                return [result]

            else:
                raise Exception(f"{tasktype} not implemented for {self.model_alias}")


class BLOOM(GPT2):
    @property
    def max_length(self):
        try:
            return self.lm.config.seq_length
        except:
            return 4096

    @property
    def description(self):
        """Returns a description of the model"""
        return f"{self.model_alias} is a BLOOM language model loaded from {self.model_path}"


# note: would not need to implement predict function again if using the same prompts as GPT style since BLOOM is a GPT style model
