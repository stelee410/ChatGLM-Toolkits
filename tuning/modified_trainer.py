from transformers import Trainer,TrainingArguments # type: ignore
import os
import torch
import json



class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir="", _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

    def save_peft_config(self,output_dir=""):
        peft_config = self.model.peft_config['default']
        output_dict = peft_config.__dict__
        output_path = os.path.join(output_dir, "adapter_config.json")
        with open(output_path, "w") as writer:
          writer.write(json.dumps(output_dict, indent=2, sort_keys=True))