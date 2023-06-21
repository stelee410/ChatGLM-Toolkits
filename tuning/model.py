from utils import CastOutputToFloat
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel # type: ignore

def prepare_model(model_path):
  model = AutoModel.from_pretrained(model_path, trust_remote_code=True,load_in_8bit=True,device_map="auto")
  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()
  model.is_parallelizable = True
  model.model_parallel = True
  model.lm_head = CastOutputToFloat(model.lm_head)
  model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
  )
  return model

def prepare_peft_model(model):
  peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
  )
  model = get_peft_model(model, peft_config)
  return model