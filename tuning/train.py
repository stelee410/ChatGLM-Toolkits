from . import prepare_model, prepare_peft_model, ModifiedTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoConfig # type: ignore
import datasets
import torch
from torch.utils.tensorboard import SummaryWriter #type: ignore
from transformers.integrations import TensorBoardCallback

def training(model_path, dataset_path):
  model = prepare_peft_model(prepare_model(model_path))
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
  dataset = datasets.load_from_disk(dataset_path)
  training_args = TrainingArguments(
          "output",
          fp16 =True,
          gradient_accumulation_steps=1,
          per_device_train_batch_size = 1,
          learning_rate = 2e-5,
          max_steps=1500,
          logging_steps=50,
          num_train_epochs=35,
          remove_unused_columns=False,
          seed=6,
          data_seed=0,
          group_by_length=False,
      )
  def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }
  writer = SummaryWriter()
  trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args= training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
  trainer.train()
  writer.close()
  return trainer