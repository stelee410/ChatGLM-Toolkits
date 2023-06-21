from transformers import TrainingArguments # type: ignore
import torch

class ArgsBuilder:
  def __init__(self, tokenizer):
    self.pad_token_id = tokenizer.pad_token_id
  def gen_data_collator(self):
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
          ids = ids + [self.pad_token_id] * (longest - ids_l)
          _ids = torch.LongTensor(ids)
          labels_list.append(torch.LongTensor(labels))
          input_ids.append(_ids)
      input_ids = torch.stack(input_ids)
      labels = torch.stack(labels_list)
      return {
          "input_ids": input_ids,
          "labels": labels,
      }
      return data_collator

  def gen_training_args(self):
    return TrainingArguments(
          "output",
          fp16 =True,
          gradient_accumulation_steps=1,
          per_device_train_batch_size = 1,
          learning_rate = 1e-4,
          max_steps=1500,
          logging_steps=50,
          remove_unused_columns=False,
          seed=0,
          data_seed=0,
          group_by_length=False,
      )