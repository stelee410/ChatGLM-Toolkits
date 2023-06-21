import json
from tqdm.auto import tqdm
import datasets


MAX_SEQ_LEN = 384
KEY_INSTRUCTION = "instruction"
KEY_INPUT = "input"
KEY_OUTPUT = "output"

KEY_CONTEXT = "context"
KEY_TARGET = "target"

def format_example(example):
    context = f"Instruction: {example[KEY_INSTRUCTION]}\n"
    if example.get("input"):
       context += f"Input: {example[KEY_INPUT]}\n"
       context += "Answer: "
    target = example[KEY_OUTPUT]
    return {"context": context, "target": target}

def preprocess(tokenizer, config, example, max_seq_length):
    context = example[KEY_CONTEXT]
    target = example[KEY_TARGET]
    context_ids = tokenizer.encode(context,max_length = max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target,
                                  max_length = max_seq_length, 
                                  truncation=True, 
                                  add_special_tokens=False)
    input_ids = context_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(context_ids)}

def convert_json2jsonl(source_path, target_path):
  with open(source_path, 'r') as f:
    examples = json.load(f)
  with open(target_path, 'w') as f:
    for example in tqdm(examples, desc = "formatting"):
      toWrite = json.dumps(format_example(example))+"\n"
      f.write(toWrite)

def read_json(tokenizer, config, path, max_seq_length, skip_overlength=False):
  with open(path, "r") as f:
    examples = json.load(f)
  for example in tqdm(examples, desc = "processing"):
    feature = preprocess(tokenizer, config, format_example(example), max_seq_length)
    if skip_overlength and len(feature["input_ids"]) > max_seq_length:
      continue
    feature["input_ids"] = feature["input_ids"][:max_seq_length]
    yield feature

def read_jsonl(tokenizer, config, path, max_seq_length, skip_overlength=False):
  with open(path, "r") as f:
      for line in tqdm(f.readlines()):
        example = json.loads(line)
        feature = preprocess(tokenizer, config, example, max_seq_length)
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
          continue
        feature["input_ids"] = feature["input_ids"][:max_seq_length]
        yield feature

def convert_jsonl2ds(tokenizer, config, source_path, target_path,max_seq_length = 384,skip_overlength = False):
  dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(tokenizer, config, source_path, max_seq_length, skip_overlength)
  )
  dataset.save_to_disk(target_path) # type: ignore

def convert_json2ds(tokenizer, config, json_path, target_path, max_seq_length=384, skip_overlength = False):
  dataset = datasets.Dataset.from_generator(
        lambda: read_json(tokenizer, config, json_path, max_seq_length, skip_overlength)
  )
  return dataset


