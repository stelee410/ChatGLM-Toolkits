import config
from transformers import AutoTokenizer, AutoConfig # type: ignore
from utils import convert_json2ds

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)
autoconfig = AutoConfig.from_pretrained(config.MODEL_PATH,trust_remote_code=True, device_map="auto")

convert_json2ds(tokenizer, autoconfig, config.TUNING_DATA_PATH, config.DATASET_PATH)