#This is the global configuration file for the application
#It contains the configuration for the model, the dataset and the various paths
import os

WORKPLACE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKPLACE_PATH, 'model')
TUNING_DATA_PATH = os.path.join(WORKPLACE_PATH, 'tuning_data')
DATASET_PATH = os.path.join(WORKPLACE_PATH, 'dataset')
LORA_PATH = os.path.join(WORKPLACE_PATH, 'lora')
LOAD_8BIT = True

MAX_SEQ_LEN = 384

