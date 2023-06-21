from config import MODEL_PATH,LORA_PATH
from infering import get_peft_model, get_model, ChatBot

def get_ori():
    model, tokenizer = get_model(MODEL_PATH)
    chatbot = ChatBot(model, tokenizer)
    return chatbot

def get_peft():
    model, tokenizer = get_peft_model(MODEL_PATH, LORA_PATH)
    chatbot = ChatBot(model, tokenizer)
    return chatbot