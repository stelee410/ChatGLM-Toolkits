from config import MODEL_PATH, LORA_PATH
from infering import get_peft_model, ChatBot

model, tokenizer = get_peft_model(MODEL_PATH, LORA_PATH)
chatbot = ChatBot(model, tokenizer)
print(chatbot.predict("你好"))