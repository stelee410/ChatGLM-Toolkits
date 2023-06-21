from config import MODEL_PATH
from infering import get_model, ChatBot

model, tokenizer = get_model(MODEL_PATH)
chatbot = ChatBot(model, tokenizer)
print(chatbot.predict("你好"))
