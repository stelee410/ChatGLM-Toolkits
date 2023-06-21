from config import MODEL_PATH
from infering import get_model, ChatBot

model, tokenizer = get_model(MODEL_PATH)
chatbot = ChatBot(model, tokenizer)
print(chatbot.predict("小李体检发现P-R间期延长，伴有T波低平和ST段异常，应该怎么办"))
