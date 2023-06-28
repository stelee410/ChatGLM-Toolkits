from config import MODEL_PATH, LORA_PATH
from infering import get_peft_model, ChatBot

if __name__ == "__main__":
    print("=== traing by pre-config ===")
    model, tokenizer = get_peft_model(MODEL_PATH, LORA_PATH)
    chatbot = ChatBot(model, tokenizer)
    while True:
        q = input(">>> ")
        if q == 'q!':
            break
        else:
            print(chatbot.predict(q))
    