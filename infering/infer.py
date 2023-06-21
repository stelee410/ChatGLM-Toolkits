from transformers import AutoModel, AutoTokenizer # type: ignore
from peft import PeftModel
import torch

def get_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/ChatGLM-6B", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/ChatGLM-6B", trust_remote_code=True,load_in_8bit=True,device_map="auto")
    model.eval() #model to freeze
    return model, tokenizer

def get_peft_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/ChatGLM-6B", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/ChatGLM-6B", trust_remote_code=True,load_in_8bit=True,device_map="auto")
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer

class ChatBot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def quick_chat(self, prompt, history=[]):
        response, history = self.model.chat(self.tokenizer, prompt,[])
        return response, history
    def predict(self,prompt, temperature = 0):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        with torch.no_grad():
            input_text =  f"Instruction: {prompt}\nAnswer: "
            input_length = len(input_text)
            ids = self.tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids])
            input_ids = input_ids.to(device)
            out = self.model.generate(
                input_ids=input_ids,
                max_length=150,
                do_sample=False,
                temperature=temperature
            )
            out_text = self.tokenizer.decode(out[0])
            return out_text[input_length:]
    

