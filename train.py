
import argparse
from config import MODEL_PATH, DATASET_PATH,LORA_PATH
from infering import ChatBot
from transformers import AutoTokenizer # type: ignore

parser = argparse.ArgumentParser(description= "Train on a model on a dataset")
parser.add_argument('--model_path', type = str, default=MODEL_PATH)
parser.add_argument('--dataset_path', type = str, default=DATASET_PATH)
parser.add_argument('--saved_path', type = str,default=LORA_PATH)
parser.add_argument('--command', type = str, default="train")
parser.add_argument('--simple_test', type = bool, default=False)

def main():
    args = parser.parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path
    saved_path = args.saved_path
    command = args.command
    simple_test = args.simple_test

    if command == "dry": #dry run
        print("This is just dry run")
    elif command == "train":
        print(f'train model of {model_path} on dataset {dataset_path} and saved on {saved_path}')
        from tuning import training
        trainer = training(model_path, dataset_path)
        trainer.model.save_pretrained(saved_path)
        if simple_test:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            chatbot = ChatBot(trainer.model, tokenizer)
            print(chatbot.predict("小李体检发现P-R间期延长，伴有T波低平和ST段异常，应该怎么办"))
    else:
        print(f'Unknown command: {command}')
        

if __name__ == "__main__":
    main()

