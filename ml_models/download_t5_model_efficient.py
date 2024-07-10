from transformers import TFAutoModel, AutoTokenizer
import os

def download_t5_model_efficient():
    model_name = "google/t5-v1_1-xl"
    model_dir = os.path.join(os.getcwd(), model_name.replace("/", "_"))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = TFAutoModel.from_pretrained(model_name, cache_dir=model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

    print(f"Model and tokenizer for {model_name} downloaded successfully to {model_dir}.")

if __name__ == "__main__":
    download_t5_model_efficient()
