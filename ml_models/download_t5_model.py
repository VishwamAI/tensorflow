from transformers import TFAutoModel, AutoTokenizer

def download_t5_model():
    model_name = "google/t5-v1_1-xl"
    model = TFAutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model and tokenizer for {model_name} downloaded successfully.")

if __name__ == "__main__":
    download_t5_model()
