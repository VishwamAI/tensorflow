from transformers import T5Config, TFAutoModel, AutoTokenizer

def setup_t5_model():
    # Load model configuration from config.json
    config = T5Config.from_json_file("config.json")

    # Initialize the model architecture using from_config method
    model = TFAutoModel.from_config(config)

    # Load tokenizer configuration from tokenizer_config.json
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_config.json")

    print("Model architecture and tokenizer setup completed successfully.")

    return model, tokenizer

if __name__ == "__main__":
    setup_t5_model()
