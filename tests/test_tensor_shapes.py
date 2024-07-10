import tensorflow as tf
import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_models.data_type_conversions import DataTypeConversions, Processor

# Create an instance of the DataTypeConversions class
converter = DataTypeConversions()

# Define a test text input
test_text = "This is a test."

# Run the text_to_audio conversion and print tensor shapes
try:
    # Print the shape of the input tensor
    processor = Processor()
    input_ids = processor.text_to_sequence(test_text)
    input_tensor = tf.constant([input_ids], dtype=tf.int32)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Load models
    tacotron2 = converter.load_text_to_audio_model()
    mb_melgan = converter.load_mb_melgan_model()
    pqmf = converter.load_pqmf_model()

    # Generate mel spectrograms and print shape
    _, mel_outputs, _, _ = tacotron2.inference(
        input_tensor,
        tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
    print(f"Mel outputs shape: {mel_outputs.shape}")

    # Synthesize audio subbands and print shape
    generated_subbands = mb_melgan(mel_outputs)
    print(f"Generated subbands shape: {generated_subbands.shape}")

    # PQMF synthesis and print shape
    audio = pqmf.synthesis(generated_subbands)[0, :-1024, 0]
    print(f"Audio shape before padding: {audio.shape}")

    # Ensure the audio tensor has the correct shape
    audio = tf.pad(audio, [[0, max(0, 16000 - tf.shape(audio)[0])]])  # Pad if necessary
    audio = audio[:16000]  # Trim if necessary
    audio = tf.expand_dims(audio, axis=0)  # Ensure shape is (1, 16000)
    print(f"Final audio tensor shape: {audio.shape}")
except Exception as e:
    print(f"Error during text-to-audio conversion: {e}")
