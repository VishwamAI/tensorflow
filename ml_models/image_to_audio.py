import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class ImageToAudio:
    def __init__(self):
        # Initialize the SoundStream model as None for lazy loading
        self.soundstream_model = None
        self.captioning_model = None

    def load_soundstream_model(self):
        if self.soundstream_model is None:
            try:
                self.soundstream_model = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')
            except Exception as e:
                print(f"Error loading SoundStream model: {e}")
        return self.soundstream_model

    def load_image_captioning_model(self):
        if self.captioning_model is None:
            try:
                self.captioning_model = hub.KerasLayer('https://tfhub.dev/google/imagenet/inception_v3/classification/5')
            except Exception as e:
                print(f"Error loading image captioning model: {e}")
        return self.captioning_model

    def calculate_spectrogram(self, samples):
        SAMPLE_RATE = 16000
        N_FFT = 1024
        HOP_LENGTH = 320
        WIN_LENGTH = 640
        N_MEL_CHANNELS = 128
        MEL_FMIN = 0.0
        MEL_FMAX = int(SAMPLE_RATE // 2)
        CLIP_VALUE_MIN = 1e-5
        CLIP_VALUE_MAX = 1e8

        MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MEL_CHANNELS,
            num_spectrogram_bins=N_FFT // 2 + 1,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=MEL_FMIN,
            upper_edge_hertz=MEL_FMAX)

        fft = tf.signal.stft(
            samples,
            frame_length=WIN_LENGTH,
            frame_step=HOP_LENGTH,
            fft_length=N_FFT,
            window_fn=tf.signal.hann_window,
            pad_end=True)
        fft_modulus = tf.abs(fft)

        output = tf.matmul(fft_modulus, MEL_BASIS)
        output = tf.clip_by_value(output, clip_value_min=CLIP_VALUE_MIN, clip_value_max=CLIP_VALUE_MAX)
        output = tf.math.log(output)
        return output

    def image_to_audio(self, image: tf.Tensor) -> tf.Tensor:
        """
        Convert image to audio by first generating a text description of the image
        and then converting the text description to audio using the SoundStream model.

        :param image: Input image tensor.
        :return: Generated audio tensor.
        """
        if not isinstance(image, tf.Tensor):
            raise ValueError("Input image must be a tensor.")

        # Load the image captioning model
        captioning_model = self.load_image_captioning_model()
        try:
            # Generate a caption for the input image
            predictions = captioning_model(image)
            predictions_np = predictions.numpy()
            # Map the predicted class indices to their corresponding labels
            top_prediction = np.argmax(predictions_np, axis=-1)
            caption = str(top_prediction)
        except Exception as e:
            print(f"Error during image captioning: {e}")
            return tf.zeros([1, 16000])  # Return a placeholder tensor on error

        # Convert the caption to a mel-spectrogram
        samples = tf.constant([caption], dtype=tf.float32)
        spectrogram = self.calculate_spectrogram(samples)

        # Use the SoundStream model to convert the mel-spectrogram to audio
        model = self.load_soundstream_model()
        try:
            audio = model(spectrogram)
            return tf.expand_dims(audio, axis=0)  # Ensure shape is (1, 16000)
        except Exception as e:
            print(f"Error during image-to-audio conversion: {e}")
            return tf.zeros([1, 16000])  # Return a placeholder tensor on error
