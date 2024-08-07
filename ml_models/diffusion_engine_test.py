import tensorflow as tf
import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_models.diffusion_engine import DiffusionEngine

class CustomSampler(tf.keras.Model):
    def __init__(self):
        super(CustomSampler, self).__init__()
        self.dense = tf.keras.layers.Dense(784, activation='relu')

    def sample(self, num_samples, conditioning=None):
        noise = tf.random.normal([num_samples, 100])
        samples = self.dense(noise)
        samples = tf.reshape(samples, (-1, 28, 28, 1))
        return samples

def test_diffusion_engine():
    # Sample configuration for the DiffusionEngine
    network_config = {
        "target": "tensorflow.keras.Sequential",
        "params": {
            "layers": [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ]
        }
    }
    denoiser_config = {
        "target": "tensorflow.keras.Sequential",
        "params": {
            "layers": [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ]
        }
    }
    first_stage_config = {
        "target": "tensorflow.keras.Sequential",
        "params": {
            "layers": [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(784, activation='relu'),
                tf.keras.layers.Reshape((28, 28, 1))
            ]
        }
    }
    conditioner_config = {
        "target": "tensorflow.keras.Sequential",
        "params": {
            "layers": [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')
            ]
        }
    }
    sampler_config = {
        "target": "diffusion_engine_test.CustomSampler",
        "params": {}
    }
    optimizer_config = {
        "target": "tensorflow.keras.optimizers.Adam",
        "params": {
            "learning_rate": 0.001
        }
    }
    loss_fn_config = {
        "target": "tensorflow.keras.losses.SparseCategoricalCrossentropy",
        "params": {
            "from_logits": True
        }
    }

    # Instantiate the DiffusionEngine
    diffusion_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        optimizer_config=optimizer_config,
        loss_fn_config=loss_fn_config,
        use_ema=True
    )

    # Create dummy input data
    dummy_input = tf.random.normal([1, 28, 28, 1])
    dummy_batch = {"labels": tf.constant([1])}

    # Test the call method
    loss, loss_dict = diffusion_engine((dummy_input, dummy_batch), training=True)
    print("Loss:", loss.numpy())
    print("Loss Dict:", loss_dict)

    # Test the encode and decode methods
    encoded = diffusion_engine.encode_first_stage(dummy_input)
    decoded = diffusion_engine.decode_first_stage(encoded)
    print("Encoded shape:", encoded.shape)
    print("Decoded shape:", decoded.shape)

    # Test the sample method
    samples = diffusion_engine.sample(num_samples=1)
    print("Samples shape:", samples.shape)

    # Test the logging methods
    diffusion_engine.log_images({"sample_image": dummy_input}, step=1)
    diffusion_engine.log_conditionings({"sample_conditioning": 0.5}, step=1)

if __name__ == "__main__":
    test_diffusion_engine()
