import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from ml_models.base_model import BaseModel

class TestBaseModel(tf.test.TestCase):
    def setUp(self):
        self.input_data = tf.random.uniform((10, 10), minval=0, maxval=1, dtype=tf.float32)
        self.model_svd = BaseModel(apply_svd=True)
        self.model_fft = BaseModel(apply_fft=True)

    def test_svd_operation(self):
        output = self.model_svd.advanced_math_operations(self.input_data)
        self.assertEqual(output.shape, self.input_data.shape)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))

    def test_fft_operation(self):
        output = self.model_fft.advanced_math_operations(self.input_data)
        self.assertEqual(output.shape, self.input_data.shape)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))

if __name__ == "__main__":
    tf.test.main()
