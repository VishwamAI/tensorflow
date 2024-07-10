import unittest
import tensorflow as tf
import numpy as np
from ml_models.data_type_conversions import DataTypeConversions

class TestDataTypeConversions(unittest.TestCase):
    def setUp(self):
        self.converter = DataTypeConversions()

    def test_text_to_text(self):
        text = "Hello, world!"
        result = self.converter.text_to_text(text)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_text_to_image(self):
        text = "dog"
        result = self.converter.text_to_image(text)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (1, 256, 256, 3))

    def test_text_to_video(self):
        text = "Hello, world!"
        result = self.converter.text_to_video(text)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (1, 30, 256, 256, 3))

    def test_text_to_audio(self):
        text = "Hallo, Welt!"
        result = self.converter.text_to_audio(text)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (1, 16000))

    def test_image_to_text(self):
        image = tf.random.uniform((1, 299, 299, 3), minval=0, maxval=1, dtype=tf.float32)
        result = self.converter.image_to_text(image)
        self.assertIsInstance(result, str)

    def test_image_to_image(self):
        image = tf.random.uniform((1, 256, 256, 3), minval=0, maxval=1, dtype=tf.float32)
        result = self.converter.image_to_image(image)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 256, 256, 3))

    def test_image_to_video(self):
        image = tf.random.uniform((1, 256, 256, 3), minval=0, maxval=1, dtype=tf.float32)
        result = self.converter.image_to_video(image)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (1, 30, 256, 256, 3))

    def test_image_to_audio(self):
        image = tf.random.uniform((1, 299, 299, 3), minval=0, maxval=1, dtype=tf.float32)
        result = self.converter.image_to_audio(image)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (1, 16000))

if __name__ == "__main__":
    unittest.main()
