import unittest
import tensorflow as tf
import numpy as np
from ml_models.data_type_conversions import DataTypeConversions

class TestDataTypeConversions(unittest.TestCase):
    def setUp(self):
        self.converter = DataTypeConversions()

    def test_text_to_text(self):
        input_text = "Hello, world!"
        result = self.converter.text_to_text(input_text)
        self.assertIsInstance(result, str)

    def test_text_to_image(self):
        input_text = "A cat sitting on a mat."
        result = self.converter.text_to_image(input_text)
        self.assertIsInstance(result, tf.Tensor)

    def test_text_to_video(self):
        input_text = "A dog running in the park."
        result = self.converter.text_to_video(input_text)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape[0], 30)  # Check if 30 frames are generated

    def test_text_to_audio(self):
        input_text = "Hello, this is a test."
        result = self.converter.text_to_audio(input_text)
        self.assertIsInstance(result, tf.Tensor)

    def test_image_to_text(self):
        input_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = self.converter.image_to_text(input_image)
        self.assertIsInstance(result, str)

    def test_image_to_image(self):
        input_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = self.converter.image_to_image(input_image)
        self.assertIsInstance(result, np.ndarray)

    def test_image_to_video(self):
        input_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = self.converter.image_to_video(input_image)
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape[0], 30)  # Check if 30 frames are generated

    def test_image_to_audio(self):
        input_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = self.converter.image_to_audio(input_image)
        self.assertIsInstance(result, tf.Tensor)

if __name__ == '__main__':
    unittest.main()
