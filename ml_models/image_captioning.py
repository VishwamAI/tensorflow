import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_datasets as tfds
import einops
import numpy as np
import matplotlib.pyplot as plt

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self):
        super(ImageCaptioningModel, self).__init__()
        self.feature_extractor = get_feature_extractor()
        self.transformer_decoder = get_transformer_decoder()
        self.tokenizer = get_tokenizer()

    def call(self, inputs, training=False):
        image, caption = inputs
        features = self.feature_extractor(image, training=training)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(caption)[1])
        padding_mask = self.create_padding_mask(caption)
        output = self.transformer_decoder([features, caption, look_ahead_mask, padding_mask], training=training)
        return output

    def generate_caption(self, image, max_length=20, temperature=1.0):
        features = self.feature_extractor(image, training=False)
        caption = [self.tokenizer.start_token]
        for _ in range(max_length):
            look_ahead_mask = self.create_look_ahead_mask(len(caption))
            padding_mask = self.create_padding_mask(tf.convert_to_tensor([caption]))
            output = self.transformer_decoder([features, tf.expand_dims(caption, 0), look_ahead_mask, padding_mask], training=False)
            next_token = tf.argmax(output, axis=-1)[0, -1].numpy()
            caption.append(next_token)
            if next_token == self.tokenizer.end_token:
                break
        return self.tokenizer.detokenize(caption)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def plot_attention_maps(self, image, str_tokens, attention_map):
        fig = plt.figure(figsize=(16, 9))
        len_result = len(str_tokens)
        titles = []
        for i in range(len_result):
            map = attention_map[i]
            grid_size = max(int(np.ceil(len_result / 2)), 2)
            ax = fig.add_subplot(3, grid_size, i + 1)
            titles.append(ax.set_title(str_tokens[i]))
            img = ax.imshow(image)
            ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(), clim=[0.0, np.max(map)])
        plt.tight_layout()

    def run_and_show_attention(self, image, temperature=0.0):
        result_txt = self.generate_caption(image, temperature)
        str_tokens = result_txt.split()
        str_tokens.append('[END]')
        attention_maps = [layer.last_attention_scores for layer in self.transformer_decoder.layers]
        attention_maps = tf.concat(attention_maps, axis=0)
        attention_maps = einops.reduce(attention_maps, 'batch heads sequence (height width) -> sequence height width', height=7, width=7, reduction='mean')
        self.plot_attention_maps(image / 255, str_tokens, attention_maps)
        t = plt.suptitle(result_txt)
        t.set_y(1.05)

# Implementing the feature extractor using a pre-trained model from TensorFlow Hub
def get_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (299, 299))),
        tf.keras.layers.Lambda(preprocess_input),
        feature_extractor,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

# Implementing the transformer decoder architecture
def get_transformer_decoder():
    num_layers = 4
    d_model = 512
    num_heads = 8
    dff = 2048
    target_vocab_size = 5000
    maximum_position_encoding = 10000

    input1 = tf.keras.layers.Input(shape=(None, d_model))
    input2 = tf.keras.layers.Input(shape=(None,))
    look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
    pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    class AddPositionalEncoding(tf.keras.layers.Layer):
        def call(self, inputs):
            input2, pos_encoding = inputs
            return input2 + pos_encoding[:, :tf.shape(input2)[1], :]

    x = AddPositionalEncoding()([input2, pos_encoding])
    for _ in range(num_layers):
        x = transformer_decoder_layer(d_model, num_heads, dff)([input1, x, look_ahead_mask, padding_mask])
    output = tf.keras.layers.Dense(target_vocab_size)(x)
    return tf.keras.Model(inputs=[input1, input2, look_ahead_mask, padding_mask], outputs=output)

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def transformer_decoder_layer(d_model, num_heads, dff):
    inputs = tf.keras.layers.Input(shape=(None, d_model))
    look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None))
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None))
    attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs, attention_mask=look_ahead_mask)
    attn1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn1 + inputs)
    attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(attn1, attn1, attention_mask=padding_mask)
    attn2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn2 + attn1)
    ffn_output = point_wise_feed_forward_network(d_model, dff)(attn2)
    ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + attn2)
    return tf.keras.Model(inputs=[inputs, look_ahead_mask, padding_mask], outputs=ffn_output)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

# Placeholder for tokenizer
def get_tokenizer():
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (text.numpy() for text in tf.data.Dataset.from_tensor_slices(["This is a sample text for tokenizer training."])),
        target_vocab_size=2**13)
    tokenizer.start_token = tokenizer.vocab_size
    tokenizer.end_token = tokenizer.vocab_size + 1
    tokenizer.vocab_size += 2

    def tokenize(text):
        return [tokenizer.start_token] + tokenizer.encode(text) + [tokenizer.end_token]

    def detokenize(tokens):
        return tokenizer.decode(tokens[1:-1])

    tokenizer.tokenize = tokenize
    tokenizer.detokenize = detokenize
    return tokenizer

if __name__ == "__main__":
    # Create a test image (random noise)
    test_image = np.random.rand(1, 299, 299, 3).astype(np.float32)

    # Instantiate the ImageCaptioningModel
    model = ImageCaptioningModel()

    # Generate a caption for the test image
    caption = model.generate_caption(test_image)

    # Print the generated caption
    print("Generated Caption:", caption)
