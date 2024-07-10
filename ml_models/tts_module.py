import os
import re
import time
import tensorflow as tf
from scipy.io import wavfile
from german_transliterate.core import GermanTransliterate

# Define special symbols and all symbols used by the model
_pad = "pad"
_eos = "eos"
_punctuation = "!'(),.? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

ALL_SYMBOLS = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + [_eos]
)

# Regular expression to match text enclosed in curly braces
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def german_cleaners(text):
    """Pipeline for German text, including number and abbreviation expansion."""
    text = GermanTransliterate(replace={';': ',', ':': ' '}, sep_abbreviation=' -- ').transliterate(text)
    print(text)
    return text

class Processor:
    """German processor."""

    def __init__(self):
        self.symbol_to_id = {symbol: id for id, symbol in enumerate(ALL_SYMBOLS)}
        self.eos_id = self.symbol_to_id["eos"]

    def text_to_sequence(self, text):
        sequence = []
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(german_cleaners(text))
                break
            sequence += self._symbols_to_sequence(german_cleaners(m.group(1)))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)
        sequence += [self.eos_id]
        return sequence

    def _symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id and s != "_" and s != "~"

class TTSModule:
    def __init__(self):
        self.mbmelgan = None
        self.tacotron2 = None

    def load_models(self):
        if self.mbmelgan is None:
            path_to_mbmelgan = tf.keras.utils.get_file(
                'german-tts-mbmelgan.tar.gz',
                'https://storage.googleapis.com/mys-released-models/german-tts-mbmelgan.tar.gz',
                extract=True,
                cache_subdir='german-tts-mbmelgan'
            )
            self.mbmelgan = tf.saved_model.load(os.path.dirname(path_to_mbmelgan))

        if self.tacotron2 is None:
            path_to_tacotron2 = tf.keras.utils.get_file(
                'german-tts-tacotron2.tar.gz',
                'https://storage.googleapis.com/mys-released-models/german-tts-tacotron2.tar.gz',
                extract=True,
                cache_subdir='german-tts-tacotron2'
            )
            self.tacotron2 = tf.saved_model.load(os.path.dirname(path_to_tacotron2))

    def text_to_speech(self, text: str, output_path: str = "output.wav"):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        self.load_models()
        proc = Processor()
        input_ids = proc.text_to_sequence(text)

        start = time.time()
        _, mel_outputs, _, _ = self.tacotron2.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
        audio = self.mbmelgan.inference(mel_outputs)[0, :-1024, 0]
        duration = time.time() - start
        print(f"it took {duration} secs")
        wavfile.write(output_path, 22050, audio.numpy())
