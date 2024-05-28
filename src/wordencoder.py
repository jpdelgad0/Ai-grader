from sklearn.model_selection import train_test_split
import numpy as np
import keras_nlp
import keras.backend as K
import tensorflow as tf
import keras
import os
os.environ['KERAS_BACKEND'] = 'jax'

preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_base_en"
)
