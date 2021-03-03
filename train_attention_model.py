from sklearn.model_selection import train_test_split
import pandas as pd
import re
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bert


def bert_encode(texts, tokenizer, max_len=510):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        input_sequence = ["[CLS]"] + text + ["[SEP]"]  # BERT can only take 512 tokens. Including these two, we get
        # 510+2
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, output_classes, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(output_classes, activation='softmax')(net)

    if output_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss=loss, metrics=['accuracy'])

    return model


def preprocess_txt(text):
    sentence = text
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


# import data here as pandas df from csv

def load_data(f_path, preprocess=True):
    """
    Loads and preprocesses data from a CSV file into a pandas dataframe. Returns split test and train
    data as well as labels.
    f_path: file path to .csv file
    preprocess: Whether to preprocess data to remove special characters, spaces, punctuation, and numbers.
    """
    data = pd.read_csv(f_path)
    data = data.drop(["Check"], axis=1).reset_index(drop=True)
    if data.isnull().values.any():  # warns us if there are null values
        print("W: Null values in data!")
    sentences = np.array(data['Sentence'])
    if preprocess:
        for i, sentence in enumerate(sentences):
            sentences[i] = preprocess_txt(sentence)
    data['Label'] = data['Label'].replace(['left', 'right'], [0, 1])
    train_x, test_x, train_y, test_y = train_test_split(sentences, np.array(data['Label']), test_size=0.1, shuffle=True)
    return train_x, test_x, train_y, test_y


def padded_batch_dataset(sentences, labels, batch_size):
    sentences = tf.data.Dataset.from_generator(lambda: (sentences, labels), output_types=(tf.int32, tf.int32))
    batched_dataset = sentences.padded_batch(batch_size, padded_shapes=((None,), ()))
    return batched_dataset


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=False)

vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower)


# fname = os.path.join(os.getcwd(), "\\commands_dataset.csv")
fname = "F:\\Documents\\Python Scripts\\RobotVoiceCommand\\commands_dataset.csv"
train_seq, test_seq, train_lab, test_lab = load_data(fname)


# make custom model


# define hyperparameters
PAD_LENGTH = len(sorted(train_seq, key=len)[-1])
OUTPUT_CLASSES = 2
NB_EPOCHS = 30
BATCH_SIZE = 8


train_input = bert_encode(train_seq, tokenizer=tokenizer, max_len=PAD_LENGTH)
test_input = bert_encode(test_seq, tokenizer=tokenizer, max_len=PAD_LENGTH)

model = build_model(bert_layer=bert_layer, output_classes=OUTPUT_CLASSES, max_len=PAD_LENGTH)

model.summary()

history = model.fit(train_input,
                    train_lab,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCHS,
                    validation_split=0.2)
