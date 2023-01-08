from numpy.random import seed

from graphPlotting import plotHisTrnngSents, plot_confusion_matrix

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import pyconll, keras, pickle, os, random, nltk, datetime, warnings, gc, urllib.request, zipfile, collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, \
    f1_score
from sklearn.metrics.classification import UndefinedMetricWarning

from bertLayr import BertLayer
from downLoadlibs import UD_ENGLISH_TRAIN, UD_ENGLISH_DEV, UD_ENGLISH_TEST
from inputEx import InputExample
from paddngInpEx import PaddingInputExample

from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer

import itertools

from tqdm import tqdm_notebook
from IPython.display import Image

import nltk
nltk.download('punkt')

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


class TrainPostagger:

    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 70
        self.EPOCHS = 1

        # Initialize session
        self.sess = tf.Session()
        # Params for bert model and tokenization
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.modelFileName = 'bert_Postagger.h5'

    def read_conllu(self, path):
        data = pyconll.load_from_file(path)
        tagged_sentences = []
        t = 0
        for sentence in data:
            tagged_sentence = []
            for token in sentence:
                if token.upos and token.form:
                    t += 1
                    tagged_sentence.append((token.form.lower(), token.upos))
            tagged_sentences.append(tagged_sentence)
        return tagged_sentences

    # Some usefull functions
    def tag_sequence(self, sentences):
        return [[t for w, t in sentence] for sentence in sentences]

    def text_sequence(self, sentences):
        return [[w for w, t in sentence] for sentence in sentences]

    def readTrainingDataFiles(self, ud_eng_train, ud_eng_dev, ud_eng_test):
        train_sentences = self.read_conllu(ud_eng_train)
        val_sentences = self.read_conllu(ud_eng_dev)
        test_sentences = self.read_conllu(ud_eng_test)

        tags = set([item for sublist in train_sentences + test_sentences + val_sentences for _, item in sublist])
        print('TOTAL TAGS: ', len(tags))

        tag2int = {}
        int2tag = {}

        for i, tag in enumerate(sorted(tags)):
            tag2int[tag] = i + 1
            int2tag[i + 1] = tag

        # Special character for the tags
        tag2int['-PAD-'] = 0
        int2tag[0] = '-PAD-'

        n_tags = len(tag2int)
        print('Total tags:', n_tags)

        return train_sentences, val_sentences, test_sentences, tags, n_tags, int2tag, tag2int

    def split(self, sentences, max):
        new = []
        for data in sentences:
            new.append(([data[x:x + max] for x in range(0, len(data), max)]))
        new = [val for sublist in new for val in sublist]
        return new

    def getSplittedTrnngSents(self, train_sentences, val_sentences, test_sentences):
        train_sentences = self.split(train_sentences, self.MAX_SEQUENCE_LENGTH)
        val_sentences = self.split(val_sentences, self.MAX_SEQUENCE_LENGTH)
        test_sentences = self.split(test_sentences, self.MAX_SEQUENCE_LENGTH)

        train_sentences = train_sentences + val_sentences

        train_text = self.text_sequence(train_sentences)
        test_text = self.text_sequence(test_sentences)
        # val_text = self.text_sequence(val_sentences)

        train_label = self.tag_sequence(train_sentences)
        test_label = self.tag_sequence(test_sentences)
        # val_label= tag_sequence(val_sentences)

        return train_text, train_label, test_text, test_label

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        bert_module = hub.Module(self.bert_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )

        return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def convert_single_example(self, tokenizer, example, tag2int, max_seq_length=256):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, PaddingInputExample):
            input_ids = [0] * max_seq_length
            input_mask = [0] * max_seq_length
            segment_ids = [0] * max_seq_length
            label_ids = [0] * max_seq_length
            return input_ids, input_mask, segment_ids, label_ids

        tokens_a = example.text_a
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0: (max_seq_length - 2)]

        # Token map will be an int -> int mapping between the `orig_tokens` index and
        # the `bert_tokens` index.

        # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
        # orig_to_tok_map == [1, 2, 4, 6]
        orig_to_tok_map = []
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens) - 1)
        # print(len(tokens_a))
        for token in tokens_a:
            tokens.extend(tokenizer.tokenize(token))
            orig_to_tok_map.append(len(tokens) - 1)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens) - 1)
        input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])
        # print(len(orig_to_tok_map), len(tokens), len(input_ids), len(segment_ids)) #for debugging

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        label_ids = []
        labels = example.label
        label_ids.append(0)
        label_ids.extend([tag2int[label] for label in labels])
        label_ids.append(0)
        # print(len(label_ids)) #for debugging
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        return input_ids, input_mask, segment_ids, label_ids

    def convert_examples_to_features(self, tokenizer, examples, tag2int, max_seq_length=256):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for example in tqdm_notebook(examples, desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = self.convert_single_example(
                tokenizer, example, tag2int, max_seq_length
            )
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
        return (
            np.array(input_ids),
            np.array(input_masks),
            np.array(segment_ids),
            np.array(labels),
        )

    def convert_text_to_examples(self, texts, labels):
        """Create InputExamples"""
        InputExamples = []
        for text, label in zip(texts, labels):
            InputExamples.append(
                InputExample(guid=None, text_a=text, text_b=None, label=label)
            )
        return InputExamples

    def bert_labels(labels):
        train_label_bert = []
        train_label_bert.append('-PAD-')
        for i in labels:
            train_label_bert.append(i)
        train_label_bert.append('-PAD-')
        print('BERT labels:', train_label_bert)


    def initialiseTokenizer(self):
        # Instantiate tokenizer
        tokenizer = self.create_tokenizer_from_hub_module()

        return tokenizer

    def preProcessTrainingData(self, train_text, train_label, test_text, test_label, tokenizer, n_tags, tag2int):
        tokens_a = train_text[2]
        orig_to_tok_map = []
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens) - 1)
        for token in tokens_a:
            # orig_to_tok_map.append(len(tokens)) # keep first piece of tokenized term
            tokens.extend(tokenizer.tokenize(token))
            orig_to_tok_map.append(len(tokens) - 1)  # # keep last piece of tokenized term -->> gives better results!
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        orig_to_tok_map.append(len(tokens) - 1)
        input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])

        """Create InputExamples"""
        InputExamples = []
        for text, label in zip(train_text[0:1], train_label[0:1]):
            InputExamples.append(
                InputExample(guid=None, text_a=text, text_b=None, label=label)
            )

        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for example in tqdm_notebook(InputExamples, desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = self.convert_single_example(
                tokenizer, example, tag2int, self.MAX_SEQUENCE_LENGTH + 2
            )
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)

        # Convert data to InputExample format
        train_examples = self.convert_text_to_examples(train_text, train_label)
        test_examples = self.convert_text_to_examples(test_text, test_label)

        # Convert to features
        (train_input_ids, train_input_masks, train_segment_ids, train_labels_ids
         ) = self.convert_examples_to_features(tokenizer, train_examples, tag2int, max_seq_length=self.MAX_SEQUENCE_LENGTH + 2)
        (test_input_ids, test_input_masks, test_segment_ids, test_labels_ids
         ) = self.convert_examples_to_features(tokenizer, test_examples, tag2int, max_seq_length=self.MAX_SEQUENCE_LENGTH + 2)

        # One-hot encode labels
        train_labels = to_categorical(train_labels_ids, num_classes=n_tags)
        test_labels = to_categorical(test_labels_ids, num_classes=n_tags)

        return train_input_ids, train_input_masks, train_segment_ids, train_labels_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels_ids, train_labels, test_labels

    # Build model
    def build_model(self, max_seq_length, n_tags):
        seed = 0
        in_id = keras.layers.Input(shape=(max_seq_length,), name="input_ids")
        in_mask = keras.layers.Input(shape=(max_seq_length,), name="input_masks")
        in_segment = keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        np.random.seed(seed)
        bert_output = BertLayer()(bert_inputs)

        np.random.seed(seed)
        outputs = keras.layers.Dense(n_tags, activation=keras.activations.softmax)(bert_output)

        np.random.seed(seed)
        model = keras.models.Model(inputs=bert_inputs, outputs=outputs)
        np.random.seed(seed)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.00004), loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        model.summary(100)
        return model

    def initialize_vars(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        K.set_session(sess)

    def trainModel(self, n_tags, train_input_ids, train_input_masks, train_segment_ids, train_labels, test_input_ids, test_input_masks, test_segment_ids, test_labels):

        self.initialize_vars(self.sess)
        model = self.build_model(self.MAX_SEQUENCE_LENGTH + 2, n_tags)  # We sum 2 for [CLS], [SEP] tokens
        plot_model(model, to_file='model.png', show_shapes=True)
        Image('model.png')
        t_ini = datetime.datetime.now()

        cp = ModelCheckpoint(filepath=self.modelFileName,
                             monitor='val_acc',
                             save_best_only=False,
                             save_weights_only=True,
                             verbose=1)

        early_stopping = EarlyStopping(monitor='val_acc', patience=5)

        history = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                            train_labels,
                            validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
                            # validation_split=0.3,
                            epochs=self.EPOCHS,
                            batch_size=16,
                            shuffle=True,
                            verbose=1,
                            callbacks=[cp, early_stopping]
                            )

        t_fin = datetime.datetime.now()
        print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))

        return history

    def y2label(self, int2tag, zipped, mask=0):
        out_true = []
        out_pred = []
        for zip_i in zipped:
            a, b = tuple(zip_i)
            if a != mask:
                out_true.append(int2tag[a])
                out_pred.append(int2tag[b])
        return out_true, out_pred

    def evaluateModel(self, test_input_ids, test_input_masks, test_segment_ids, test_labels, int2tag, n_tags):
        model = self.build_model(self.MAX_SEQUENCE_LENGTH + 2, n_tags)
        model.load_weights(self.modelFileName)
        y_pred = model.predict([test_input_ids, test_input_masks, test_segment_ids]).argmax(-1)
        y_true = test_labels.argmax(-1)
        y_zipped = zip(y_true.flat, y_pred.flat)
        y_true, y_pred = self.y2label(int2tag, y_zipped)

        name = 'Bert fine-tuned model'
        print('\n------------ Result of {} ----------\n'.format(name))
        print(classification_report(y_true, y_pred, digits=4))

        print("Accuracy: {0:.4f}".format(accuracy_score(y_true, y_pred)))
        print('f1-macro score: {0:.4f}'.format(f1_score(y_true, y_pred, average='macro')))

        return y_true, y_pred

    def executeProcessing(self):
        train_sentences, val_sentences, test_sentences, tags, n_tags, int2tag, tag2int = self.readTrainingDataFiles(
            UD_ENGLISH_TRAIN,
            UD_ENGLISH_DEV,
            UD_ENGLISH_TEST)
        # download_files()
        # Plot 'Max sentence length:'
        plotHisTrnngSents(train_sentences)
        print('Max sentence length:', len(max(train_sentences + val_sentences, key=len)))

        train_text, train_label, test_text, test_label = self.getSplittedTrnngSents(train_sentences, val_sentences, test_sentences)

        tokenizer = self.initialiseTokenizer()
        train_input_ids, train_input_masks, train_segment_ids, train_labels_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels_ids, train_labels, test_labels = self.preProcessTrainingData(train_text, train_label, test_text, test_label, tokenizer, n_tags, tag2int)
        history = self.trainModel(n_tags, train_input_ids, train_input_masks, train_segment_ids, train_labels, test_input_ids,
                   test_input_masks, test_segment_ids, test_labels)

        y_true, y_pred  = self.evaluateModel(test_input_ids, test_input_masks, test_segment_ids, test_labels, int2tag, n_tags)

        name = 'Bert fine-tuned model'
        tags = sorted(set(y_pred + y_true))
        cnf_matrix = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(f1_score(y_true, y_pred, average='macro'), cnf_matrix, target_names=tags, title=name,
                              normalize=False)


if __name__ == "__main__":
    trnObj = TrainPostagger()
    trnObj.executeProcessing()