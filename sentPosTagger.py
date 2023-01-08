from numpy.random import seed

from bertLayr import BertLayer
from downLoadlibs import UD_ENGLISH_TRAIN, UD_ENGLISH_DEV, UD_ENGLISH_TEST
from inputEx import InputExample
from paddngInpEx import PaddingInputExample

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import pyconll, keras, nltk, warnings
import numpy as np
from sklearn.metrics.classification import UndefinedMetricWarning
from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer

from tqdm import tqdm_notebook
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


class PredictPosTags:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 70
        self.EPOCHS = 1
        # Params for bert model and tokenization
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        # Initialize session
        self.sess = tf.Session()

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

    def loadTokenizer(self):
        # Instantiate tokenizer
        tokenizer = self.create_tokenizer_from_hub_module()
        return tokenizer

    def loadModel(self, n_tags):
        model = self.build_model(self.MAX_SEQUENCE_LENGTH+2, n_tags)
        model.load_weights('bert_tagger.h5')
        return model

    def preProcessInputText(self, sentence_raw, tag2int, tokenizer):
        sentence_ini = nltk.word_tokenize(sentence_raw.lower())
        # sentence_bert = tokenizer.tokenize(sentence_raw)
        tokens_a = sentence_ini
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
        # input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])

        print('Original tokens:', tokens_a)
        print('BERT tokens:', tokens)
        print('orig_to_tok_map', orig_to_tok_map)

        # Convert data to InputExample format
        test_example = self.convert_text_to_examples([sentence_ini], [['-PAD-'] * len(sentence_ini)])

        # Convert to features
        (input_ids, input_masks, segment_ids, _
         ) = self.convert_examples_to_features(tokenizer, test_example, tag2int, max_seq_length=self.MAX_SEQUENCE_LENGTH + 2)

        return tokens, orig_to_tok_map, sentence_ini, input_ids, input_masks, segment_ids

    def getPrediction(self, model, input_ids, input_masks, segment_ids, tokens, orig_to_tok_map, sentence_ini, int2tag):
        predictions = model.predict([input_ids, input_masks, segment_ids], batch_size=1).argmax(-1)[0]
        #print("\n{:20}| {:15}: {:15}".format("Word in BERT layer", 'Initial word', "Predicted POS-tag"))
        #print(61 * '-')
        k = 0
        answers = []
        for i, pred in enumerate(predictions):
            try:
                if pred != 0:
                    #print("{:20}| {:15}: {:15}".format([tokens[i] for i in orig_to_tok_map][i], sentence_ini[i - 1],int2tag[pred]))
                    #print("{:15}: {:15}".format( sentence_ini[i - 1],int2tag[pred]))
                    #answers.append(sentence_ini[i - 1],int2tag[pred])
                    #answers.append()
                    #dicta = dict(zip(sentence_ini[i - 1],int2tag[pred]))
                    dicta2 = list((sentence_ini[i - 1], int2tag[pred]))
                    answers.append(dicta2)
                    k += 1
            except:
                pass
        #print(answers)
        return answers

    def executeProcessing(self, sentence_raw):
        train_sentences, val_sentences, test_sentences, tags, n_tags, int2tag, tag2int = self.readTrainingDataFiles(
            UD_ENGLISH_TRAIN,
            UD_ENGLISH_DEV,
            UD_ENGLISH_TEST)
        model = self.loadModel(n_tags)
        tokenizer = self.loadTokenizer()
        tokens, orig_to_tok_map, sentence_ini, input_ids, input_masks, segment_ids = self.preProcessInputText(
            sentence_raw, tag2int, tokenizer)
        result = self.getPrediction(model, input_ids, input_masks, segment_ids, tokens, orig_to_tok_map, sentence_ini, int2tag)
        #print("---------------------results---------------------")
        return result


#if __name__ == "__main__":
#    sentence_raw = 'Word embeddings provide a dense representation of words and their relative meanings.'
#    prdctObj = PredictPosTags()
#    prdctObj.executeProcessing(sentence_raw)