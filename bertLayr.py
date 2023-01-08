from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from keras import backend as K
import tensorflow_hub as hub

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# Model building
class BertLayer(Layer):
    def __init__(self, output_representation='sequence_output', trainable=True, **kwargs):
        self.bert = None
        super(BertLayer, self).__init__(**kwargs)

        self.trainable = trainable
        self.output_representation = output_representation

    def build(self, input_shape):
        # SetUp tensorflow Hub module
        self.bert = hub.Module(bert_path,
                               trainable=self.trainable,
                               name="{}_module".format(self.name))

        # Assign module's trainable weights to model
        # Remove unused layers and set trainable parameters
        # s = ["/cls/", "/pooler/", 'layer_11', 'layer_10', 'layer_9', 'layer_8', 'layer_7', 'layer_6']
        s = ["/cls/", "/pooler/"]
        self.trainable_weights += [var for var in self.bert.variables[:] if not any(x in var.name for x in s)]

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        # See Trainable Variables
        # tf.logging.info("**** Trainable Variables ****")
        # for var in self.trainable_weights:
        #    init_string = ", *INIT_FROM_CKPT*"
        #    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        print('Trainable weights:', len(self.trainable_weights))
        super(BertLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            self.output_representation
        ]
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs[0], 0.0)

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, None, 768)