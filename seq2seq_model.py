import tensorflow as tf
from verbose_model import VBModel
from tensorflow.python.layers import core as layers_core

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """
    dropout = 0.5
    hidden_size = 200
    batch_size = 128
    n_epochs = 400
    lr = 0.01
    n_layers = 3

    def __init__(self, embed_size, vocab_size, max_encoder_timesteps, max_decoder_timesteps):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_encoder_timesteps = max_encoder_timesteps
        self.max_decoder_timesteps = max_decoder_timesteps
        self.hidden_size = embed_size



class Seq2SeqModel(VBModel):

    def add_placeholders(self):
        self.encoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_encoder_timesteps),
                                                        name="encoder_in")
        self.decoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_decoder_timesteps),
                                                        name="decoder_in")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_decoder_timesteps),
                                                 name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(),
                                                  name='dropout')
        self.encoder_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                          name='enc_lengths')
        self.decoder_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                          name='dec_lengths')

    def create_feed_dict(self, encoder_inputs_batch, decoder_inputs_batch,
                         labels_batch=None, encoder_lengths_batch=None, decoder_lengths_batch=None, dropout=1):
        feed_dict = {
            self.encoder_input_placeholder: encoder_inputs_batch,
            self.decoder_input_placeholder: decoder_inputs_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if encoder_lengths_batch is not None:
            feed_dict[self.encoder_lengths_placeholder] = encoder_lengths_batch
        if decoder_lengths_batch is not None:
            feed_dict[self.decoder_lengths_placeholder] = decoder_lengths_batch
        return feed_dict

    def add_embedding(self):
        pretrained_embeddings = tf.Variable(self.pretrained_embeddings)
        encoder_embeddings = tf.nn.embedding_lookup(
                pretrained_embeddings, self.encoder_input_placeholder)
        decoder_embeddings = tf.nn.embedding_lookup(
                pretrained_embeddings, self.decoder_input_placeholder)
        encoder_embeddings = tf.cast(encoder_embeddings, tf.float32)
        decoder_embeddings = tf.cast(decoder_embeddings, tf.float32)
        return encoder_embeddings, decoder_embeddings

    def add_prediction_op(self):

        encoder_in, decoder_in = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        # Build RNN cell
        encoder_cells = []
        for i in range(self.config.n_layers):
            encoder_cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size))
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)

        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        initial_state = encoder_cell.zero_state(self.config.batch_size, dtype=tf.float32)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell, inputs=encoder_in,
            #sequence_length=self.encoder_lengths_placeholder,
            dtype=tf.float32#, initial_state=initial_state
        )

        projection_layer = layers_core.Dense(
            self.config.vocab_size, use_bias=False)

        # Build RNN cell

        decoder_cells = []
        for i in range(self.config.n_layers):
            decoder_cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size))
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_in, self.decoder_lengths_placeholder)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        assert(logits.get_shape()[2] == self.config.vocab_size)
        return logits

    def add_loss_op(self, pred):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder, logits=pred))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        # pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i + len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret


    def predict_on_batch(self, sess, encoder_inputs_batch, decoder_inputs_batch, labels_batch,
                         encoder_lengths_batch, decoder_lengths_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(encoder_inputs_batch, decoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch,
                                     decoder_lengths_batch=decoder_lengths_batch)
        predictions, loss = sess.run([tf.argmax(self.pred, axis=2), self.loss], feed_dict=feed)
        return predictions, loss

    def train_on_batch(self, sess, encoder_inputs_batch, decoder_inputs_batch,
                       encoder_lengths_batch, decoder_lengths_batch, labels_batch):
        feed = self.create_feed_dict(encoder_inputs_batch, decoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch, decoder_lengths_batch=decoder_lengths_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, config, pretrained_embeddings, report=None):
        super(Seq2SeqModel, self).__init__(config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.encoder_input_placeholder = None
        self.decoder_input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.encoder_lengths_placeholder = None
        self.decoder_lengths_placeholder = None

        self.build()

    def preprocess_sequence_data(self, examples):
        return examples

