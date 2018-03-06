import tensorflow as tf
from verbose_model import VBModel
from tensorflow.python.layers import core as layers_core

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """
    dropout = 0.0
    hidden_size = 32
    batch_size = 32
    n_epochs = 1000
    lr = 0.01
    n_layers = 1
    beam_width = 10
    reg_weight = 0.00
    max_gradient_norm = 5.0

    def __init__(self, embed_size, vocab_size, max_encoder_timesteps, max_decoder_timesteps,
                 pad_token, start_token, end_token, attention, bidirectional, beamsearch=False,
                 mode='TRAIN', large=True):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_encoder_timesteps = max_encoder_timesteps
        self.max_decoder_timesteps = max_decoder_timesteps
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.attention = attention
        self.bidirectional = bidirectional
        self.mode = mode
        self.beamsearch = beamsearch
        if large:
            self.dropout = 0.4
            self.batch_size = 32
            self.hidden_size = 64
            self.n_layers = 3



class Seq2SeqModel(VBModel):

    def add_placeholders(self):
        self.encoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_encoder_timesteps),
                                                        name="encoder_in")
        self.decoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_decoder_timesteps + 1),
                                                        name="decoder_in")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_decoder_timesteps + 1),
                                                 name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(),
                                                  name='dropout')
        self.encoder_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                          name='enc_lengths')
        self.decoder_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                          name='dec_lengths')
        self.dynamic_batch_size = tf.placeholder(tf.int32, shape=(), name='dynamic_batch_size')

    def create_feed_dict(self, encoder_inputs_batch, decoder_inputs_batch,
                         labels_batch=None, encoder_lengths_batch=None, decoder_lengths_batch=None,
                         batch_size=None, dropout=0.0):
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
        if batch_size is not None:
            feed_dict[self.dynamic_batch_size] = batch_size
        return feed_dict

    def add_embedding(self):
        pretrained_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        encoder_embeddings = tf.nn.embedding_lookup(
                pretrained_embeddings, self.encoder_input_placeholder)
        decoder_embeddings = tf.nn.embedding_lookup(
                pretrained_embeddings, self.decoder_input_placeholder)
        encoder_embeddings = tf.cast(encoder_embeddings, tf.float32)
        decoder_embeddings = tf.cast(decoder_embeddings, tf.float32)
        return encoder_embeddings, decoder_embeddings

    def get_lstm_cell(self):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1.0 - self.dropout_placeholder)
        return drop

    def add_encoder(self, encoder_in):
        #encoder_lengths_constant = tf.fill(tf.shape(self.encoder_lengths_placeholder),
        #                                   self.config.max_encoder_timesteps)
        '''if False:
            # forward lstm
            forward_cells = []
            for i in range(self.config.n_layers):
                forward_cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size))
            forward_cell = tf.contrib.rnn.MultiRNNCell(forward_cells)

            # backward lstm
            backward_cells = []
            for i in range(self.config.n_layers):
                backward_cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size))
            backward_cell = tf.contrib.rnn.MultiRNNCell(backward_cells)
            (output_fw, output_bw), bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell, encoder_in,
                sequence_length=encoder_lengths_constant,
                dtype=tf.float32
            )
            encoder_outputs = tf.concat([output_fw, output_bw], axis=-1)
            if self.config.n_layers == 1:
                encoder_state = bi_encoder_state
            else:
                # alternatively concat forward and backward states
                encoder_state = []
                for layer_id in range(self.config.n_layers):
                    encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                    encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                encoder_state = tuple(encoder_state)
        else:'''
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.config.n_layers)])
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell, inputs=encoder_in,
            # sequence_length=encoder_lengths_constant,
            dtype=tf.float32
        )

        return encoder_outputs, encoder_state

    def add_attention(self, encoder_outputs, encoder_state, decoder_cell):
        if self.config.attention:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.config.hidden_size, encoder_outputs)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism, attention_layer_size=self.config.hidden_size
            )
            decoder_initial_state = decoder_cell.zero_state(self.dynamic_batch_size, tf.float32).clone(
                cell_state=encoder_state
            )
        else:
            decoder_initial_state = encoder_state
        return decoder_cell, decoder_initial_state

    def add_decoder(self, decoder_in, encoder_outputs, encoder_state):
        decoder_lengths_constant = tf.fill(tf.shape(self.decoder_lengths_placeholder),
                                           self.config.max_decoder_timesteps + 1)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.config.n_layers)])
        # Helper
        if self.config.mode == 'TRAIN':
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_in, decoder_lengths_constant
            )
        else:
            assert self.config.mode == 'TEST'
            if self.config.beamsearch:
                helper = None
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.pretrained_embeddings,
                    tf.fill([self.dynamic_batch_size], self.config.start_token),
                    self.config.end_token
                )

        decoder_cell, decoder_initial_state = self.add_attention(encoder_outputs, encoder_state, decoder_cell)

        projection_layer = layers_core.Dense(
            self.config.vocab_size, use_bias=True)

        # Decoder
        if self.config.beamsearch:
            # Replicate encoder infos beam_width times
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=self.config.beam_width)
            # Define a beam-search decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.pretrained_embeddings,
                start_tokens=self.config.start_token,
                end_token=self.config.end_token,
                initial_state=decoder_initial_state,
                beam_width=self.config.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)
        else:
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, decoder_initial_state,
                output_layer=projection_layer
            )
        #decoder_initial_state = (encoder_state, encoder_state)
        '''decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, decoder_in,
                                               initial_state=decoder_initial_state,
                                               dtype=tf.float32)'''

        #decoder_outputs = tf.reshape(decoder_outputs, shape=(-1, self.config.hidden_size))

        '''W = tf.get_variable('W_out', shape=(self.config.hidden_size, self.config.vocab_size),
                            dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.config.reg_weight))
        b = tf.get_variable('b_out', shape=(1, self.config.vocab_size),
                            dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(self.config.reg_weight))
        logits = tf.matmul(decoder_outputs, W) + b

        logits = tf.reshape(logits, shape=(-1, 41, 203020))'''
        #logits = tf.layers.dense(inputs=decoder_outputs, units=self.config.vocab_size, activation=tf.nn.relu)

        return decoder

    def add_prediction_op(self):
        encoder_in, decoder_in = self.add_embedding()

        with tf.variable_scope('encoder'):
            encoder_outputs, encoder_state = self.add_encoder(encoder_in)

        with tf.variable_scope('decoder'):
            decoder = self.add_decoder(decoder_in, encoder_outputs, encoder_state)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

        assert(logits.get_shape()[2] == self.config.vocab_size)
        return logits

    def add_loss_op(self, pred):
        mask = tf.sequence_mask(self.decoder_lengths_placeholder + 1, self.config.max_decoder_timesteps + 1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(tf.boolean_mask(loss, mask))
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                        self.config.max_gradient_norm)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
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
                         encoder_lengths_batch, decoder_lengths_batch, batch_size):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(encoder_inputs_batch, decoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch,
                                     decoder_lengths_batch=decoder_lengths_batch,
                                     batch_size=batch_size)
        predictions, loss = sess.run([tf.argmax(self.pred, axis=2), self.loss], feed_dict=feed)
        return predictions, loss

    def train_on_batch(self, sess, encoder_inputs_batch, decoder_inputs_batch,
                       encoder_lengths_batch, decoder_lengths_batch, labels_batch, batch_size):
        feed = self.create_feed_dict(encoder_inputs_batch, decoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch, decoder_lengths_batch=decoder_lengths_batch,
                                     batch_size=batch_size, dropout=self.config.dropout)
        predictions, _, loss = sess.run([tf.argmax(self.pred, axis=2), self.train_op, self.loss], feed_dict=feed)
        return predictions, loss

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
        self.dynamic_batch_size = None

        self.build()

    def preprocess_sequence_data(self, examples):
        return examples

