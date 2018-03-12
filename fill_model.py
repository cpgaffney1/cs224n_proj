import tensorflow as tf
from verbose_model import VBModel
from tensorflow.python.layers import core as layers_core
import numpy as np

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """
    dropout = 0.0
    hidden_size = 128
    attention_size = 64
    batch_size = 64
    n_epochs = 1
    lr = 0.01
    n_layers = 1
    beam_width = 10
    reg_weight = 0.00
    max_gradient_norm = 5.0
    cache_size = 100
    uses_regularization = True

    def __init__(self, embed_size, vocab_size, max_encoder_timesteps, max_decoder_timesteps,
                 pad_token, start_token, end_token, attention, bidirectional, id2tok, beamsearch=False,
                 mode='TRAIN', large=True, cache=False):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_encoder_timesteps = max_encoder_timesteps
        self.pad_token = pad_token
        self.attention = attention
        self.bidirectional = bidirectional
        self.mode = mode
        self.id2tok = id2tok
        self.use_cache = cache
        if large:
            self.dropout = 0.6
            self.batch_size = 64
            self.hidden_size = 128
            self.n_layers = 1

    def __str__(self):
        return 'RegularizationWeight_{}_HiddenSize_{}_Dropout_{}_NLayers_{}_Lr_{}_Bidirectional_{}_Attention_{}_Cache_{}_Embed_100'.format(self.reg_weight,
                                            self.hidden_size, self.dropout, self.n_layers, self.lr, self.bidirectional,
                                            self.attention,self.use_cache, self.embed_size)


class FillModel(VBModel):
    cache = None

    def add_placeholders(self):
        self.encoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_encoder_timesteps),
                                                        name="encoder_in")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                 name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(),
                                                  name='dropout')
        self.encoder_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,),
                                                          name='enc_lengths')
        self.cache_placeholder = tf.placeholder(tf.float32, shape=(100, self.config.hidden_size),
                                                          name='cache')
        self.dynamic_batch_size = tf.placeholder(tf.int32, shape=(), name='dynamic_batch_size')

    def create_feed_dict(self, encoder_inputs_batch,
                         labels_batch=None, encoder_lengths_batch=None,
                         batch_size=None, dropout=0.0):
        feed_dict = {
            self.encoder_input_placeholder: encoder_inputs_batch,
            self.dropout_placeholder: dropout,
            self.cache_placeholder: self.cache
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if encoder_lengths_batch is not None:
            feed_dict[self.encoder_lengths_placeholder] = encoder_lengths_batch
        if batch_size is not None:
            feed_dict[self.dynamic_batch_size] = batch_size
        return feed_dict

    def add_embedding(self):
        pretrained_embeddings = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
        #self.variable_summaries(pretrained_embeddings, name='embeddings')
        encoder_embeddings = tf.nn.embedding_lookup(
            pretrained_embeddings, self.encoder_input_placeholder)
        encoder_embeddings = tf.cast(encoder_embeddings, tf.float32)
        return encoder_embeddings

    def get_lstm_cell(self):
        lstm = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=1.0 - self.dropout_placeholder)
        return lstm

    # TODO modify this
    # additive attention
    def attention(self, inputs):
        W = tf.get_variable("weights_W", [self.config.hidden_size, self.config.attention_size])
        v = tf.get_variable("weights_v", [self.config.attention_size, 1])
        #if self.config.uses_regularization:
        #    W = tf.nn.l2_normalize(W, dim=-1)
         #   v = tf.nn.l2_normalize(v, dim=-1)

        inputs = tf.reshape(inputs, shape=(-1, self.config.hidden_size))
        M = tf.tanh(tf.matmul(inputs, W))
        a = tf.nn.softmax(tf.matmul(M, v), dim=-1)
        print(a)
        a = tf.reshape(a, shape=(self.dynamic_batch_size, self.config.max_encoder_timesteps))
        a = tf.tile(tf.expand_dims(a, dim=-1), multiples=(1, 1, self.config.hidden_size))
        inputs = tf.reshape(inputs, shape=(self.dynamic_batch_size, self.config.max_encoder_timesteps, self.config.hidden_size))
        weighted_input = tf.reduce_sum(inputs * a, axis=1)
        return weighted_input

    # TODO modify this
    def cache_attention(self):
        self.cache_W = tf.get_variable("cache_weights_W", [self.config.hidden_size, self.config.attention_size])
        self.cache_v = tf.get_variable("cache_weights_v", [self.config.attention_size])
        #if self.config.uses_regularization:
        #    self.cache_W = tf.nn.l2_normalize(self.cache_W, dim=-1)
        #    self.cache_v = tf.nn.l2_normalize(self.cache_v, dim=-1)

        M = tf.tanh(tf.matmul(self.cache_placeholder, self.cache_W))
        print(M)
        a = tf.nn.softmax(tf.matmul(M, tf.transpose(tf.expand_dims(self.cache_v,0), [1,0])))
        print(a)
        weighted_cache = tf.reduce_sum(self.cache_placeholder * a, axis=0)
        print(weighted_cache)
        return weighted_cache


    def get_encoder(self, encoder_in):
        if self.config.bidirectional:
            # forward lstm
            forward_cells = [self.get_lstm_cell() for _ in range(self.config.n_layers)]
            # backward lstm
            backward_cells = [self.get_lstm_cell() for _ in range(self.config.n_layers)]
            encoder_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                forward_cells, backward_cells, encoder_in,
                dtype=tf.float32
            )
            '''for lstm in forward_cells:
                kernel, bias = lstm.variables
                self.variable_summaries(kernel, name='lstm_kernel')
                self.variable_summaries(bias, name='lstm_bias')
            for lstm in backward_cells:
                kernel, bias = lstm.variables
                self.variable_summaries(kernel, name='lstm_kernel')
                self.variable_summaries(bias, name='lstm_bias')'''
        else:
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.config.n_layers)])
            encoder_outputs, _ = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=encoder_in,
                dtype=tf.float32
            )
        return encoder_outputs

    def add_prediction_op(self):
        encoder_in = self.add_embedding()
        with tf.variable_scope('prediction'):
            encoder_outputs = self.get_encoder(encoder_in)
            print(encoder_outputs)
            if self.config.attention:
                self.last_output = self.attention(encoder_outputs)
            else:
                encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
                self.last_output = tf.gather(encoder_outputs, int(encoder_outputs.get_shape()[0]) - 1)
                print(self.last_output)
            state = self.last_output
            if self.config.use_cache:
                z = tf.get_variable("final_cache_weights_z", [self.config.hidden_size])
            #    z = tf.nn.l2_normalize(z, dim=-1)
                weighted_cache = self.cache_attention()
                state = state + z * weighted_cache
            pred = tf.layers.dense(state, self.config.vocab_size)
            #if self.config.uses_regularization:
            #    self.last_output = tf.nn.l2_normalize(self.last_output, dim=-1)
            #    pred = tf.nn.l2_normalize(pred, dim=-1)
        return pred


    def add_loss_op(self, pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer().minimize((loss))
        return train_op


    def predict_on_batch(self, sess, encoder_inputs_batch, decoder_inputs_batch, labels_batch=None,
                         encoder_lengths_batch=None, decoder_lengths_batch=None, batch_size=None):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(encoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch,
                                     batch_size=batch_size)
        if labels_batch is None:
            predictions = sess.run([tf.argmax(self.pred, axis=1)], feed_dict=feed)
            loss = 0
        else:
            predictions, loss = sess.run([tf.argmax(self.pred, axis=1), self.loss], feed_dict=feed)
        return predictions, loss

    def train_on_batch(self, sess, encoder_inputs_batch, decoder_inputs_batch,
                       encoder_lengths_batch, decoder_lengths_batch, labels_batch, batch_size):
        #merge = tf.summary.merge_all()
        feed = self.create_feed_dict(encoder_inputs_batch, labels_batch=labels_batch,
                                     encoder_lengths_batch=encoder_lengths_batch,
                                     batch_size=batch_size, dropout=self.config.dropout)
        predictions, _, loss = sess.run([tf.argmax(self.pred, axis=1), self.train_op, self.loss], feed_dict=feed)
        if self.config.use_cache:
            candidate_batch, W, v = sess.run([self.last_output,
                                              self.cache_W, self.cache_v], feed_dict=feed)
            self.cache = candidate_batch
            '''for i in range(len(candidate_batch)):
                candidate = candidate_batch[i]
                if self.insert_cache_candidate(candidate, W, v):
                    print('inserted cache')
                    self.maintain_cache(0, W, v)'''
        return predictions, loss

    def insert_cache_candidate(self, candidate, W, v):
        candidate_score = np.dot(np.matmul(candidate, W), v)
        min_score = np.dot(np.matmul(self.cache[0], W), v)
        if candidate_score > min_score:
            self.cache[0] = candidate
            return True
        return False


    def maintain_cache(self, i, W, v):
        def parent(j):
            return int(j / 2)

        def right(j):
            return j * 2

        def left(j):
            return j * 2 + 1

        if i + 1 >= len(self.cache):
            return

        candidate_score = np.dot(np.matmul(self.cache[i], W), v)
        if right(i) < len(self.cache):
            right_score = np.dot(np.matmul(self.cache[right(i)], W), v)
        if left(i) < len(self.cache):
            left_score = np.dot(np.matmul(self.cache[left(i)], W), v)

        if right(i) < len(self.cache) and candidate_score > right_score:
            temp = self.cache[right(i)].copy()
            self.cache[right(i)] = self.cache[i].copy()
            self.cache[i] = temp
            self.maintain_cache(right(i), W, v)
        elif left(i) < len(self.cache) and candidate_score > left_score:
            temp = self.cache[left(i)].copy()
            self.cache[left(i)] = self.cache[i].copy()
            self.cache[i] = temp
            self.maintain_cache(left(i), W, v)



    def __init__(self, config, pretrained_embeddings, report=None):
        super(FillModel, self).__init__(config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.encoder_input_placeholder = None
        self.decoder_input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.encoder_lengths_placeholder = None
        self.decoder_lengths_placeholder = None
        self.dynamic_batch_size = None
        self.cache_placeholder = None

        self.build()

        self.cache = np.zeros((self.config.cache_size, self.config.hidden_size))

    def preprocess_sequence_data(self, examples):
        return examples
