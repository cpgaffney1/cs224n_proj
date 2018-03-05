#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import logging
import embedding_util as embedder
from functional_util import Progbar, minibatches
from model import Model

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class VBModel(Model):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, config, report=None):
        self.config = config
        self.report = report

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def evaluate(self, sess, examples, pad_tokens, write_preds=True):
        loss = 0.0
        count = 0
        predictions = []
        print('Dev predictions')
        print('---------------------')
        for i, batch in enumerate(minibatches(examples, self.config.batch_size)):
            encoder_inputs_batch, decoder_inputs_batch, labels_batch, \
            encoder_lengths_batch, decoder_lengths_batch = batch
            predictions, batch_loss = self.predict_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                                decoder_inputs_batch=decoder_inputs_batch,
                                                labels_batch=labels_batch,
                                                encoder_lengths_batch=encoder_lengths_batch,
                                                decoder_lengths_batch=decoder_lengths_batch,
                                                batch_size=encoder_lengths_batch.shape[0])
            loss += batch_loss
            predictions = self.index_to_word(predictions)
            print(self.print_pred(predictions[0]))
            with open('dev_predict.txt', 'a') as of:
               for p in predictions:
                   of.write(self.print_pred(p) + '\n')
            count += 1
        with open('dev_predict.txt', 'a') as of:
            of.write('\n')
        return predictions, loss / count

    def index_to_word(self, predictions):
        sentences = [[] for _ in predictions]
        for i, example in enumerate(predictions):
            sentences[i] = ' '.join([embedder.id2tok.get(id_num, '<unk>') for id_num in example])
        return sentences

    def print_pred(self, pred):
        end = pred.find('<end>')
        if end == -1:
            return pred
        else:
            return pred[:end + 1]

    def fit(self, sess, saver, train_examples_raw, dev_set_raw, pad_tokens=None):
        if pad_tokens is None:
            pad_tokens = []
        train_set = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)
        target = 1 + int(len(train_set) / self.config.batch_size)
        prog = Progbar(target=target)
        print('iterating over batches')
        for i, batch in enumerate(minibatches(train_set, self.config.batch_size)):
            prog.update(i)
            encoder_inputs_batch, decoder_inputs_batch, labels_batch, \
                encoder_lengths_batch, decoder_lengths_batch = batch
            predictions, loss = self.train_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                       decoder_inputs_batch=decoder_inputs_batch,
                                       encoder_lengths_batch=encoder_lengths_batch,
                                       decoder_lengths_batch=decoder_lengths_batch,
                                       labels_batch=labels_batch,
                                       batch_size=encoder_lengths_batch.shape[0])
            predictions = self.index_to_word(predictions)
            print()
            print('Sample training predictions, loss = {}'.format(loss))
            print('--------------------------')
            print(self.print_pred(predictions[0]))
            print()
            with open('training_output.txt', 'a') as of:
                of.write("Batch: {}, Loss: {}\n".format(i + 1, loss))

        _, loss = self.evaluate(sess, dev_set, pad_tokens)
        print("Dev set loss: " + str(loss))
        saver.save(sess, "models/seq2seq_model.ckpt")



