#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import pdb
import logging
import util.embedding_util as embedder
import numpy as np

import tensorflow as tf
from util.functional_util import ConfusionMatrix, Progbar, minibatches
from src.model import Model

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


    def evaluate(self, sess, examples, pad_tokens, write_preds=False):
        loss = 0
        count = 0
        for i, batch in enumerate(minibatches(examples, self.config.batch_size)):
            encoder_inputs_batch, decoder_inputs_batch, labels_batch = batch
            encoder_lengths_batch = [len([word for word in example if word not in pad_tokens])
                                     for example in encoder_inputs_batch]
            decoder_lengths_batch = [len([word for word in example if word not in pad_tokens])
                                     for example in decoder_inputs_batch]
            predictions = self.predict_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                                decoder_inputs_batch=decoder_inputs_batch,
                                                labels_batch=labels_batch,
                                                encoder_lengths_batch=encoder_lengths_batch,
                                                decoder_lengths_batch=decoder_lengths_batch)
            predictions = self.index_to_word(predictions)
            print(predictions[:5])
            if write_preds:
                with open('../predictions/dev_predict.txt', 'a') as of:
                    for p in predictions:
                        of.write(p)
            loss += np.mean(np.sum((predictions != labels_batch), axis=-1))
            count += 1
        return loss / count

    def index_to_word(self, predictions):
        sentences = [[] for _ in predictions]
        for i, example in enumerate(predictions):
            sentences[i] = ' '.join([embedder.id2tok[id_num] for id_num in example])
        return sentences

    def fit(self, sess, saver, train_examples_raw, dev_set_raw, pad_tokens=None):
        best_score = 0.
        if pad_tokens is None:
            pad_tokens = []
        train_set = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            print()
            print('Epoch {} out of {}'.format(epoch + 1, self.config.n_epochs))
            # You may use the progress bar to monitor the training progress
            # Addition of progress bar will not be graded, but may help when debugging
            prog = Progbar(target=1 + int(len(train_set) / self.config.batch_size))
            for i, batch in enumerate(minibatches(train_set, self.config.batch_size)):
                prog.update(i)
                encoder_inputs_batch, decoder_inputs_batch, labels_batch = batch
                encoder_lengths_batch = [len([word for word in example if word not in pad_tokens])
                                         for example in encoder_inputs_batch]
                decoder_lengths_batch = [len([word for word in example if word not in pad_tokens])
                                         for example in decoder_inputs_batch]
                loss = self.train_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                           decoder_inputs_batch=decoder_inputs_batch,
                                           encoder_lengths_batch=encoder_lengths_batch,
                                           decoder_lengths_batch=decoder_lengths_batch,
                                           labels_batch=labels_batch)
                print(" Loss: " + str(loss))
            loss = self.evaluate(sess, dev_set, pad_tokens)
            print("Mean incorrect predictions on dev set: " + str(loss))

        saver.save(sess, "../models/seq2seq_model.ckpt")
        return best_score



