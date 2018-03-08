#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import logging
import embedding_util as embedder
from functional_util import Progbar, minibatches
from model import Model
import numpy as np
import time


class VBModel(Model):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, config, report=None):
        self.config = config
        self.report = report
        self.dev_loss_sum = 0
        self.train_loss_sum = 0
        self.total_batches_done = 0
        self.best_dev_loss = float('inf')

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

    def index_to_word(self, predictions):
        sentences = ['' for _ in predictions]
        for i, example in enumerate(predictions):
            sentences[i] = ' '.join([self.config.id2tok.get(id_num, '<unk>') for id_num in example])
        return sentences

    def print_pred(self, pred):
        end = pred.find('<end>')
        if end == -1:
            return pred
        else:
            return pred[:end]

    def get_batch_list(self, data, batch_size):
        np.random.shuffle(data)
        length = len(data)
        batch_list = []
        data = [np.array(col) for col in zip(*data)]
        for i in range(int(length / batch_size)):
            batch_list.append([col[i * batch_size: (i + 1) * batch_size] for col in data])
        return batch_list

    def evaluate(self, sess, examples, pad_tokens, write_preds=True):
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
            self.dev_loss_sum += batch_loss
            predictions = self.index_to_word(predictions)
            print(self.print_pred(predictions[0]))
            with open('dev_predict.txt', 'a') as of:
                for p in predictions:
                    of.write(self.print_pred(p) + '\n')
        with open('dev_predict.txt', 'a') as of:
            of.write('\n')
        return predictions, self.dev_loss_sum / self.total_batches_done

    def fit(self, sess, saver, train_examples, dev_set, pad_tokens=None, epoch=0):
        if pad_tokens is None:
            pad_tokens = []
        target = 1 + int(len(train_examples) / self.config.batch_size)
        prog = Progbar(target=target)
        print('iterating over batches')
        start_epoch = time.time()
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            prog.update(i)
            start_batch = time.time()
            encoder_inputs_batch, decoder_inputs_batch, labels_batch, \
            encoder_lengths_batch, decoder_lengths_batch = batch
            #print(self.print_pred(self.index_to_word(encoder_inputs_batch)[0]))
            #print(self.print_pred(self.index_to_word(decoder_inputs_batch)[0]))
            #print(self.print_pred(self.index_to_word(labels_batch)[0]))
            predictions, loss = self.train_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                                    decoder_inputs_batch=decoder_inputs_batch,
                                                    encoder_lengths_batch=encoder_lengths_batch,
                                                    decoder_lengths_batch=decoder_lengths_batch,
                                                    labels_batch=labels_batch,
                                                    batch_size=encoder_lengths_batch.shape[0])
            self.train_loss_sum += loss
            self.total_batches_done += 1
            predictions = self.index_to_word(predictions)
            print()
            print('Loss = {}'.format(self.train_loss_sum / self.total_batches_done))
            print('--------------------------')
            print(self.print_pred(predictions[0]))
            print('Batch took {} sec'.format(time.time() - start_batch))
            print()
            with open('training_output.txt', 'a') as of:
                of.write("Batch: {}, Loss: {}\n".format(i + 1, self.train_loss_sum / self.total_batches_done))

        #_, loss = self.evaluate(sess, dev_set, pad_tokens)
        #print("Dev set loss: " + str(loss))
        if epoch % 10 == 0 and self.dev_loss_sum / self.total_batches_done < self.best_dev_loss:
            saver.save(sess, "models/seq2seq_model{}.ckpt".format(epoch))
            self.best_dev_loss = self.dev_loss_sum / self.total_batches_done
        with open('train_loss.txt', 'a') as of:
            of.write("{}".format(self.train_loss_sum / self.total_batches_done))
        with open('dev_loss.txt', 'a') as of:
            of.write("{}".format(self.dev_loss_sum / self.total_batches_done))
        print('Epoch took {} sec'.format(time.time() - start_epoch))

