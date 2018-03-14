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
import tensorflow as tf


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

    def variable_summaries(self, var, name='summaries'):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


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
        for i, batch in enumerate(minibatches(examples, self.config.batch_size, shuffle=False)):
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
            with open('models/{}/dev_predict.txt'.format(self.config), 'a') as of:
                for p in predictions:
                    of.write(self.print_pred(p) + '\n')
        with open('models/{}/dev_predict.txt'.format(self.config), 'a') as of:
            of.write('\n')
        return predictions, self.dev_loss_sum / self.total_batches_done

    def fit(self, sess, saver, writer, train_examples, dev_set, pad_tokens=None, epoch=0):
        if pad_tokens is None:
            pad_tokens = []
        target = 1 + int(len(train_examples) / self.config.batch_size)
        prog = Progbar(target=target)
        print('iterating over batches')
        start_epoch = time.time()
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size, shuffle=True)):
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

        _, loss = self.evaluate(sess, dev_set, pad_tokens)
        print("Dev set loss: " + str(loss))
        if epoch % 20 == 0:# and self.dev_loss_sum / self.total_batches_done < self.best_dev_loss:
            saver.save(sess, "models/seq2seq_model.ckpt")
            self.best_dev_loss = self.dev_loss_sum / self.total_batches_done
        with open('train_loss.txt', 'a') as of:
            of.write("{}".format(self.train_loss_sum / self.total_batches_done))
        with open('dev_loss.txt', 'a') as of:
            of.write("{}".format(self.dev_loss_sum / self.total_batches_done))
        print('Epoch took {} sec'.format(time.time() - start_epoch))
        #writer.add_summary(summaries, epoch)


    def evaluate_fill(self, sess, examples, pad_tokens, write_preds=True):
        predictions = []
        loss = 0.0
        count = 0
        for i, batch in enumerate(minibatches(examples, self.config.batch_size, shuffle=False)):
            encoder_inputs_batch, labels_batch, encoder_lengths_batch = batch
            pred, batch_loss = self.predict_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                                            decoder_inputs_batch=None,
                                                            labels_batch=labels_batch,
                                                            encoder_lengths_batch=encoder_lengths_batch,
                                                            decoder_lengths_batch=None,
                                                            batch_size=encoder_lengths_batch.shape[0])
            predictions.append(pred)
            self.dev_loss_sum += batch_loss
            loss += batch_loss
            count += 1
        return predictions, loss / count

    def fit_fill(self, sess, saver, writer, train_examples, dev_set, pad_tokens=None, epoch=0):
        self.dev_loss_sum = 0
        self.train_loss_sum = 0
        self.total_batches_done = 0
        if pad_tokens is None:
            pad_tokens = []
        target = 1 + int(len(train_examples) / self.config.batch_size)
        prog = Progbar(target=target)
        print('iterating over batches')
        start_epoch = time.time()
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size, shuffle=True)):
            if i > 3005:
                break
            prog.update(i)
            start_batch = time.time()
            encoder_inputs_batch, labels_batch, encoder_lengths_batch = batch
            print('\n' + self.config.id2tok[labels_batch[0]])
            predictions, train_loss, cache = self.train_on_batch(sess, encoder_inputs_batch=encoder_inputs_batch,
                                                    decoder_inputs_batch=None,
                                                    encoder_lengths_batch=encoder_lengths_batch,
                                                    decoder_lengths_batch=None,
                                                    labels_batch=labels_batch,
                                                    batch_size=encoder_lengths_batch.shape[0])
            self.train_loss_sum += train_loss
            self.total_batches_done += 1
            predictions = [self.config.id2tok[pred] for pred in predictions]
            print(self.print_pred(self.index_to_word(encoder_inputs_batch)[0]) + ' + ' + predictions[0])
            print('Loss = {}'.format(train_loss))
            print('--------------------------')
            print()
            #print('Batch took {} sec'.format(time.time() - start_batch))
            #print()
            with open('training_output.txt', 'a') as of:
                of.write("Batch: {}, Loss: {}\n".format(i + 1, train_loss))
            if i % 100 == 0:
                _, dev_loss = self.evaluate_fill(sess, dev_set, pad_tokens)
                print("Dev set loss: " + str(dev_loss))
                with open('models/{}/dev_loss.txt'.format(self.config), 'a') as of:
                    of.write("{}\n".format(dev_loss))
                if dev_loss < self.best_dev_loss:
                    print('Saving new model')
                    np.save('models/{}/saved_cache.npy'.format(self.config), cache)
                    saver.save(sess, "models/{}/fill_model.ckpt".format(self.config))
                    self.best_dev_loss = dev_loss
            with open('models/{}/train_loss.txt'.format(self.config), 'a') as of:
                of.write("{}\n".format(train_loss))





