from parser_util import make_seq2seq_data, make_seq2seq_data_v2
import parser_util
import embedding_util as embedder
from seq2seq_model import Seq2SeqModel, Config
import tensorflow as tf
import time, sys, random, os
import numpy as np
from threading import Thread
import argparse
import pickle
from functional_util import minibatches


def split_train_dev(data):
    split = int(len(data) * 0.85)
    return data[:split], data[split:]


def load_and_split(args):
    def index_to_word(predictions):
        sentences = [[] for _ in predictions]
        for i, example in enumerate(predictions):
            sentences[i] = ' '.join([id2tok.get(id_num, '<unk>') for id_num in example])
        return sentences

    def print_pred(self, pred):
        return pred[:pred.find('<end>')]

    # simple and normal are just a list of sentences (as strings)
    _, _, _, tok2id, id2tok = embedder.load_embeddings(mode='full')
    _, normal, simple, _, _ = embedder.load_embeddings(mode='train')

    data = make_seq2seq_data_v2(simple, normal, embedder.START, embedder.END, embedder.PAD, tok2id, id2tok=id2tok)
    random.shuffle(data)
    obs_per_file = 512
    for i in range(int(len(data) / obs_per_file)):
        subset = data[i * obs_per_file : (i+1) * obs_per_file]
        pickle.dump(subset, open('data//' + str(i) + ".data", "wb"))

def load_one_datafile(filenum):
    # simple and normal are just a list of sentences (as strings)
    data = pickle.load(open('data//' + str(filenum) + ".data", "rb"))
    return data

def train(args):
    pretrained_embeddings, _, _, tok2id, id2tok = embedder.load_embeddings(large=args.large, mode='full')
    # clear training log file
    with open('training_output.txt', 'w') as of:
        print('Beginning train with params:')
        print('Attention: {}, Bidirectional: {}'.format(args.attention, args.bidirectional))
        print()
        of.write('Beginning train with params:\n')
        of.write('Attention: {}, Bidirectional: {}\n\n'.format(args.attention, args.bidirectional))

    if args.resume:
        tf.reset_default_graph()
    else:
        with open('dev_predict.txt', 'w') as of:
            of.write('\n')
        with open('train_loss.txt', 'w') as of:
            of.write('\n')
        with open('dev_loss.txt', 'w') as of:
            of.write('\n')


    with tf.Graph().as_default():
        print("Building model...",)
        start = time.time()
        config = Config(len(pretrained_embeddings[0]), len(pretrained_embeddings),
                        parser_util.max_normal_timesteps, parser_util.max_simple_timesteps,
                        embedder.PAD, tok2id[embedder.START], tok2id[embedder.END], args.attention, args.bidirectional,
                        id2tok,
                        large=args.large)
        model = Seq2SeqModel(config, pretrained_embeddings)
        print("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            if args.resume:
                saver.restore(session, 'models/seq2seq_model.ckpt')
            else:
                session.run(init)
            if args.buildmodel:
                exit(0)
            for epoch in range(config.n_epochs):
                print()
                print('Epoch {} out of {}'.format(epoch + 1, config.n_epochs))
                with open('training_output.txt', 'a') as of:
                    of.write('Epoch {} out of {}\n'.format(epoch + 1, config.n_epochs))
                randchoice = random.randint(0, len([file for file in os.listdir('data') if file.endswith('.data')]) - 1)
                randchoice = 0
                data = load_one_datafile(randchoice)
                data = data[:20]
                #data= [
                #    (np.random.randint(2, size=20), np.random.randint(2, size=41), np.random.randint(2, size=41), 20, 41) for _ in range(400)
                #]
                train_data, dev_data = split_train_dev(data)
                model.fit(session, saver, train_data, dev_data, pad_tokens=[embedder.PAD, embedder.END], epoch=epoch)

def evaluate(args):
    pretrained_embeddings, _, _, tok2id, id2tok = embedder.load_embeddings(large=args.large, mode='full')
    _, normal, simple, _, _ = embedder.load_embeddings(large=args.large, mode='test')
    data = make_seq2seq_data(simple, normal, embedder.START, embedder.END, embedder.PAD, tok2id)
    tf.reset_default_graph()

    with tf.Graph().as_default():
        config = Config(len(pretrained_embeddings[0]), len(pretrained_embeddings),
                        parser_util.max_normal_timesteps, parser_util.max_simple_timesteps,
                        embedder.PAD, tok2id[embedder.START], tok2id[embedder.END], args.attention, args.bidirectional,
                        id2tok, mode='TEST',
                        beamsearch=args.beamsearch, large=args.large)
        model = Seq2SeqModel(config, pretrained_embeddings)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, 'models/seq2seq_model.ckpt')
            predictions, loss = model.evaluate(session, data, pad_tokens=[embedder.PAD, embedder.END])
            predictions = model.index_to_word(predictions)
            length_dif_sum_pred = 0
            length_dif_sum_actual = 0
            for i in range(len(predictions)):
                print('INPUT')
                print(simple[i])
                print('---------')
                print('EXPECTED')
                print(normal[i])
                print('---------')
                print('PREDICTED')
                print(model.print_pred(predictions[i]))
                print()
                length_dif_sum_pred += (len(model.print_pred(predictions[i]).split(' ')) - len(simple[i].split(' ')))
                length_dif_sum_actual += (len(normal[i].split(' ')) - len(simple[i].split(' ')))
            print('Average length differences for predicted and actual.')
            print('Actual: {}'.format(length_dif_sum_actual / len(predictions)))
            print('Predicted: {}'.format(length_dif_sum_pred / len(predictions)))
            with open('eval_predictions.txt', 'w') as of:
                for pred in predictions:
                    of.write(model.print_pred(pred))
                    of.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests the sentence verbosity model.')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='trains model')
    command_parser.add_argument('-r', '--resume', action='store_true', default=False, help="Resume training with existing model")
    command_parser.add_argument('-m', '--buildmodel', action='store_true', default=False,
                                help="Build model and exit")
    command_parser.add_argument('-l', '--large', action='store_true', default=False,
                                help="Larger model")
    command_parser.add_argument('-a', '--attention', action='store_true', default=False, help="Use attention")
    command_parser.add_argument('-b', '--bidirectional', action='store_true', default=False, help="Use bidirectional")
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('eval', help='evaluate model')
    command_parser.add_argument('-a', '--attention', action='store_true', default=False, help="Use attention")
    command_parser.add_argument('-l', '--large', action='store_true', default=False,
                                help="Larger model")
    command_parser.add_argument('-b', '--bidirectional', action='store_true', default=False, help="Use bidirectional")
    command_parser.add_argument('-bs', '--beamsearch', action='store_true', default=False,
                                help="Use beam search decoding")
    command_parser.set_defaults(func=evaluate)

    command_parser = subparsers.add_parser('load', help='Load and split data into pieces')
    command_parser.set_defaults(func=load_and_split)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)