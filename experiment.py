from parser_util import make_seq2seq_data, make_seq2seq_data_v2, make_fill_blank_data
import parser_util
import embedding_util as embedder
from seq2seq_model import Seq2SeqModel
from fill_model import FillModel, Config
import tensorflow as tf
import time, sys, random, os
import numpy as np
from threading import Thread
import argparse
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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

def plot_svd(X, config):
    U, s, Vh = np.linalg.svd(X, full_matrices=True)
    for i in range(len(X)):
        plt.text(U[i,0], U[i,1], '{}'.format(i))
    plt.savefig('models/{}/dim_reduction.png'.format(config))



def eval_model(model, data, config, id2tok, session, cache=False, mode='test'):
    predictions, cache_weights, loss = model.evaluate_fill(session, data, pad_tokens=[embedder.PAD, embedder.END])
    print('Loss = {}'.format(loss))
    acc_count = 0.0
    with open('models/{}/{}_predictions.txt'.format(config, mode), 'w') as of:
        count = 0
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                input, label, _ = data[count]
                acc_count += int(label == predictions[i][j])
                try:
                    of.write(' '.join([id2tok[tok] for tok in input]) + '\n')
                    of.write('ACTUAL: {}, PREDICTED: {}\n'.format(id2tok[label], id2tok[predictions[i][j]]))
                    of.write('\n')
                except:
                    print('failed to write character')
                count += 1
    print('Accuracy = {}'.format(acc_count / count))
    if cache:
        ## visualize cache attention
        with open('models/{}/{}_cache_attention.txt'.format(config, mode), 'w') as of:
            for i in range(len(cache_weights)):
                of.write('{}\n'.format(cache_weights[i]))
        with open('models/{}/{}_cache_sentences.txt'.format(config, mode), 'w') as of:
            for i in range(len(model.cache_sentences)):
                try:
                    of.write(' '.join([id2tok[tok] for tok in model.cache_sentences[i]]) + '\n')
                except:
                    print('failed to write character')
        with open('models/{}/{}_cache_vectors.txt'.format(config, mode), 'w') as of:
            for i in range(len(model.cache)):
                for j in range(len(model.cache[i])):
                    of.write('{}\t'.format(model.cache[i][j]))
                of.write('\n')
        plot_svd(model.cache, config)


def evaluate_v2(args):
    pretrained_embeddings, _, _, tok2id, id2tok = embedder.load_embeddings(large=args.large, mode='full')
    config = Config(len(pretrained_embeddings[0]), len(pretrained_embeddings),
                    parser_util.max_normal_timesteps, parser_util.max_simple_timesteps,
                    embedder.PAD, tok2id[embedder.START], tok2id[embedder.END], args.attention, args.bidirectional,
                    id2tok, cache=args.cache,
                    large=args.large, mode='test')
    with open('data/test.txt') as f:
        test = f.readlines()
    with open('data/dev.txt') as f:
        dev = f.readlines()

    test_data, dev_data = make_fill_blank_data(test, dev, embedder.PAD, tok2id, id2tok=id2tok)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        saver = tf.train.Saver()
        model = FillModel(config, pretrained_embeddings)
        with tf.Session() as session:
            saver.restore(session, 'models/{}/fill_model.ckpt'.format(config))
            eval_model(model, test_data, config, id2tok, session, cache=args.cache, mode='test')
            eval_model(model, dev_data, config, id2tok, session, cache=args.cache, mode='dev')
        

def train_v2(args):
    pretrained_embeddings, _, _, tok2id, id2tok = embedder.load_embeddings(large=args.large, mode='full')
    #normal, simple = parser_util.parse_pwkp(mode='train')
    mode = 'train'
    if args.resume:
        mode = 'restore'
    config = Config(len(pretrained_embeddings[0]), len(pretrained_embeddings),
                    parser_util.max_normal_timesteps, parser_util.max_simple_timesteps,
                    embedder.PAD, tok2id[embedder.START], tok2id[embedder.END], args.attention, args.bidirectional,
                    id2tok, cache=args.cache,
                    large=args.large, mode=mode)
    if not args.resume:
        try:
            os.mkdir('models/{}'.format(config))
        except:
            pass
        with open('models/{}/dev_predict.txt'.format(config), 'w') as of:
            of.write('\n')
        with open('models/{}/train_loss.txt'.format(config), 'w') as of:
            of.write('\n')
        with open('models/{}/dev_loss.txt'.format(config), 'w') as of:
            of.write('\n')
        with open('models/{}/training_output.txt'.format(config), 'w') as of:
            print('Start: {}\n'.format(time.time()))
            print('Beginning train with params: {}\n\n'.format(config))
            of.write('Beginning train with params: {}\n\n'.format(config))

    with open('data/train.txt') as f:
        train = f.readlines()
    with open('data/dev.txt') as f:
        dev = f.readlines()

    train_data, dev_data = make_fill_blank_data(train, dev, embedder.PAD, tok2id, id2tok=id2tok)
    run_session(args, config, pretrained_embeddings, train_data, dev_data)


def grid_search(args, config, pretrained_embeddings, normal_data, simple_data):
    for hs in [256]:
        for nl in [2]:
            for d in [0.0, 0.2, 0.5, 0.8]:
                config.hidden_size = hs
                config.n_layers = nl
                config.dropout = d
                run_session(args, config, pretrained_embeddings, normal_data, simple_data)

def run_session(args, config, pretrained_embeddings, train_data, dev_data):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            print("Building model...", )
            start = time.time()
            model = FillModel(config, pretrained_embeddings)
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            saver = tf.train.Saver(save_relative_paths=True)
            print("took %.2f seconds", time.time() - start)
            with tf.Session() as session:
                print('initialized variables')
                if args.resume:
                    print('resuming from previous checkpoint')
                    saver.restore(session, 'models/{}/fill_model.ckpt'.format(config))
                else:
                    session.run(init)
                if args.buildmodel:
                    exit(0)
                for epoch in range(config.n_epochs):
                    model.fit_fill(session, saver, None, train_data, dev_data, pad_tokens=[embedder.PAD, embedder.END],
                               epoch=epoch)



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
    command_parser.add_argument('-gs', '--gridsearch', action='store_true', default=False, help="Do param grid search")
    command_parser.add_argument('-c', '--cache', action='store_true', default=False, help="Use cache")
    command_parser.set_defaults(func=train_v2)

    command_parser = subparsers.add_parser('eval', help='evaluate model')
    command_parser.add_argument('-a', '--attention', action='store_true', default=False, help="Use attention")
    command_parser.add_argument('-l', '--large', action='store_true', default=False,
                                help="Larger model")
    command_parser.add_argument('-b', '--bidirectional', action='store_true', default=False, help="Use bidirectional")
    command_parser.add_argument('-bs', '--beamsearch', action='store_true', default=False,
                                help="Use beam search decoding")
    command_parser.add_argument('-c', '--cache', action='store_true', default=False, help="Use cache")
    command_parser.set_defaults(func=evaluate_v2)

    command_parser = subparsers.add_parser('load', help='Load and split data into pieces')
    command_parser.set_defaults(func=load_and_split)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


'''np.random.shuffle(normal)
    train = normal[:int(0.75 * len(normal))]
    dev = normal[int(0.75 * len(normal)) : int(0.8 * len(normal))]
    test = normal[int(0.8 * len(normal)):]
    with open('data/train.txt', 'w') as of:
        for sentence in train:
            try:
                of.write(sentence)
                of.write('\n')
            except:
                print('unwritable character')
    with open('data/dev.txt', 'w') as of:
        for sentence in dev:
            try:
                of.write(sentence)
                of.write('\n')
            except:
                print('unwritable character')
    with open('data/test.txt', 'w') as of:
        for sentence in test:
            try:
                of.write(sentence)
                of.write('\n')
            except:
                print('unwritable character')
    exit()'''
