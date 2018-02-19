from util import embedding_util as embedder
from util import parser_util as parser
from src.seq2seq_model import Seq2SeqModel, Config
import tensorflow as tf
import time
import numpy as np

def split_train_dev(data):
    split = int(len(data) * 0.85)
    return data[:split], data[split:]

def main():
    # simple and normal are just a list of sentences (as strings)
    pretrained_embeddings, normal, simple, tok2id = embedder.load_embeddings()
    data = parser.make_seq2seq_data(simple, normal, embedder.START, embedder.END, embedder.PAD, tok2id)
    print(embedder.id2tok)
    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        print("Building model...",)
        start = time.time()
        config = Config(embedder.embed_size, embedder.vocab_size,
                        parser.max_encoder_timesteps, parser.max_decoder_timesteps)
        model = Seq2SeqModel(config, pretrained_embeddings)
        print("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        train, dev = split_train_dev(data)

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev, pad_tokens=[embedder.PAD, embedder.END])
            predictions, loss = model.evaluate(session, dev, pad_tokens=[embedder.PAD, embedder.END], write_preds=True)
            print("Overall mean incorrect predictions on dev set: " + str(loss))
            input_sentences = [d[0] for d in dev]
            label_sentences = [d[1] for d in dev]
            print('Average sentence length input: {}, labels: {}, predictions: {}'.format(
                np.mean([len(sentence) for sentence in input_sentences]),
                np.mean([len(sentence) for sentence in label_sentences]),
                np.mean([len(sentence.split(' ')) for sentence in predictions])
            ))



if __name__ == '__main__':
    main()