import os
import numpy as np

class Config(object):
    normal_file = '..//data//normal.aligned'
    simple_file = '..//data//simple.aligned'
    pwkp_file = '..//data//PWKP_108016.txt'

max_encoder_timesteps = 0
max_decoder_timesteps = 0


# start, end are vectors for start and end tokens
def make_seq2seq_data(simple, normal, start, end, pad, tok2id):
    global max_decoder_timesteps
    global max_encoder_timesteps
    assert(len(simple) == len(normal))
    data = []
    for i in range(len(simple)):
        enc = [tok2id[pad]] * max_encoder_timesteps
        sentence = simple[i].split(' ')
        offset = max_encoder_timesteps - len(sentence)
        for j in range(offset, max_encoder_timesteps):
            enc[j] = tok2id[sentence[j - offset]]

        dec = [tok2id[end]] * max_decoder_timesteps
        sentence = normal[i].split(' ')
        for j in range(len(sentence)):
            dec[j] = tok2id[sentence[j]]

        data.append((enc, [tok2id[start]] + dec, dec + [tok2id[end]]))
    max_decoder_timesteps += 1
    return data

#returns list of sentences for simple and aligned
#sentences are a single string
def parse_aligned():
    config = Config()
    normal = []
    with open(config.normal_file) as f:
        for line in f.readlines():
            sp = line.strip().split('\t')
            normal.append(sp[2])
    simple = []
    with open(config.simple_file) as f:
        for line in f.readlines():
            sp = line.strip().split('\t')
            simple.append(sp[2])
    return normal, simple

def parse_pwkp():
    global max_encoder_timesteps
    global max_decoder_timesteps
    config = Config()
    normal = []
    simple = []
    count = 0
    with open(config.pwkp_file, 'rb') as f:
        parsing_normal = True
        simple_sentence = ''
        parse_head = ['', '']
        i = 0
        for ln in f:
            if i > 100:
                break
            i+=1
            decoded = False
            sentence = ''
            for cp in ('cp1252', 'cp850', 'utf-8', 'utf8'):
                try:
                    sentence = ln.decode(cp)
                    decoded = True
                    break
                except UnicodeDecodeError:
                    pass
            if decoded:
                sentence = sentence.strip()
                if sentence == '':
                    parse_head[1] = simple_sentence.strip()
                    max_encoder_timesteps = max(max_encoder_timesteps, len(simple_sentence))
                    if parse_head[0] != '' and parse_head[1] != '':
                        normal.append(parse_head[0])
                        simple.append(parse_head[1])
                    parse_head = ['', '']
                    simple_sentence = ''
                    parsing_normal = True
                elif parsing_normal:
                    parse_head[0] = sentence
                    max_decoder_timesteps = max(max_decoder_timesteps, len(sentence))
                    parsing_normal = False
                else:
                    simple_sentence += ' ' + sentence
    return normal, simple
