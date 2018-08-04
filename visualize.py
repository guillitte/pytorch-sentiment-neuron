import argparse
import ast
import math
import pickle
import timeit
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def batchify(data, bsz):
    tokens = len(data.encode())
    ids = torch.LongTensor(tokens)
    token = 0
    for char in data.encode():
        ids[token] = char
        token += 1
    nbatch = ids.size(0) // bsz
    ids = ids.narrow(0, 0, nbatch * bsz)
    ids = ids.view(bsz, -1).t().contiguous()
    return ids, data


def color(p):
    p = math.tanh(3 * p) * .5 + .5
    q = 1. - p * 1.3
    r = 1. - abs(0.5 - p) * 1.3 + .3 * q
    p = 1.3 * p - .3
    i = int(p * 255)
    j = int(q * 255)
    k = int(r * 255)
    if j < 0:
        j = 0
    if k < 0:
        k = 0
    if k > 255:
        k = 255
    if i < 0:
        i = 0
    return ('\033[38;2;%d;%d;%dm' % (j, k, i)).encode()


def make_cuda(state):
    if isinstance(state, tuple):
        return (state[0].cuda(), state[1].cuda())
    else:
        return state.cuda()


def rnn_infer(text, states, batch, embed, rnn):
    if isinstance(states, tuple):
        hidden, cell = states
    else:
        hidden = states
    last = hidden.size(0) - 1
    if opt.layer <= last and opt.layer >= 0:
        last = opt.layer

    if opt.cuda:
        batch = batch.cuda()
        states = make_cuda(states)
        embed.cuda()
        rnn.cuda()

    sentiment_values = []
    gen = bytearray()
    for t in range(text.size(0)):
        emb = embed(batch[t])
        ni = (batch[t]).data[0]
        states, output = rnn(emb, states)
        if isinstance(states, tuple):
            hidden, cell = states
        else:
            hidden = states
        feat = hidden.data[last, 0, opt.neuron]
        sentiment_values.append(feat.data.tolist())
        if ni < 128:
            col = color(feat)
            gen += (col)
        gen.append(ni)

    if opt.temperature == 0:
        topv, topi = output.data.topk(1)
        ni = topi[0][0]
        gen.append(ni)
        inp = Variable(topi[0], volatile=True)
        if opt.cuda:
            inp = inp.cuda()

        for t in range(opt.seq_length):

            emb = embed(inp)
            states, output = rnn(emb, states)
            topv, topi = output.data.topk(1)
            ni = topi[0][0]
            gen.append(ni)
            inp = Variable(topi[0])
            if opt.cuda:
                inp = inp.cuda()

    else:
        probs = F.softmax(output[0].squeeze().div(opt.temperature)).data.cpu()
        ni = torch.multinomial(probs, 1)[0]
        feat = hidden.data[last, 0, opt.neuron]

        if ni < 128:
            col = color(feat)
            gen += (col)

        gen.append(ni)

        inp = Variable(torch.LongTensor([ni]), volatile=True)
        if opt.cuda:
            inp = inp.cuda()
        for t in range(opt.seq_length):

            emb = embed(inp)
            states, output = rnn(emb, states)
            if isinstance(states, tuple):
                hidden, cell = states
            else:
                hidden = states
            feat = hidden.data[last, 0, opt.neuron]
            if isinstance(output, list):
                output = output[0]
            probs = F.softmax(output.squeeze().div(opt.temperature)).data.cpu()
            ni = torch.multinomial(probs, 1)[0]

            if ni < 128:
                col = color(feat)
                gen += (col)
            gen.append(ni)
            inp = Variable(torch.LongTensor([ni]))
            if opt.cuda:
                inp = inp.cuda()
            if opt.overwrite != 0:
                hidden.data[last, 0, opt.neuron] = opt.overwrite

    return gen, sentiment_values


def main(opt):
    batch_size = 1
    checkpoint = torch.load(opt.load_model)
    embed = checkpoint['embed']
    rnn = checkpoint['rnn']

    if opt.mode == 'save_set':

        # store clean used characters and its predicted sentiment as a list of dicts
        analyzed_texts = []

        # init input texts as list of texts

        if type(opt.init) == str:
            if opt.init[-4:] == '.csv':
                list_texts = pd.read_csv(opt.init)
                list_texts = list_texts['text_en'].tolist()
            else:
                list_texts = ast.literal_eval(opt.init)
        else:
            list_texts = opt.init

        start = timeit.default_timer()
        for t in list_texts:
            try:
                sent_text = {}
                text, text_clean = batchify(t, batch_size)
                batch = Variable(text)
                states = rnn.state0(batch_size)
                gen, sentiment_values = rnn_infer(text, states, batch, embed, rnn)

                # Save cleaned text and sentiment per character into its dict
                sent_text['text'] = text_clean
                sent_text['sentiment'] = sentiment_values
                analyzed_texts.append(sent_text)

            except Exception as exp:
                print(f"There was a failure in text {t}: {exp}")
                pickle.dump(analyzed_texts, open(opt.pickle + '.checkpoint', "wb"))

        pickle.dump(analyzed_texts, open(opt.pickle, "wb"))
        stop = timeit.default_timer()
        time_cost = stop - start
        print(f"It takes {time_cost}s to calculate the sentiment of {len(list_texts)} texts.")


    else:
        text, text_clean = batchify(opt.init, batch_size)
        batch = Variable(text)
        states = rnn.state0(batch_size)

        if opt.mode == 'show':

            gen, sentiment_values = rnn_infer(text, states, batch, embed, rnn)
            gen += ('\033[0m').encode()
            print(gen.decode("utf-8", errors='ignore'))

        elif opt.mode == 'save_text':
            gen, sentiment_values = rnn_infer(text, states, batch, embed, rnn)

            sentiment = {}
            sentiment['text'] = text_clean
            sentiment['sentiment'] = sentiment_values
            pickle.dump(sentiment, open(opt.pickle, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample.py')

    parser.add_argument('-init', default='The meaning of life is ',
                        help="""Initial text or list of texts""")
    parser.add_argument('-mode', choices=['show', 'save_text', 'save_set'], default='show')
    parser.add_argument('-load_model', default='',
                        help="""Model filename to load""")
    parser.add_argument('-seq_length', type=int, default=50,
                        help="""Maximum sequence length""")
    parser.add_argument('-temperature', type=float, default=0.4,
                        help="""Temperature for sampling.""")
    parser.add_argument('-neuron', type=int, default=0,
                        help="""Neuron to read.""")
    parser.add_argument('-overwrite', type=float, default=0,
                        help="""Value used to overwrite the neuron. 0 means don't overwrite.""")
    parser.add_argument('-layer', type=int, default=-1,
                        help="""Layer to read. -1 = last layer""")
    parser.add_argument('-pickle', default="sentiment.pkl",
                        help="""Name of file to save sentiment results""")
    # GPU
    parser.add_argument('-cuda', action='store_true',
                        help="""Use CUDA""")

    opt = parser.parse_args()

    main(opt)
