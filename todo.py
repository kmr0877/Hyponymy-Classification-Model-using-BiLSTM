import torch
from config import config
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
_config = config()


def evaluate(golden_list, predict_list):
    a = []
    for i in golden_list:
        a += i
    b = []
    for i in predict_list:
        b += i
    idx_dict = dict(zip(set(a),np.arange(5)))
    n = len(set(a))
    conf = np.zeros((n, n))
    for i,j in zip(a,b):
        conf[idx_dict[i], idx_dict[j]] += 1 
    TP = np.diag(conf)
    FP = np.sum(conf, axis=0) - TP
    FN = np.sum(conf, axis=1) - TP
    atp = np.mean(TP)
    afp = np.mean(FP)
    afn = np.mean(FN)
    precision = atp/(atp+afp)
    recall = atp/(atp+afn)
    return np.mean((precision*recall)/(precision+recall))



def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1-forgetgate) * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy

def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    

    def init_hidden(model):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2, batch_word_len_lists.shape[1], 50)),
                Variable(torch.zeros(2, batch_word_len_lists.shape[1], 50)))
    model.hidden = init_hidden(model)
#     print(batch_char_index_matrices.shape[2], batch_word_len_lists.shape)
    model.BiLSTM= torch.nn.LSTM(input_size=batch_char_index_matrices.shape[2], hidden_size=50, bidirectional=True)
    lstm_out, _ = model.BiLSTM(batch_char_index_matrices.float(), model.hidden)
#     print("lstm",lstm_out.shape)
    return lstm_out

