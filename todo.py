import torch
from config import config
from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config
from model import sequence_labeling
from tqdm import tqdm_notebook as tqdm
from todo import evaluate
import torch
#from ipywidgets import IntProgress
from randomness import apply_random_seed
_config = config()


def evaluate(golden_list, predict_list):
	def evaluate(golden_list, predict_list):
    a = []
    for i in golden_list:
        a += i
    b = []
    for i in predict_list:
        b += i
    TP, FP, FN = 0, 0, 0
    for i,j in zip(a,b):
        if i=="O" and j != "O":
            FP +=1
        elif i!="O" and j == "O":
            FN +=1
        elif i!="O" and j != "O" and i != j:
            FP += 1
            FN += 1
        elif i == j and i != "O":
            TP += 1
#     afn = np.sum(FN)
    if TP+FP ==0:
        precision =0
    else:
        precision = TP/(TP+FP)
    if TP+FN==0:
        recall=0
    else:    
        recall = TP/(TP+FN)
    if precision+recall==0:
        f1 = 0
    else:
        f1 = ((precision*recall)/(precision+recall)) #np.mean((precision*recall)/(precision+recall))
    return f1

# print(evaluate(golden_list,predict_list))

	print(golden_list)
	print(predict_list)


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
 
    
#     Reshape to (14,14)
    batch_char_index_matrices = np.reshape(batch_char_index_matrices, [14,14])
    
#   Making embeddings of 14,14,50  
    batch_char_embedding = np.zeros((14,14,50))
    batch_char_index_matrices, idxs = list(zip(*sorted(zip(batch_char_index_matrices, np.arange(14)),
                                                 key = lambda x: np.count_nonzero(x[0]))))
    for i, batch_char_idx in enumerate(batch_char_index_matrices):
        for j, idx in enumerate(batch_char_idx):
            batch_char_embedding[i][j] = char_embedding[idx]
    print("Embedding Shape:", batch_char_embedding.shape)
    
#     Make hidden Layes
    def init_hidden(model):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2,14, 50)),
                Variable(torch.zeros(2,14, 50)))
    
#     Make model
    model.hidden = init_hidden(model)
    model.BiLSTM= torch.nn.LSTM(input_size=50, hidden_size=50, bidirectional=True)
    lstm_out, _ = model.BiLSTM(torch.tensor(batch_char_embedding).float(), model.hidden)
    
#     Take the Last LSTM output
    lstm_out = lstm_out[-1]
    lstm_out = lstm_out.data.numpy()
    
#     Convert back to original order
    orignal_out = np.zeros((14,100))
    for i, idx in enumerate(idxs):
        orignal_out[idx] = lstm_out[i]
    orignal_out = np.reshape(orignal_out, (2,7,100))
    
    print("lstm",orignal_out.shape)
    return torch.tensor(orignal_out)

#     #return result
#     pass
