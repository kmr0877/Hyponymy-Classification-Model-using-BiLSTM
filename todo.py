import torch
from config import config
from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config
from model import sequence_labeling
from tqdm import tqdm_notebook as tqdm
from todo import evaluate
import torch
from torch.autograd import Variable
import torch.nn.functional as F
#from ipywidgets import IntProgress
from randomness import apply_random_seed
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
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
    
    
    perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_len_lists.view(-1))
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    batch_char_index_matrices = batch_char_index_matrices.view((batch_word_len_lists.size()[0]*batch_word_len_lists.size()[1], -1))
    char_embedding = model.char_embeds(batch_char_index_matrices)
    sorted_input_embeds = char_embedding[perm_idx]
  
    output_sequence = pack_padded_sequence(torch.tensor(sorted_input_embeds), 
                                            lengths=sorted_batch_word_len_lists.data.tolist(), batch_first=True)
    
#   Make embeddings of 14,14,50  

    lstm_out, state = model.char_lstm(output_sequence.float())
    lstm_out, state = pad_packed_sequence(lstm_out,batch_first=True)
    temp = []
    for i in range(len(sorted_batch_word_len_lists)):
        temp.append(torch.cat([lstm_out[i, sorted_batch_word_len_lists[i] - 1, :model.config.char_lstm_output_dim], lstm_out[i, 0, model.config.char_lstm_output_dim:]],-1).unsqueeze(0))
        
 #     Used state 0       
    answer = torch.cat(temp,0)
    answer = answer[desorted_indices]


#     answer = torch.cat((state[0][0],state[0][1] ),1)
    

    
    return answer.view((batch_word_len_lists.size()[0],batch_word_len_lists.size()[1],-1))


#     #return result
#     pass