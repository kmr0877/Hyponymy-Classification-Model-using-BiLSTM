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
    
    # Given an input of the size [2,7,14], we will convert it a minibatch of the shape [14,14] to 
    # represent 14 words(7 in each sentence), and 14 characters in each word.
    
    ## NOTE: Please DO NOT USE for Loops to iterate over the mini-batch.
    
    
    # Get corresponding char_Embeddings, we will have a Final Tensor of the shape [14, 14, 50]
    # Sort the mini-batch wrt word-lengths, to form a pack_padded sequence.
    # Feed the pack_padded sequence to the char_LSTM layer.
    
    
    # Get hidden state of the shape [2,14,50].
    # Recover the hidden_states corresponding to the sorted index.
    # Re-shape it to get a Tensor the shape [2,7,100].
    
#     Reshape to (14,14)
    
    perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_len_lists.view(14))
    sorted_input_embeds = char_embedding[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    batch_char_index_matrices = batch_char_index_matrices.view(14,14)
    sorted_input_embeds = char_embedding[batch_char_index_matrices]
    print(sorted_input_embeds.shape)
    output_sequence = pack_padded_sequence(torch.tensor(sorted_input_embeds), 
                                            lengths=sorted_batch_word_len_lists.data.tolist(), batch_first=True)
    
#   Make embeddings of 14,14,50  

    hidden = (torch.ones(2, 14, 100), torch.ones(2, 14, 100)) 
    model.BiLSTM = nn.LSTM(50,100,  bidirectional=True)
    lstm_out, state = model.BiLSTM(output_sequence.float(), hidden)
#     lstm_out, _ = model.BiLSTM(torch.tensor(batch_char_embedding).float())
    
#     Take the Last LSTM output
    lstm_out = lstm_out[-1]
#     lstm_out = lstm_out

    print("packed:", output_sequence[0].shape)
    output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)

    output_sequence = output_sequence[desorted_indices]
    print(output_sequence.shape)
    output_sequence = model.non_recurrent_dropout(output_sequence)

    print("lstm",output_sequence.shape)
    return output_sequence

#     #return result
#     pass