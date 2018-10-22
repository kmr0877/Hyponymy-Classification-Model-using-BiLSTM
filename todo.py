import torch
from config import config

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
    pass;


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    pass;
evaluate

