# Hyponymy-Classification-Model-using-BiLSTM

# Objective
In this project, you need to build a system that can extract hyponym and hypernym from a sentence.

For example, in sentence Stephen Hawking is a physicist ., phrase Stephen Hawking is the hyponym of physicist, and physicist is the hypernym of Stephen Hawking.

We formulate the problem as a sequence tagging task where Input is a sentence formed as a sequence of words with length  l.

output is a tag sequence with length  l.

We use IOB2) scheme to represent the results (i.e., encode the tags), as both hypernyms and hyponyms can be phrases.

Thus, in the above example,
[Stephen, Hawking, is, a, physicist, .] is the input word list

[B-TAR, I-TAR, O, O, B-HYP, O] is the corresponding output tag sequence, 

where TAR corresponds to hyponyms and HYP corresponds to hypernyms.
