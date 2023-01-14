import pandas as pd
import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer

class SentenceGetter(object):
    def __init__(self, data):
        # define this sentence counter for get_next method 
        self.n_sent = 1
        
        # save the data itself
        self.data = data
        
        # define attribute that checks if data is empty
        if not data.empty:
            self.empty = False
        else:
            self.empty = True
        
        # function that creates tuples containing a Word and its corresponding tag / label
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        
        # first group by the sentence number and then apply tuples to them (see below)
        self.grouped = self.data.groupby("SentenceNumber").apply(agg_func)
        
        # put sentences (tuples) in a list
        self.sentences = [s for s in self.grouped]

    # iterator function for the sentences
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def define_tag_values(data):
    """this method determines the tag_values and the tag2idx datastructure"""
    
    # create set of tag / label values and save it in a list
    tag_values = sorted(list(set(data["Tag"].values))) # sorted such that the order is always the same when the labels are the same

    # add "PAD" to the tag values list
    # later we will pad sequences of the labels. Thereby we will use "PAD" as the padding value
    tag_values.append("PAD") 

    # assign each tag a unique ID (probably because NNs only recognize numbers as inputs)
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    print("Following tags and their corresponding IDs will be used: ", tag2idx, "\n")

    return tag_values, tag2idx

def load_bert_tokenizer(bert_model):
    """bert tokenizer is loaded.
    The tokenizer does not lower case the inputs.
    """

    if not "roberta" in bert_model:
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(bert_model, do_lower_case=False)

    return tokenizer

def tokenize_and_preserve_labels(sentences: list, labels: list, tokenizer, sequence_length=512):
    """this method tokenizes a sentence into 'bert-tokens' based on the wordpiece vocabulary 
    (BERT tokenizes large words into smaller components),
    accordingly the text labels are ordered w.r.t. these bert-tokens.
    
    Args:
        sentence (list): a list containing all tokens of a sentence in the correct chronological order
        text_labels (list): a list containing all text labels of a sentence in the correct chronological order (with respect to sentence)
    
    Returns:
        list, list: bert-tokenized sentences and their corresponding labels
    """
    tokenized_sentences = []
    new_labels = []
    filter_masks = []
    orig_to_tok_map = []
    input_masks = []

    for sen, sen_labels in zip(sentences, labels):
        t = ["[CLS]"]
        l = ["O"]
        a = [0] # pytorch crf wants the first element of the attention mask to be 1, but for the filter mask its ok
        o = []
        i = [1]
        for word, label in zip(sen, sen_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            if not tokenized_word:
                tokenized_word = ['[UNK]']
            for bert_token in tokenized_word:
                l.append(label)
                t.append(bert_token)
                i.append(1)
                # a.append(1) # or 0 in lower if and 1 in else 
                if bert_token.startswith("##"):
                    #l.append(label) # or "X"
                    a.append(0)
                else:
                    #l.append(label)
                    a.append(1)
                    o.append(len(t) - 1)

        #truncate
        if len(t) > sequence_length - 1:
            t = t[0:(sequence_length - 1)]
            l = l[0:(sequence_length - 1)]
            a = a[0:(sequence_length - 1)]
            i = i[0:(sequence_length - 1)]
            new_o = []
            for index in o:
                if index < sequence_length - 1:
                    new_o.append(index)
                else:
                    break
            o = new_o

        t.append('[SEP]')
        l.append('O')
        a.append(0) 
        i.append(1)

        assert len(t) == len(l) and len(l) == len(a), "mistake happened"

        tokenized_sentences.append(t)
        new_labels.append(l)
        filter_masks.append(a)
        input_masks.append(i)
        orig_to_tok_map.append(o)

    return tokenized_sentences, new_labels, input_masks, filter_masks, orig_to_tok_map

def roberta_tokenize_and_preserve_labels(sentences: list, labels: list, tokenizer, sequence_length=512):
    """this method tokenizes a sentence into 'tokens' based on BPE
    
    Args:
        sentence (list): a list containing all tokens of a sentence in the correct chronological order
        text_labels (list): a list containing all text labels of a sentence in the correct chronological order (with respect to sentence)
    
    Returns:
        list, list: bert-tokenized sentences and their corresponding labels
    """
    tokenized_sentences = []
    new_labels = []
    filter_masks = []
    orig_to_tok_map = []
    input_masks = []

    for sen, sen_labels in zip(sentences, labels):
        t = ["<s>"]
        l = ["O"]
        a = [0] # pytorch crf wants the first element of the attention mask to be 1, but for the filter mask its ok
        o = []
        i = [1]
        for word, label in zip(sen, sen_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            if not tokenized_word:
                tokenized_word = ['<unk>']
            first_token = True
            for bert_token in tokenized_word:
                l.append(label)
                t.append(bert_token)
                i.append(1)
                # a.append(1) # or 0 in lower if and 1 in else 
                if not first_token:
                    #l.append(label) # or "X"
                    a.append(0)
                else:
                    #l.append(label)
                    a.append(1)
                    o.append(len(t) - 1)
                    first_token = False

        #truncate
        if len(t) > sequence_length - 1:
            t = t[0:(sequence_length - 1)]
            l = l[0:(sequence_length - 1)]
            a = a[0:(sequence_length - 1)]
            i = i[0:(sequence_length - 1)]
            new_o = []
            for index in o:
                if index < sequence_length - 1:
                    new_o.append(index)
                else:
                    break
            o = new_o

        t.append('</s>')
        l.append('O')
        a.append(0) 
        i.append(1)

        assert len(t) == len(l) and len(l) == len(a), "mistake happened"

        tokenized_sentences.append(t)
        new_labels.append(l)
        filter_masks.append(a)
        input_masks.append(i)
        orig_to_tok_map.append(o)

    return tokenized_sentences, new_labels, input_masks, filter_masks, orig_to_tok_map


def count_label_occurences(labels):
    output_dict = {}
    for l1 in labels:
        for l2 in l1:
            if l2 not in output_dict:
                output_dict[l2] = 1
            else:
                output_dict[l2] += 1
    print(output_dict)

def sequence_filler(sentences, labels, tokenizer, bert_model, sequence_length=512):
    init_len = len(sentences)

    new_sentences = []
    new_labels = []

    filled_sentences = []
    filled_labels = []
    current_length = 0
    sentence_counter = 0
    for s, l in zip(sentences, labels):
        bert_length_of_current_sentence = get_length_of_bert_sentence(s, tokenizer=tokenizer, bert_model=bert_model)
        current_length += bert_length_of_current_sentence
        if current_length > sequence_length - 2: # for CLS and SEP tokens
            if sentence_counter != 0:
                new_sentences.append(filled_sentences)
                new_labels.append(filled_labels)

                filled_sentences = s
                filled_labels = l

                current_length = bert_length_of_current_sentence
                sentence_counter = 1
            else:
                filled_sentences.extend(s)
                filled_labels.extend(l)

                new_sentences.append(filled_sentences)
                new_labels.append(filled_labels)

                filled_sentences = []
                filled_labels = []

                current_length = 0
                sentence_counter = 0
        else:
            filled_sentences.extend(s)
            filled_labels.extend(l)

            sentence_counter += 1 

    if filled_sentences:
        if not filled_labels:
            print("Mistake in sequence filler method")
        new_sentences.append(filled_sentences)
        new_labels.append(filled_labels)
    return new_sentences, new_labels

def get_length_of_bert_sentence(sentence, tokenizer, bert_model):
    length = 0
    for word in sentence:
        tokenized_word = tokenizer.tokenize(word)
        if not tokenized_word:
            if "roberta" not in bert_model:
                tokenized_word = ['[UNK]']
            else:
                tokenized_word = ['<unk>']
        length += len(tokenized_word)
    return length

def count_sentences_and_labels(labels):
    sentence_counter = len(labels)
    labels_counter = 0
    for l in labels:
        labels_counter += len(l)
    print()
    print("Total sentences: ", sentence_counter)
    print("Total labels: ", labels_counter)
    print()

def count_label_occurences(labels):
    output_dict = {}
    for l1 in labels:
        for l2 in l1:
            if l2 not in output_dict:
                output_dict[l2] = 1
            else:
                output_dict[l2] += 1
    print(output_dict)

def get_dataloader(path, data_kind="train", tag_values=None, bert_model="bert-base-cased", dev=False, sequence_length=128, batch_size=16):
    data = pd.read_csv(path, encoding="utf-8").fillna(method="ffill")

    getter = SentenceGetter(data)

    # define the sentences by always picking first elements of the tuples of one sentence index
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]

    # define the labels for each word of the sentences by always picking second elements of the tuples of one sentence index
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]

    # for development mode: take subset of dataset (1/8) -> faster computation -> easier debugging
    if dev:
        sentences = sentences[:int(len(sentences)/8)]
        labels = labels[:int(len(labels)/8)]

    count_label_occurences(labels)
    count_sentences_and_labels(labels)

    if tag_values is None:
        # get the set of entities and their corresponding ids
        tag_values, tag2idx = define_tag_values(data)

        # saving tag values and tag2idx such that they can be loaded when doing checkpoint training or testing or raw prediction
        # with open('saved_datastructures/tag_values_tag2idx.pkl', 'wb') as pickle_save: # drive/MyDrive/HLE-project/saved_datastructures/tag_values_tag2idx.pkl
            # pickle.dump([tag_values, tag2idx], pickle_save, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        tag2idx = {t: i for i, t in enumerate(tag_values)}
        print("Following tags and their corresponding IDs will be used: ", tag2idx, "\n")

    # load tokenizer
    tokenizer = load_bert_tokenizer(bert_model)

    # fill sequences up until maximum sequence length
    sentences, labels = sequence_filler(sentences, labels, tokenizer=tokenizer, bert_model=bert_model, sequence_length=sequence_length)

    # tokenizing and preserving labels of the data
    if "roberta" not in bert_model:
        tokenized_sentences, labels, in_masks, filt_masks, _ = tokenize_and_preserve_labels(sentences, labels, tokenizer=tokenizer, sequence_length=sequence_length)
    
        # tokenizer.convert_tokens_to_ids(txt): convert the tokens to ids by using BERTs local dictionary
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences], # List of sequences: each splittet token by wordpiece bert tokenizer
                                  maxlen=sequence_length, # this is the length of how long the arrays need to be
                                  dtype="long", # Type of the output sequences
                                  value=0, # this is the value of the padding
                                  truncating="post", # remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences
                                  padding="post") # padding will be added at the end of arrays

    else:
        tokenized_sentences, labels, in_masks, filt_masks, _ = roberta_tokenize_and_preserve_labels(sentences, labels, tokenizer=tokenizer, sequence_length=sequence_length)

        # tokenizer.convert_tokens_to_ids(txt): convert the tokens to ids by using BERTs local dictionary
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(sen) for sen in tokenized_sentences], # List of sequences: each splittet token by wordpiece bert tokenizer
                                  maxlen=sequence_length, # this is the length of how long the arrays need to be
                                  dtype="long", # Type of the output sequences
                                  value=1, # this is the value of the padding in roberta
                                  truncating="post", # remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences
                                  padding="post") # padding will be added at the end of arrays

    # look above for clearer explanation
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=sequence_length, 
                         dtype="long", 
                         value=tag2idx["PAD"],
                         truncating="post",
                         padding="post")

    input_masks = pad_sequences([[m for m in mask] for mask in in_masks],
                         maxlen=sequence_length, 
                         dtype="long", 
                         value=0,
                         truncating="post",
                         padding="post")

    filter_masks = pad_sequences([[m for m in mask] for mask in filt_masks],
                         maxlen=sequence_length, 
                         dtype="long", 
                         value=0,
                         truncating="post",
                         padding="post")

    #print(input_ids[0])
    #print(tags[0])
    #print(input_masks[0])
    #print(filter_masks[0])

    # Since we’re operating in pytorch, we have to convert the dataset to torch tensors
    inputs = torch.LongTensor(input_ids) # tr_inputs = torch.tensor(tr_inputs)
    tags = torch.LongTensor(tags) # tr_tags = torch.tensor(tr_tags)
    masks = torch.ByteTensor(input_masks) # tr_masks = torch.tensor(tr_masks)
    filter_masks = torch.ByteTensor(filter_masks)

    # Dataset wrapping tensors. Each sample will be retrieved by indexing tensors along the first dimension.
    data_for_dataloader = TensorDataset(inputs, masks, tags, filter_masks)

    if data_kind=="train":
        # shuffle the data at training time with the RandomSampler
        data_sampler = RandomSampler(data_for_dataloader)
    else:
        # for validation and test pass the data sequentially with the SequentialSampler
        data_sampler = SequentialSampler(data_for_dataloader)

    dataloader = DataLoader(data_for_dataloader, sampler=data_sampler, batch_size=batch_size)

    return dataloader, tag_values, tag2idx