import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import torch


def inverse_tanh_cpu(y):
    return 0.5 * (np.log(1 + y) - np.log(1 - y))

def inverse_tanh_cuda(y):
    return 0.5 * (torch.log(1 + y) - torch.log(1-y))

def batch_iterator(X, batch_size = 256):
    l = len(X)
    for idx in range(0, l, batch_size):
        yield X[idx:idx + batch_size]


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    emoji_regex = '&#[0-9]+'
    seperator_regex = ';+|:+'
    quotation_regex = '"+|\-\-+'
    exclamation_regex = '!+'
    dots_regex = '\.\.+'
    dot_regex = ' \.'
    sdot_regex = '\.'
    comma_regex = ','
    RT_regex = 'RT'


    parsed_text = re.sub(quotation_regex, ' ', text_string)
    parsed_text = re.sub(RT_regex, ' ', text_string)
    parsed_text = re.sub(dot_regex, '.', parsed_text)
    parsed_text = re.sub(comma_regex, ' ,', parsed_text)
    parsed_text = re.sub('!', ' !', parsed_text)
    # parsed_text = re.sub('\?', ' \?', parsed_text)
    # parsed_text = re.sub('\\', ' ', parsed_text)
    parsed_text = re.sub(dots_regex, '', parsed_text)
    parsed_text = re.sub(exclamation_regex, '', parsed_text)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(emoji_regex,'', parsed_text)
    parsed_text = re.sub(seperator_regex,'', parsed_text)
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = re.sub(sdot_regex, ' .', parsed_text)
    return parsed_text

def plot_attention_heatmap(text, attention, true_label, predict_label, label_map = None, num_sentence=10, index = None, savename=None, cell_height=0.325, cell_width=0.15, dpi=100, show = False, pause = 0, fig_num = 2):
    if index is None:
        index = np.random.randint(len(text), size = num_sentence)
    else:
        num_sentence = len(index)
        index = np.array(index)

    text = np.array(text)
    text = text[index]

    if isinstance(text[0], str):
        text = [s.split() for s in text]
    
    true_label = np.array(true_label)
    true_label = true_label[index]
    
    predict_label = np.array(predict_label)
    predict_label = predict_label[index]

    if label_map is not None:
        label_map = np.array(label_map)
        true_label = label_map[true_label]
        predict_label = label_map[predict_label]


    attention = attention[index, 1:, :] 

    plt.figure(fig_num, figsize=(19.20,10.80))
    plt.clf()

    for i in range(num_sentence):
        te = text[i]
        num_words = len(te)
        at = attention[i][:num_words]

        te = np.array(te)
        te = te.reshape(1, -1)
        at = at.reshape(1, -1)
        plt.subplot(num_sentence, 1, i+1)
        ax = sns.heatmap(at, annot=te, fmt='', vmin=0, vmax=1, cmap='YlOrRd', xticklabels=False, cbar=False)
        ax.set_yticklabels([str(predict_label[i]) + '/' + str(true_label[i])], rotation = 0)
    
    if show:
        plt.show()
        if pause > 0:
            plt.pause(pause)
    if savename is not None:   
        plt.savefig(savename)

    return index

def plot_pooling_indices_heatmap(text, indices, true_label, predict_label, label_map = None, num_sentence=10, index = None, savename=None, cell_height=0.325, cell_width=0.15, dpi=100, show = False, pause = 0, fig_num = 3):
    if index is None:
        index = np.random.randint(len(text), size = num_sentence)
    else:
        num_sentence = len(index)
        index = np.array(index)

    text = np.array(text)
    text = text[index]

    # normalization = indices.shape[1]

    if isinstance(text[0], str):
        text = [s.split() for s in text]
    
    true_label = np.array(true_label)
    true_label = true_label[index]
    
    predict_label = np.array(predict_label)
    predict_label = predict_label[index]

    if label_map is not None:
        label_map = np.array(label_map)
        true_label = label_map[true_label]
        predict_label = label_map[predict_label]

    # indices batch_size, hidden size

    indices = indices[index, :] 

    plt.figure(fig_num, figsize=(19.20,10.80))
    plt.clf()

    for i in range(num_sentence):
        te = text[i]
        num_words = len(te)
        idx = np.concatenate((indices[i], np.arange(num_words)), axis = 0)
        unique_array, unique_count = np.unique(idx, return_counts = True)
        unique_count[:num_words] -= 1
        at = unique_count[1:num_words+1]/np.sum(unique_count[1:num_words+1])

        te = np.array(te)
        te = te.reshape(1, -1)
        at = at.reshape(1, -1)
        plt.subplot(num_sentence, 1, i+1)
        ax = sns.heatmap(at, annot=te, fmt='', vmin=0, vmax=1, cmap='YlOrRd', xticklabels=False, cbar=False)
        ax.set_yticklabels([str(predict_label[i]) + '/' + str(true_label[i])], rotation = 0)

    if show:
        plt.show()
        if pause > 0:
            plt.pause(pause)
    if savename is not None:   
        plt.savefig(savename)

    return index







    

    

    



