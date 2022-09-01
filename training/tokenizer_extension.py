# NLP | How to add a domain-specific vocabulary (new tokens) to a subword tokenizer already trained like BERT WordPiece
# https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41

# BioBERT uses modified embeddings, without modifying the tokenizer
# these methods are not necessarily the most efficient (high number of subword tokens per tokenized text vs high training costs in terms of training data size and computation time).

# num_added_toks = tokenizer.add_tokens(new_tokens)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pickle
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# print('before: {}'.format(tokenizer))
# print('len before: {}'.format(len(tokenizer)))
# #
# #
# new_toks = ['hallo', 'einschrankung', 'topluluk', 'macera']
# # Add new tokenizers
# num_added_toks = tokenizer.add_tokens(new_toks)
# print('We have added', num_added_toks, 'tokens')
#  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
# # model.resize_token_embeddings(len(tokenizer))
# # print('after: {}'.format(tokenizer))
# print('len after: {}'.format(len(tokenizer)))

def new_tokens():
    df = pd.read_csv('/home/kaan/scrape_data/germanwiki_vocab.csv')
    df['no_punc'] = df['0'].str.replace('[^\w\s]', '', regex=True)
    df.dropna(subset=['no_punc'], inplace=True)
    unique_list = df['no_punc'].unique()
    return unique_list

def save_new_tokens_to_add(base_tokenizer, new_toks):
    new_toks_to_add = [new_word for new_word in tqdm(new_toks) if new_word not in base_tokenizer.vocab.keys()]
    file_name = "new_vocab_{}.txt".format(base_tokenizer.name_or_path)

    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(new_toks_to_add, fp)

def load_new_tokens_to_add(path='bert-base-uncased'):
    with open("new_vocab_{}.txt".format(path), "rb") as fp:  # Unpickling
        b = pickle.load(fp)
        print("Size of new token: {}".format(len(b)))
        print("Some samples: {}".format(b[0:10]))
        return b


# toks = new_tokens()
# save_new_tokens_to_add(tokenizer, toks)
# load_new_tokens_to_add()

def extend_tokenizer_and_model(base_tokenizer, base_model):
    # new_toks_to_add = [new_word for new_word in tqdm(new_toks) if new_word not in base_tokenizer.vocab.keys()]
    # print('{} new domain-specific words will be added to the tokenizer''s vocabulary.'.format(len(new_toks_to_add)))

    # Note: .add_tokens returns an integer - therefore we can't create a new variable
    # like new_tokenizer = tokenizer.add_tokens(...)

    new_toks_to_add = load_new_tokens_to_add()
    num_added_toks = 0

    addition = [
        new_toks_to_add[0:20300]
        # new_toks_to_add[0:2499], new_toks_to_add[2500:4999],
        # new_toks_to_add[5000:7499], new_toks_to_add[7500:9999],
        # new_toks_to_add[10000:12499], new_toks_to_add[12500:14999],
        # new_toks_to_add[15000:17499], new_toks_to_add[17500:19999],
        # new_toks_to_add[20000:22499], new_toks_to_add[22500:24998],
        # new_toks_to_add[25000:27499], new_toks_to_add[27500:29999],
        # new_toks_to_add[30000:32499], new_toks_to_add[32500:34999],
        # new_toks_to_add[35000:37499], new_toks_to_add[37500:39999],
        # new_toks_to_add[40000:42499], new_toks_to_add[42500:44999],
        # new_toks_to_add[45000:47499], new_toks_to_add[47500:49999],
        # new_toks_to_add[50000:52499], new_toks_to_add[52500:54999],
        # new_toks_to_add[55000:57499], new_toks_to_add[57500:59999]
    ]

    for batch in addition:
        num_added_toks = num_added_toks + base_tokenizer.add_tokens(batch)



    # num_added_toks = num_added_toks + base_tokenizer.add_tokens(new_toks_to_add)
    base_model.resize_token_embeddings(len(tokenizer))
    # print('Len tokenizer before: {}'.format(len(base_tokenizer)))
    # print('Len tokenizer after: {}'.format(len(extended_tokenizer)))
    return base_tokenizer, base_model, num_added_toks

# new_token_list = new_tokens()
new_tokenizer, new_model, num_added_tokens = extend_tokenizer_and_model(tokenizer, model)
print('new tokenizer: ')
print(new_tokenizer)
print('len of new tokenizer: ')
print(len(new_tokenizer))
print('new model: ')
print(new_model.bert.embeddings.word_embeddings)
# # print('New tokens - Five examples: {}'.format(new_token_list[0:5]))
# # print('Length of new_token_list: {}'.format(len(new_token_list)))
new_tokenizer.save_pretrained(f"./extended_tokenizer_library/{new_tokenizer.name_or_path}_{num_added_tokens}_new_tokens/")
new_model.save_pretrained(f"./extended_tokenizer_library/{new_tokenizer.name_or_path}_{num_added_tokens}_model/")


# ------ NOTES -------
# - There's a limit for # of words (25000) that can be added to the tokenizer at once
#   https://github.com/huggingface/tokenizers/issues/175

# - Continued pretraining
#   https://discuss.huggingface.co/t/continue-pre-training-of-greek-bert-with-domain-specific-dataset/4005/6
#   https://discuss.huggingface.co/t/continue-pre-training-greek-bert-with-domain-specific-dataset/4339