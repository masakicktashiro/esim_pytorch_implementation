import os
import numpy as np
import pickle
from collections import Counter
from gensim.models import KeyedVectors
from utils import *

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, "data")
    glove_file = datapath(os.path.join(data_dir, 'glove.840B.300d.txt'))
    tmp_file = get_tmpfile(os.path.join(data_dir, "glove.840B.300d_wd2vec.bin"))
    if not os.path.exists(glove_file):
        _ = glove2word2vec(glove_file, tmp_file)
    df_train = preprocess_data(os.path.join(data_dir, "snli/snli_1.0_train.txt"))
    df_dev = preprocess_data(os.path.join(data_dir, "snli/snli_1.0_dev.txt"))
    df_test = preprocess_data(os.path.join(data_dir, "snli/snli_1.0_test.txt"))
    ct = Counter()
    for i in df_train.iterrows():
        ct.update(sum(i[1][1:].values, []))
    wd2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for i in ct.keys():
        wd2id[i] = len(wd2id)
    dig_df_train = digitalize_df(df_train, wd2id)
    dig_df_dev = digitalize_df(df_dev, wd2id)
    dig_df_test = digitalize_df(df_test, wd2id)
    pickle.dump(wd2id, open(os.path.join(data_dir,"wd2id.bin"), "wb"))
    pickle.dump(dig_df_train, open(os.path.join(data_dir, "df_train.bin"), "wb"))
    pickle.dump(dig_df_dev, open(os.path.join(data_dir, "df_dev.bin"), "wb"))
    pickle.dump(dig_df_test, open(os.path.join(data_dir, "df_test.bin"), "wb"))