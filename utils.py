from collections import Counter
import os
import pickle
import numpy as np
import torch
import pandas as pd
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD = 0
UNK = 1
BOS = 2
EOS = 3
random_state = 0

def preprocess_data(path):
    df = pd.read_csv(path, sep="\t")
    df = df.loc[:, ["gold_label", "sentence1_binary_parse", "sentence2_binary_parse"]]
    df = df[~df.sentence2_binary_parse.isnull()]
    df.sentence1_binary_parse = df.sentence1_binary_parse.apply(lambda x: [i for i in x.split() if i not in ["(", ")"]])
    df.sentence2_binary_parse = df.sentence2_binary_parse.apply(lambda x: [i for i in x.split() if i not in ["(", ")"]])
    df = df[df.gold_label != "-"]
    label2digit = {j: i for i, j in enumerate(np.unique(df.gold_label.values))}
    df.columns = ["gold_label", "sentence1", "sentence2"]
    df.gold_label = df.gold_label.map(label2digit)
    return df

def digitalize_df(df, wd2id):
    dig_df = df.copy()
    dig_df.sentence1 = dig_df.sentence1.apply(lambda x:[wd2id.get(i) if wd2id.get(i) is not None else UNK for i in x])
    dig_df.sentence2 = dig_df.sentence2.apply(lambda x:[wd2id.get(i) if wd2id.get(i) is not None else UNK for i in x])
    return dig_df

class DataLoader(object):

    def __init__(self, sen1, sen2, Y, batch_size, shuffle=False):
        """
        :param sen1: list, 入力言語の文章（単語IDのリスト）のリスト
        :param Y: list, 出力言語の文章（単語IDのリスト）のリスト
        :param batch_size: int, バッチサイズ
        :param shuffle: bool, サンプルの順番をシャッフルするか否か
        """
        self.data = list(zip(sen1, sen2, Y))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0

        self.reset()

    def reset(self):
        if self.shuffle:  # サンプルの順番をシャッフルする
            self.data = np.random.shuffle(self.data, random_state=random_state)
        self.start_index = 0  # ポインタの位置を初期化する

    def __iter__(self):
        return self

    def __next__(self):
        # ポインタが最後まで到達したら初期化する
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        # バッチを取得
        sen1, sen2, Y = zip(*self.data[self.start_index:self.start_index +self.batch_size])
        # 短い系列の末尾をパディングする
        lengths_sen1 = [len(s) for s in sen1]
        max_length_sen1 = np.max(lengths_sen1)
        padded_sen1 = [self.pad_seq(s, max_length_sen1) for s in sen1]
        lengths_sen2 = [len(s) for s in sen2]
        max_length_sen2 = np.max(lengths_sen2)
        padded_sen2 = [self.pad_seq(s, max_length_sen2) for s in sen2]
        # tensorに変換
        batch_sen1 = torch.tensor(padded_sen1, dtype=torch.long, device=device)
        batch_sen2 = torch.tensor(padded_sen2, dtype=torch.long, device=device)
        batch_Y = torch.tensor(Y, dtype=torch.long, device=device)
        batch_size = batch_sen1.shape[0]
        mask_sen1 = torch.where(batch_sen1==PAD, torch.ones_like(batch_sen1), torch.zeros_like(batch_sen1))
        mask_sen2 = torch.where(batch_sen2==PAD, torch.ones_like(batch_sen2), torch.zeros_like(batch_sen2))

        # ポインタを更新する
        self.start_index += self.batch_size

        return batch_sen1, batch_sen2, batch_Y, mask_sen1, mask_sen2

    @staticmethod
    def pad_seq(seq, max_length):
        """
        系列の末尾をパディングする
        :param seq: list of int, 単語のインデックスのリスト
        :param max_length: int, バッチ内の系列の最大長
        :return seq: list of int, 単語のインデックスのリスト
        """
        seq = [BOS] + seq + [EOS]
        seq += [PAD for i in range(max_length + 2 - len(seq))]
        return seq
