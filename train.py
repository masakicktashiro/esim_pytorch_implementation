import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from gensim.models import KeyedVectors
from utils import DataLoader
from model import ESIM

if __name__ == "__main__":
    np.random.seed(0)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, "data")
    wd2id = pickle.load(open(os.path.join(data_dir, "wd2id.bin"), "rb"))
    id2wd = {j : i for i, j in wd2id.items()}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    df_train = pickle.load(open(os.path.join(data_dir, "df_train.bin"), "rb"))
    df_dev = pickle.load(open(os.path.join(data_dir, "df_dev.bin"), "rb"))
    train_premices = df_train.sentence1.values
    train_hypos = df_train.sentence2.values
    train_labels = df_train.gold_label.values
    dev_premices = df_dev.sentence1.values
    dev_hypos = df_dev.sentence2.values
    dev_labels = df_dev.gold_label.values
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    num_epoch = 15
    vocab_size = len(wd2id)
    embedding_size = 300
    hidden_size = 300
    output_size = 300
    dropout = 0.5
    batch_size = 32
    clip = 10
    lr = 0.0004

    weight = np.random.randn(vocab_size, embedding_size)
    glove = KeyedVectors.load_word2vec_format(os.path.join(data_dir, "glove.6B.300d_wd2vec.bin"))
    for ind in range(2, vocab_size):
        if glove.vocab.get(id2wd[ind]) is not None:
            weight[ind] = glove.wv.word_vec(id2wd.get(ind))


    train_data_loader = DataLoader(train_premices, train_hypos, train_labels, batch_size=batch_size)
    dev_data_loader = DataLoader(dev_premices, dev_hypos, dev_labels, batch_size=batch_size)
    model = ESIM(vocab_size, embedding_size, hidden_size, output_size, initial_weight=weight, dropout=dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)

    for epoch in range(num_epoch):
        score_ls = []
        loss_ls = []
        cnt = 0
        for sen1, sen2, lab, mask1, mask2 in tqdm(train_data_loader):
            model.train()
            output = model(sen1, sen2, mask1, mask2)
            loss = F.nll_loss(output, lab)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            optimizer.step()
            pred = np.argmax(output.cpu().data, axis=-1)
            score = accuracy_score(pred, lab.cpu().data)
            score_ls.append(score)
            loss_ls.append(loss.item())
            cnt += 1
        print("train")
        print(np.mean(score_ls), epoch)
        print(np.mean(loss_ls), epoch)
        score_ls = []
        loss_ls = []
        for sen1, sen2, lab, mask1, mask2 in tqdm(dev_data_loader):
            model.eval()
            with torch.no_grad():
                output = model(sen1, sen2, mask1, mask2)
                loss = F.nll_loss(output, lab)
                pred = np.argmax(output.cpu().data, axis=-1)
                score = accuracy_score(pred, lab.cpu().data)
            score_ls.append(score)
            loss_ls.append(loss.item())
        print("test")
        print(np.mean(score_ls), epoch)
        print(np.mean(loss_ls), epoch)
        torch.save(model.state_dict(), "epoch%d.model"%epoch)