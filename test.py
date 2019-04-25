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
    id2wd = {j: i for i, j in wd2id.items()}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    df_test = pickle.load(open(os.path.join(data_dir, "df_test.bin"), "rb"))
    test_premices = df_test.sentence1.values
    test_hypos = df_test.sentence2.values
    test_labels = df_test.gold_label.values
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

    test_data_loader = DataLoader(test_premices, test_hypos, test_labels, batch_size=32)
    model = ESIM(vocab_size, embedding_size, hidden_size, output_size)
    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(data_dir, "epoch_13.model"), map_location='cpu'))
    score_ls = []
    loss_ls = []
    for sen1, sen2, lab, mask1, mask2 in tqdm(test_data_loader):
        model.eval()
        with torch.no_grad():
            output = model(sen1, sen2, mask1, mask2)
            loss = F.cross_entropy(output, lab)
            pred = np.argmax(output.data, axis=-1)
            score = accuracy_score(pred, lab.data)
        score_ls.append(score)
        loss_ls.append(loss.item())
    print("test")
    print(np.mean(score_ls))
    print(np.mean(loss_ls))