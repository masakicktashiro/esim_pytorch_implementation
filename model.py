import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD = 0
UNK = 1
BOS = 2
EOS = 3

class ESIM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, dropout=0.1, initial_weight=None,
                 num_layers=1):
        super(ESIM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        if initial_weight is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(initial_weight, dtype=torch.float, device=device))
            # self.embedding.weight.requires_grad = False
        self.lstm_1 = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True,
                              num_layers=num_layers)
        for name, param in self.lstm_1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.lstm_2 = nn.LSTM(hidden_size, output_size, batch_first=True, bidirectional=True,
                              num_layers=num_layers)
        for name, param in self.lstm_2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.linear_1 = nn.Linear(hidden_size * 8, hidden_size)
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0.0)
        self.linear_3 = nn.Linear(output_size * 8, output_size)
        nn.init.xavier_normal_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0.0)
        self.linear_4 = nn.Linear(output_size, 3)
        nn.init.xavier_normal_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0.0)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, a, b, mask_a=None, mask_b=None):
        """
        Parameters
        ==========
            a : torch.FloatTensor
                precise, (batch_size, length_a)
            b : torch.FloatTensor
                hypothesis, (batch_size, num_candidate(5 in this case), length_b)
        Return
        ==========
            output : torch.FloatTensor
                output
        """
        batch_size, length_a = a.shape
        batch_size, length_b = b.shape
        assert a.shape[0] == b.shape[0]
        if mask_a is None:
            mask_a = torch.zeros_like(a)  # (batch_size, length_a)
        if mask_b is None:
            mask_b = torch.zeros((batch_size, length_b))
        mask_b = mask_b.reshape(batch_size, length_b)  # (batch_size, length_b)
        mask_a = mask_a.byte()
        mask_b = mask_b.byte()
        a = self.embedding(a)
        b = self.embedding(b)  # (batch_size, length, embedding_size)
        a = self.dropout(a)
        b = self.dropout(b)
        a, _ = self.lstm_1(a)
        b, _ = self.lstm_1(b)  # (batch_size, length, hidden_size)
        a = a.masked_fill(mask_a.unsqueeze(-1).repeat(1, 1, self.hidden_size * 2), 0)
        b = b.masked_fill(mask_b.unsqueeze(-1).repeat(1, 1, self.hidden_size * 2), 0)
        m_a, m_b = self._local_inference(a, b, mask_a, mask_b)  # (batch_size, length, hidden_size*4)
        m_a = self.dropout(self.relu(self.linear_1(m_a)))
        m_b = self.dropout(self.relu(self.linear_1(m_b)))
        v_a, _ = self.lstm_2(m_a)
        v_b, _ = self.lstm_2(m_b)  # (batch_size, length, output_size*2)
        v_a = v_a.masked_fill(mask_a.unsqueeze(-1).repeat(1, 1, self.output_size * 2), 0)
        v_b = v_b.masked_fill(mask_b.unsqueeze(-1).repeat(1, 1, self.output_size * 2), 0)
        v_a = v_a.transpose(2, 1)
        v_b = v_b.transpose(2, 1)  # (batch_size, output_size, length)
        mask_a = torch.where(mask_a == 0, torch.ones_like(mask_a), torch.zeros_like(mask_a))  # reverse for next calc
        mask_b = torch.where(mask_b == 0, torch.ones_like(mask_b), torch.zeros_like(mask_b))
        avg_a = v_a.sum(dim=-1) / mask_a.float().sum(dim=-1, keepdim=True)
        avg_b = v_b.sum(dim=-1) / mask_b.float().sum(dim=-1, keepdim=True)
        self.avg_a = avg_a
        self.avg_b = avg_b
        self.v_a = v_a
        self.v_b = v_b
        max_a = F.max_pool1d(v_a, kernel_size=length_a).squeeze(-1)
        max_b = F.max_pool1d(v_b, kernel_size=length_b).squeeze(-1)
        v = torch.cat([avg_a, max_a, avg_b, max_b], dim=-1)  # (batch_size, output_size * 4)
        v = self.dropout(v)
        output = self.dropout(self.tanh(self.linear_3(v)))  # (batch_size, output_size)
        output = self.linear_4(output)
        output = F.log_softmax(output)
        return output

    def _local_inference(self, a, b, mask_a, mask_b):
        """
        Parameters
        ==========
            a : torch.FloatTensor
                precise, (batch_size, length_a, hidden_size)
            b : torch.FloatTensor
                hypothesis, (batch_size, length_b, hidden_size)
            mask_a : torch.FloatTensor
                precise, (batch_size, length_a, hidden_size)
            mask_b : torch.FloatTensor
                hypothesis, (batch_size, length_b, hidden_size)
        Return
        ==========
            m_a : torch.FloatTensor
                precise, (batch_size, length_a, hidden_size * 4)
            m_b : torch.FloatTensor
                hypothesis, (batch_size, length_b, hidden_size * 4)
        """
        _, len_a, hidden_size = a.shape
        _, len_b, hidden_size = b.shape
        b = b.transpose(2, 1)  # (batch_size, hidden_size, length_b)
        mask_b = mask_b.unsqueeze(-1).transpose(2, 1)
        mask_a = mask_a.unsqueeze(-1)
        r_mask_b = torch.where(mask_b == 0, torch.ones_like(mask_b), torch.zeros_like(mask_b)).float()  # reverse
        r_mask_a = torch.where(mask_a == 0, torch.ones_like(mask_a), torch.zeros_like(mask_a)).float()
        mask = torch.bmm(r_mask_a, r_mask_b)
        mask = torch.where(mask == 0, torch.ones_like(mask), torch.zeros_like(mask)).byte()
        e = torch.bmm(a, b)  # (batch_size, len_a, len_b)
        e = e.masked_fill(mask, -1e10)
        e_a = F.softmax(e, dim=-1)
        e_b = F.softmax(e, dim=1)
        _a = torch.cat([(e_a[:, [i], :] * b).sum(dim=-1).unsqueeze(dim=1) for i in range(len_a)], dim=1)
        _b = torch.cat([(e_b[:, :, [i]] * a).sum(dim=1).unsqueeze(dim=-1) for i in range(len_b)], dim=-1)
        assert _b.shape == b.shape
        _b = _b.transpose(2, 1)
        b = b.transpose(2, 1)
        _a = _a.masked_fill(mask_a, 0)
        _b = _b.masked_fill(mask_b.transpose(2, 1), 0)
        m_a = torch.cat([a, _a, a - _a, a * _a], dim=-1)
        m_b = torch.cat([b, _b, b - _b, b * _b], dim=-1)
        return m_a, m_b