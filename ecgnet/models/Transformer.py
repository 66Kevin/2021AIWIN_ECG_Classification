import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数番目
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数番目
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights


class ECGNet1(nn.Module):
    def __init__(self, in_channel=12, ninp=512, nhead=4, nhid=1024, dropout=0.1, nlayers=2):
        super(ECGNet1, self).__init__()

        d_model = 256
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer = TransformerEncoder(encoder_layers, nlayers)


        self.top_layer1 = nn.Sequential(
                             nn.BatchNorm1d(12),
                             nn.ReLU(inplace=True),
                             nn.Conv1d(in_channel, 32, kernel_size=50, stride=2),
                         )

        self.top_layer2 = nn.Sequential(
                             nn.BatchNorm1d(32),
                             nn.ReLU(inplace=True),
                             nn.Conv1d(32, 64, kernel_size=15, stride=2),
                         )

        self.top_layer3 = nn.Sequential(
                             nn.BatchNorm1d(64),
                             nn.ReLU(inplace=True),
                             nn.Conv1d(64, 128, kernel_size=15, stride=2),
                         )

        self.top_layer4 = nn.Sequential(
                             nn.BatchNorm1d(128),
                             nn.ReLU(inplace=True),
                             nn.Conv1d(128, 256, kernel_size=15, stride=2),
        )


        self.bottom_linear = nn.Sequential(
                                 nn.Linear(d_model+1, d_model//2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model//2, 2)
                             )

        self.pos_encoder = PositionalEncoding(d_model)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=1,
                                    batch_first=False, bidirectional=True)



    def forward(self, x, info):
        x = x.squeeze(1) # torch.Size([1, 12, 5000])
        x = self.top_layer1(x) # torch.Size([1, 32, 4988])  -> [1, 32, 2476] ks=50, s=2
        x = self.top_layer2(x) # torch.Size([1, 64, 4982])  -> [1, 64, 1231] ks=15, s=2
        x = self.top_layer3(x) # torch.Size([1, 128, 4978]) -> [1, 128, 609] ks=15, s=2
        x = self.top_layer4(x) # torch.Size([1, 256, 4976]) -> [1, 256, 298] ks=15, s=2
        x = x.permute(2, 0, 1) # torch.Size([4976, 1, 256]) -> [298, 1, 256]
        print(x.shape)

        # x = self.pos_encoder(x) # torch.Size([4976, 1, 256])      -> [298, 1, 256]
        x, _ = self.lstm(x)
        x_t, _ = self.transformer(x) # torch.Size([4976, 1, 256]) -> [298, 1, 256]

        x = x_t.permute(1, 2, 0) # torch.Size([1, 256, 4976])                   -> [1, 256, 298]
        x = F.max_pool1d(x, kernel_size=x.size()[2:]) # torch.Size([1, 256, 1]) -> [1, 256, 1]
        x = x.contiguous().view(x.size()[0], -1) # torch.Size([1, 256])
        if info != None:
            info = torch.unsqueeze(info, 1)
            x = torch.cat([x, info], dim=1)
        x = self.bottom_linear(x)
        return x


# if __name__ == '__main__':
#     input = torch.randn(1, 12, 5000)
#     info = torch.randn(1, 1)
#     ECGNet1 = Model_TFEncoder_AttnMap_V2()
#     output = ECGNet1(input, info)
#     print(output)
