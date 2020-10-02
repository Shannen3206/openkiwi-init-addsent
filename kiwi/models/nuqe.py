#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiwi import constants as const
from kiwi.data.fieldsets.quetch import build_fieldset
from kiwi.models.model import Model
from kiwi.models.quetch import QUETCH
from kiwi.models.utils import make_loss_weights
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoConfig, AutoModel
import numpy as np
import os


@Model.register_subclass
class NuQE(QUETCH):
    """Neural Quality Estimation (NuQE) model for word level quality
    estimation."""

    title = 'NuQE'

    def __init__(self, vocabs, **kwargs):

        self.source_emb = None
        self.target_emb = None
        self.linear_1 = None
        self.linear_2 = None
        self.linear_3 = None
        self.linear_4 = None
        self.linear_5 = None
        self.linear_6 = None
        self.linear_out = None
        self.embeddings_dropout = None
        self.dropout = None
        self.gru1 = None
        self.gru2 = None
        self.rand_gate = None
        self.bert_gate = None
        self.is_built = False
        super().__init__(vocabs, **kwargs)
        '''start'''
        # config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
        # self.en_model = BertModel.from_pretrained('bert-base-cased')
        self.de_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        # self.de_model = AutoModel.from_config(config)
        '''end'''

    def build(self, source_vectors=None, target_vectors=None):
        nb_classes = self.config.nb_classes
        # FIXME: Remove dependency on magic number
        weight = make_loss_weights(nb_classes, const.BAD_ID, self.config.bad_weight)

        '''start:将词表中的每个token用bert生成embedding表示，修改格式，分别保存在self.source_bert和self.target_bert中'''
        # path = os.getcwd()
        # source_bert, target_bert = [], []
        # src_bert = np.load(path + '/data/exp2/en_emb.npy', allow_pickle=True).item()
        # tgt_bert = np.load(path + '/data/exp2/de_emb.npy', allow_pickle=True).item()
        # for s in src_bert:
        #     source_bert.append(list(src_bert[s].squeeze(0)))
        # for t in tgt_bert:
        #     target_bert.append(list(tgt_bert[t].squeeze(0)))
        # self.source_bert = torch.Tensor(source_bert)
        # self.target_bert = torch.Tensor(target_bert)
        '''end'''

        self._loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.config.tags_pad_id, reduction='sum')

        # Embeddings layers:
        self._build_embeddings(source_vectors, target_vectors)

        feature_set_size = (self.config.source_embeddings_size + self.config.target_embeddings_size) * self.config.window_size

        l1_dim = self.config.hidden_sizes[0]
        l2_dim = self.config.hidden_sizes[1]
        l3_dim = self.config.hidden_sizes[2]
        l4_dim = self.config.hidden_sizes[3]

        nb_classes = self.config.nb_classes
        dropout = self.config.dropout

        # Linear layers
        self.linear_1 = nn.Linear(768, l1_dim)
        self.linear_2 = nn.Linear(l1_dim, l1_dim)
        self.linear_3 = nn.Linear(2 * l2_dim, l2_dim)
        self.linear_4 = nn.Linear(l2_dim, l2_dim)
        self.linear_5 = nn.Linear(2 * l2_dim, l3_dim)
        self.linear_6 = nn.Linear(l3_dim, l4_dim)

        # Output layer
        self.linear_out = nn.Linear(l4_dim, nb_classes)

        # Recurrent Layers
        self.gru_1 = nn.GRU(l1_dim, l2_dim, bidirectional=True, batch_first=True)
        self.gru_2 = nn.GRU(l2_dim, l2_dim, bidirectional=True, batch_first=True)

        # Dropout after linear layers
        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        # Explicit initializations
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.xavier_uniform_(self.linear_5.weight)
        nn.init.xavier_uniform_(self.linear_6.weight)
        # nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_1.bias, 0.0)
        nn.init.constant_(self.linear_2.bias, 0.0)
        nn.init.constant_(self.linear_3.bias, 0.0)
        nn.init.constant_(self.linear_4.bias, 0.0)
        nn.init.constant_(self.linear_5.bias, 0.0)
        nn.init.constant_(self.linear_6.bias, 0.0)
        # nn.init.constant_(self.linear_out.bias, 0.0)

        self.is_built = True

    @staticmethod
    def fieldset(*args, **kwargs):
        return build_fieldset(*args, **kwargs)

    @staticmethod
    def from_options(vocabs, opts):
        model = NuQE(
            vocabs=vocabs,
            predict_target=opts.predict_target,
            predict_gaps=opts.predict_gaps,
            predict_source=opts.predict_source,
            source_embeddings_size=opts.source_embeddings_size,
            target_embeddings_size=opts.target_embeddings_size,
            hidden_sizes=opts.hidden_sizes,
            bad_weight=opts.bad_weight,
            window_size=opts.window_size,
            max_aligned=opts.max_aligned,
            dropout=opts.dropout,
            embeddings_dropout=opts.embeddings_dropout,
            freeze_embeddings=opts.freeze_embeddings,
        )
        return model

    def forward(self, batch):
        assert self.is_built
        device = getattr(batch, 'source')[0].device

        if self.config.predict_source:
            align_side = const.SOURCE_TAGS
        else:
            align_side = const.TARGET_TAGS

        '''start:bert计算词向量'''
        # 未经过处理的语料的index表示
        src_input, _ = getattr(batch, const.SOURCE)     # [b, s_t]
        tgt_input, _ = getattr(batch, const.TARGET)     # [b, t_t]

        # 计算bert词向量并保存
        # src_emb = self.save_bert_embed(src_input, 'src')
        # tgt_emb = self.save_bert_embed(tgt_input, 'tgt')
        '''end'''

        # soft align
        # src_emb = self.en_model(source_input)[0]    # [b, s_t, 768]
        h_target = self.de_model(tgt_input)[0]    # [b, t_t, 768]
        # alpha = F.softmax(tgt_emb.bmm(src_emb.transpose(-1, -2)), dim=-1)   # [b, t_t, s_t]
        # alpha_src_emb = alpha.bmm(src_emb)          # [b, t_t, 768]
        # h = tgt_emb + alpha_src_emb

        # target_input[b, t_t] -> [b, t_t, w]       source_input[b, s_t] -> [b, t_t, a, w]
        # target_input, source_input, nb_alignments = self.make_input(batch, align_side)

        # h_source = self.source_emb(source_input)      # [b, t, a, w, d]
        # h_source = self.load_bert_embed(source_input, src_emb, 'src')   # [b, t, a, w, 768]
        # h_source = self.embeddings_dropout(h_source)
        # if len(h_source.shape) == 5:
        #     h_source = h_source.sum(2, keepdim=False) / nb_alignments.unsqueeze(-1).unsqueeze(-1)  # [b, t, w, d]
        # h_source = h_source.view(source_input.size(0), source_input.size(1), -1)  # [b, t, w*d]

        # h_target = self.target_emb(target_input)
        # h_target = self.load_bert_embed(target_input, tgt_emb, 'tgt')   # [b, t, w, 768]
        # h_target = self.embeddings_dropout(h_target)
        # if len(h_target.shape) == 5:
        #     h_target = h_target.sum(2, keepdim=False) / nb_alignments.unsqueeze(-1).unsqueeze(-1)
        # h_target = h_target.view(target_input.size(0), target_input.size(1), -1)    # [b, t, w*d]

        '''start:src_bert'''
        # h_source_bert = self.source_bert_emb(source_input)  # [b, t, a, w, 1024]
        # # h_source_bert = self.trans1(h_source_bert)  # 1024 -> 200
        # h_source_bert = self.embeddings_dropout(h_source_bert)
        # if len(h_source_bert.shape) == 5:
        #     h_source_bert = h_source_bert.sum(2, keepdim=False) / nb_alignments.unsqueeze(-1).unsqueeze(-1)  # [b, t, w, 200]
        # h_source_bert = h_source_bert.view(source_input.size(0), source_input.size(1), -1)  # [b, t, 600]
        '''end'''

        '''start:tgt_bert'''
        # h_target_bert = self.target_bert_emb(target_input)  # [b, t, w, 768]
        # # h_target_bert = self.trans2(h_target_bert)  # 768 -> 200
        # h_target_bert = self.embeddings_dropout(h_target_bert)
        # if len(h_target_bert.shape) == 5:
        #     h_target_bert = h_target_bert.sum(2, keepdim=False) / nb_alignments.unsqueeze(-1).unsqueeze(-1)  # [b, t, w, 200]
        # h_target_bert = h_target_bert.view(target_input.size(0), target_input.size(1), -1)  # [b, t, 600]
        '''end'''

        '''start:[b, t, d] -> [b, d]'''
        # src_sent = h_source_bert.mean(1)
        # src_sent = h_source_bert.sum(1, keepdim=False)
        # src_sent = torch.max(h_source_bert, 1)[0]
        # tgt_sent = h_target_bert.mean(1)
        # tgt_sent = h_target_bert.sum(1, keepdim=False)
        # tgt_sent = torch.max(h_target_bert, 1)[0]
        '''end'''

        '''start:加权处理'''
        # source_input, source_lengths = getattr(batch, const.SOURCE)
        # target_input, target_lengths = getattr(batch, const.TARGET)
        # mask1 = torch.ones_like(source_input, dtype=torch.uint8)
        # mask1 &= torch.as_tensor(source_input != 1, device=device, dtype=torch.uint8)
        # mask1 = F.pad(mask1, pad=(0, target_input.size(1)-source_input.size(1)), value=0)
        # mask2 = torch.ones_like(target_input, dtype=torch.uint8)
        # mask2 &= torch.as_tensor(target_input != 1, device=device, dtype=torch.uint8)
        #
        # src_weight = self.transto1(h_source_bert).squeeze(-1)  # [b, t, 1]
        # src_weight = (src_weight * mask1.type(torch.float))
        # src_weight = F.log_softmax(src_weight, dim=-1).unsqueee(-1)
        #
        # tgt_weight = self.transto2(h_target_bert).squeeze(-1)  # [b, t, 1]
        # tgt_weight = (tgt_weight * mask2.type(torch.float))
        # tgt_weight = F.log_softmax(tgt_weight, dim=-1).unsqueeze(-1)
        #
        # h_source_w = h_source_bert * src_weight
        # h_target_w = h_target_bert * tgt_weight
        # src_sent = h_source_w.mean(1)
        # tgt_sent = h_target_w.mean(1)
        '''end'''

        '''start:将句子级表示[b, d]融入token级表示中[b, t, d]中'''
        # temp_src = []
        # for i in range(getattr(batch, "batch_size")):
        #     h_src = h_source_bert[i]  # [t, d1]
        #     s_sent = src_sent[i]  # [d2]
        #     temp_s = []
        #     for h_s in h_src:
        #         # s = torch.cat((h_s.detach().cpu(), s_sent.detach().cpu()), dim=-1).tolist()
        #         s = (h_s.detach().cpu()*0.9 + s_sent.detach().cpu()*0.1).tolist()
        #         # tmp_s = torch.cat((h_s, s_sent), dim=-1)
        #         # f_s = F.sigmoid(self.addsent_s(tmp_s))
        #         # s = (h_s * f_s + s_sent * (1 - f_s)).detach().cpu().tolist()
        #         temp_s.append(s)
        #     temp_src.append(temp_s)
        # temp_tgt = []
        # for i in range(getattr(batch, "batch_size")):
        #     h_tgt = h_target_bert[i]  # [t, d1]
        #     t_sent = tgt_sent[i]  # [d2]
        #     temp_t = []
        #     for h_t in h_tgt:
        #         # t = torch.cat((h_t.detach().cpu(), t_sent.detach().cpu()), dim=-1).tolist()
        #         t = (h_t.detach().cpu()*0.9 + t_sent.detach().cpu()*0.1).tolist()
        #         # tmp_t = torch.cat((h_t, t_sent), dim=-1)
        #         # f_t = F.sigmoid(self.addsent_t(tmp_t))
        #         # t = (h_t * f_t + t_sent * (1 - f_t)).detach().cpu().tolist()
        #         temp_t.append(t)
        #     temp_tgt.append(temp_t)
        # temp_src = torch.Tensor(temp_src).to(device)
        # temp_tgt = torch.Tensor(temp_tgt).to(device)
        '''end'''

        '''start:gate'''
        # rand = torch.cat((h_source, h_target), dim=-1)
        # bert = torch.cat((h_source_bert, h_target_bert), dim=-1)
        # vector = torch.cat((rand, bert), dim=-1)
        # fr = F.sigmoid(self.rand_gate(vector))
        # # fb = F.sigmoid(self.bert_gate(vector))
        # fb = 1-fr
        '''end'''

        # feature_set = (h_source_bert, h_target_bert)
        # aaa = h_source + h_target
        # bbb = h_source_bert + h_target_bert
        # feature_set = (aaa, bbb, torch.abs(aaa-bbb), aaa*bbb)
        # feature_set = (temp_src, temp_tgt)
        # # h = rand * fr + bert * fb
        # feature_set = (h_source, h_target)
        # h = torch.cat(feature_set, dim=-1)
        h = h_target
        h = self.dropout_in(h)
        h = F.relu(self.linear_1(h))
        h = F.relu(self.linear_2(h))
        h, _ = self.gru_1(h)
        h = F.relu(self.linear_3(h))
        h = F.relu(self.linear_4(h))
        h, _ = self.gru_2(h)
        h = F.relu(self.linear_5(h))
        h = F.relu(self.linear_6(h))
        h = self.dropout_out(h)
        h = self.linear_out(h)
        # h = F.log_softmax(h, dim=-1)
        outputs = OrderedDict()

        if self.config.predict_target:
            outputs[const.TARGET_TAGS] = h
        if self.config.predict_gaps:
            outputs[const.GAP_TAGS] = h
        if self.config.predict_source:
            outputs[const.SOURCE_TAGS] = h

        return outputs

    def save_bert_embed(self, input, dir):
        input = torch.cat((input, input.new_full((input.shape[0],), 1).unsqueeze(-1)), dim=-1)  # 1 <pad>
        input = torch.cat((input, input.new_full((input.shape[0],), 4).unsqueeze(-1)), dim=-1)  # 4 <unaligned>

        if dir == 'src':
            embed = self.en_model(input)[0]
        else:
            embed = self.de_model(input)[0]

        data_all = []
        for i, sent in enumerate(input):
            data = {}
            for j, word in enumerate(sent):
                data[word.tolist()] = embed[i][j]
            data_all.append(data)
        return data_all

    def load_bert_embed(self, input, data, dir):
        if dir == 'src':
            result = input.new_full((input.shape[0], input.shape[1], input.shape[2], input.shape[3], 768), 0.0, dtype=torch.float32)
            for i, sent in enumerate(input):
                for j, word in enumerate(sent):
                    for k, aligned in enumerate(word):
                        for l, win in enumerate(aligned):
                            selected = data[i][win.tolist()]
                            result[i][j][k][l] = selected
        else:
            result = input.new_full((input.shape[0], input.shape[1], input.shape[2], 768), 0.0, dtype=torch.float32)
            for i, sent in enumerate(input):
                for j, word in enumerate(sent):
                    for k, aligned in enumerate(word):
                        selected = data[i][aligned.tolist()]
                        result[i][j][k] = selected
        return result
