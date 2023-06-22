

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


init = nn.initorch.xavier_uniform_
uniform_init = nn.initorch.uniform

def sparse_dropout(x, keep_prob):
    msk = (torch.rand(x._values().size()) + keep_prob).floor().type(torch.bool)
    idx = x._indices()[:, msk]
    val = x._values()[msk]
    return torch.sparse.FloatTensor(idx, val, x.shape).cuda()

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.item_emb = nn.Parameter(init(torch.empty(cfg.item, cfg.hidden_size))) # cfg.item = num_real_item + 1
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(cfg.num_gcn_layers)])

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, encoder_adj):
        embeds = [self.item_emb]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds

class TrivialDecoder(nn.Module):

    def __init__(self, cfg):
        super(TrivialDecoder, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(cfg.hidden_size * 3, cfg.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        pos_emb.append(embeds[-1][pos[:,0]])
        pos_emb.append(embeds[-1][pos[:,1]])
        pos_emb.append(embeds[-1][pos[:,0]] * embeds[-1][pos[:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]])
        neg_emb.append(embeds[-1][neg[:,:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]] * embeds[-1][neg[:,:,1]])
        pos_emb = torch.cat(pos_emb, -1) # (n, hidden_size * 3)
        neg_emb = torch.cat(neg_emb, -1) # (n, num_reco_neg, hidden_size * 3)
        pos_scr = torch.exp(torch.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = torch.exp(torch.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = torch.sum(neg_scr, -1) + pos_scr
        loss = -torch.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.cfg = cfg

        self.MLP = nn.Sequential(
            nn.Linear(cfg.hidden_size * cfg.num_gcn_layers ** 2, cfg.hidden_size * cfg.num_gcn_layers, bias=True),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size * cfg.num_gcn_layers, cfg.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        for i in range(self.cfg.num_gcn_layers):
            for j in range(self.cfg.num_gcn_layers):
                pos_emb.append(embeds[i][pos[:,0]] * embeds[j][pos[:,1]])
                neg_emb.append(embeds[i][neg[:,:,0]] * embeds[j][neg[:,:,1]])
        pos_emb = torch.cat(pos_emb, -1) # (n, hidden_size * num_gcn_layers ** 2)
        neg_emb = torch.cat(neg_emb, -1) # (n, num_reco_neg, hidden_size * num_gcn_layers ** 2)
        pos_scr = torch.exp(torch.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = torch.exp(torch.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = torch.sum(neg_scr, -1) + pos_scr
        loss = -torch.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

class SASRec(nn.Module):
    def __init__(self, cfg):
        super(SASRec, self).__init__()

        self.pos_emb = nn.Parameter(init(torch.empty(cfg.maxlen, cfg.hidden_size)))
        self.layers = nn.Sequential(*[TransformerLayer() for i in range(cfg.num_trm_layers)])
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.apply(self.init_weights)

    def get_seq_emb(self, sequence, item_emb):
        seq_len = sequence.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=sequence.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(sequence)
        itm_emb = item_emb[sequence]
        pos_emb = self.pos_emb[pos_ids]
        seq_emb = itm_emb + pos_emb
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        return seq_emb

    def forward(self, input_ids, item_emb):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        seq_embs = [self.get_seq_emb(input_ids, item_emb)]
        for trm in self.layers:
            seq_embs.append(trm(seq_embs[-1], extended_attention_mask))
        seq_emb = sum(seq_embs)

        return seq_emb

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class TransformerLayer(nn.Module):

    def __init__(self):
        super(TransformerLayer, self).__init__()

        self.attention = SelfAttentionLayer()
        self.intermediate = IntermediateLayer()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output

class SelfAttentionLayer(nn.Module):

    def __init__(self, cfg):
        super(SelfAttentionLayer, self).__init__()

        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = int(cfg.hidden_size / cfg.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size)
        self.out_dropout = nn.Dropout(cfg.hidden_dropout_prob)

        self.apply(self.init_weights)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class IntermediateLayer(nn.Module):

    def __init__(self, cfg):
        super(IntermediateLayer, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4, bias=True),
            nn.GELU(),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size, bias=True),
            nn.Dropout(cfg.hidden_dropout_prob),
            nn.LayerNorm(cfg.hidden_size)
        )

    def forward(self, x):
        return self.layers(x)

class LocalGraph(nn.Module):

    def __init__(self, cfg):
        super(LocalGraph, self).__init__()

        self.cfg = cfg

    def make_noise(self, scores):
        noise = torch.rand(scores.shape).cuda()
        noise = -torch.log(-torch.log(noise))
        return scores + noise

    def forward(self, adj, embeds, foo=None):
        order = torch.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = torch.spmm(adj, embeds) - embeds
        fstNum = order

        emb = [fstEmbeds]
        num = [fstNum]

        for i in range(self.cfg.mask_depth):
            adj = sparse_dropout(adj, self.cfg.path_prob ** (i + 1))
            emb.append((torch.spmm(adj, emb[-1]) - emb[-1]) - order * emb[-1])
            num.append((torch.spmm(adj, num[-1]) - num[-1]) - order)
            order = torch.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])

        subgraphEmbeds = sum(emb) / (sum(num) + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)

        embeds = F.normalize(embeds, p=2)
        scores = torch.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.make_noise(scores)

        _, candidates = torch.topk(scores, self.cfg.num_mask_cand)

        return scores, candidates

class RandomMaskSubgraphs(nn.Module):

    def __init__(self, cfg):
        super(RandomMaskSubgraphs, self).__init__()

        self.cfg = cfg

    def normalize(self, adj):
        degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return torch.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        masked_rows = []
        masked_cols = []
        masked_idct = []

        for i in range(self.cfg.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            idct = None
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                if idct == None:
                    idct = torch.logical_or(rowIdct, colIdct)
                else:
                    idct = torch.logical_or(idct, torch.logical_or(rowIdct, colIdct))
            nxtRows = rows[idct]
            nxtCols = cols[idct]
            masked_rows.extend(nxtRows)
            masked_cols.extend(nxtCols)
            rows = rows[torch.logical_not(idct)]
            cols = cols[torch.logical_not(idct)]
            nxtSeeds = nxtRows + nxtCols
            if len(nxtSeeds) > 0 and i != self.cfg.mask_depth - 1:
                nxtSeeds = torch.unique(nxtSeeds)
                cand = torch.randperm(nxtSeeds.shape[0])
                nxtSeeds = nxtSeeds[cand[:int(nxtSeeds.shape[0] * self.cfg.path_prob ** (i + 1))]] # the dropped edges from P^k

        masked_rows = torch.unsqueeze(torch.LongTensor(masked_rows), -1)
        masked_cols = torch.unsqueeze(torch.LongTensor(masked_cols), -1)
        masked_edge = torch.hstack([masked_rows, masked_cols])
        encoder_adj = self.normalize(torch.sparse.FloatTensor(torch.stack([rows, cols], dim=0), torch.ones_like(rows).cuda(), adj.shape))

        return encoder_adj, masked_edge