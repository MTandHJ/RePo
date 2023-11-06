

import torch
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, ITEM, ID

from copy import deepcopy



class UnifiedBackbone(freerec.models.RecSysArch):

    def predict(
        self, 
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ):
        r"""
        users: torch.Tensor
            - userIDs: (B, 1)
            - seqs: (B, S)
        """
        features = self.forward(users) # (B, *, D)
        posEmbds = self.Item.look_up(positives) # (B, *, D)
        negEmbds = self.Item.look_up(negatives) # (B, *, D)
        return features.mul(posEmbds).sum(-1).flatten(), features.mul(negEmbds).sum(-1).flatten()

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        features = self.forward(seqs)[:, [-1], :]  # (B, 1, D)
        others = self.Item.look_up(pool) # (B, K, D)
        return features.mul(others).sum(-1)

    def recommend_from_full(self, seqs: torch.Tensor):
        features = self.forward(seqs)[:, -1, :]  # (B, D)
        items = self.Item.embeddings.weight[self.NUM_PADS:] # (N, D)
        return features.matmul(items.t()) # (B, N)


class BPRMF(UnifiedBackbone):

    def __init__(
        self, 
        fields: FieldModuleList, 
        embedding_dim: int, 
        cfg
    ) -> None:
        super().__init__()

        self.fields = deepcopy(fields)
        self.fields.embed(
            embedding_dim, ID
        )
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, users: torch.Tensor):
        userEmbs = self.User.look_up(users) # B x 1 x D
        return userEmbs


class GRU4Rec(freerec.models.RecSysArch):

    def __init__(
        self, 
        fields: FieldModuleList, 
        embedding_dim: int,
        cfg,
    ) -> None:
        super().__init__()

        hidden_size = cfg.hidden_size
        num_gru_layers = cfg.num_gru_layers
        dropout_rate = cfg.dropout_rate

        self.NUM_PADS = cfg.NUM_PADS

        self.fields = deepcopy(fields)
        self.fields.embed(
            embedding_dim, ITEM, ID, padding_idx=0
        )
        self.Item = self.fields[ITEM, ID]

        self.emb_dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_gru_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0)
                nn.init.xavier_uniform_(m.weight_ih_l0)

    def forward(self, seqs: torch.Tensor):
        masks = seqs.not_equal(0).unsqueeze(-1) # (B, S, 1)
        seqs = self.Item.look_up(seqs) # (B, S, D)
        seqs = self.emb_dropout(seqs)
        gru_out, _ = self.gru(seqs) # (B, S, H)

        gru_out = self.dense(gru_out) # (B, S, D)
        features = gru_out.gather(
            dim=1,
            index=masks.sum(1, keepdim=True).add(-1).expand((-1, 1, gru_out.size(-1)))
        ) # (B, 1, D)

        return features


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: int):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D)
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))) # -> (B, D, S)
        outputs = outputs.transpose(-1, -2) # -> (B, S, D)
        outputs += inputs
        return outputs


class SASRec(freerec.models.RecSysArch):

    def __init__(
        self, 
        fields: FieldModuleList, 
        embedding_dim: int,
        cfg
    ) -> None:
        super().__init__()

        maxlen = cfg.maxlen
        num_heads = cfg.num_heads
        num_blocks = cfg.num_blocks
        dropout_rate = cfg.dropout_rate

        self.NUM_PADS = cfg.NUM_PADS

        self.num_blocks = num_blocks
        self.fields = deepcopy(fields)
        self.fields.embed(
            embedding_dim, padding_idx=0
        )
        self.Item = self.fields[ITEM, ID]

        self.Position = nn.Embedding(maxlen, embedding_dim)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.attnLNs = nn.ModuleList() # to be Q for self-attention
        self.attnLayers = nn.ModuleList()
        self.fwdLNs = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        self.lastLN = nn.LayerNorm(embedding_dim, eps=1e-8)

        for _ in range(num_blocks):
            self.attnLNs.append(nn.LayerNorm(
                embedding_dim, eps=1e-8
            ))

            self.attnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True # !!!
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                embedding_dim, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                embedding_dim, dropout_rate
            ))

        # False True  True ...
        # False False True ...
        # False False False ...
        # ....
        # True indices that the corresponding position is not allowed to attend !
        self.register_buffer(
            'attnMask',
            torch.ones((maxlen, maxlen), dtype=torch.bool).triu(diagonal=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs, seqs, 
            attn_mask=self.attnMask,
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def forward(self, seqs: torch.Tensor):
        padding_mask = (seqs == 0).unsqueeze(-1)
        seqs = self.Item.look_up(seqs) # (B, S) -> (B, S, D)
        seqs *= self.Item.dimension ** 0.5
        seqs = self.embdDropout(self.position(seqs))
        seqs.masked_fill_(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)
        
        features = self.lastLN(seqs) # (B, S, D)

        return features