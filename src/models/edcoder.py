from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .sgc import SGC
from .dot_gat import DotGAT
from .loss_func import sce_loss
from utils import create_norm


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "sgc":
        mod = SGC(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.concat_linear = nn.Linear(enc_num_hidden * 2, enc_num_hidden)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, mask_nodes

    def encoding_feature_level_mask(self, g, x, mask_rate=0.3):
        normal_dist = torch.randn_like(x)
        offset = torch.log(torch.tensor(mask_rate / (1 - mask_rate)))
        mask_probabilities = torch.sigmoid(normal_dist + offset)
        mask = torch.bernoulli(mask_probabilities).to(x.device)
        out_x = x * (1 - mask)
        use_g = g.clone()

        return use_g, out_x, mask

    def new_forward(self, g, x, conjx, missing_feature_mask):
        loss = self.new_mask_attr_prediction(g, x, conjx, missing_feature_mask)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    # def mask_attr_prediction(self, g, x):
    #     # pre_use_g, use_x, mask_nodes = self.encoding_mask_noise(g, x, self._mask_rate)
    #     pre_use_g, use_x, mask = self.encoding_feature_level_mask(g, x, self._mask_rate)

    #     use_g = pre_use_g

    #     enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
    #     if self._concat_hidden:
    #         enc_rep = torch.cat(all_hidden, dim=1)

    #     rep = self.encoder_to_decoder(enc_rep)

    #     # rep[mask_nodes] = 0
    #     normal_dist = torch.randn_like(rep)
    #     offset = torch.log(torch.tensor(self._mask_rate / (1 - self._mask_rate)))
    #     mask_probabilities = torch.sigmoid(normal_dist + offset)
    #     lowdim_mask = torch.bernoulli(mask_probabilities).to(rep.device)
    #     rep = rep * (1 - lowdim_mask)

    #     recon = self.decoder(pre_use_g, rep)      

    #     # x_init = x[mask_nodes]
    #     # x_rec = recon[mask_nodes]
    #     x_init = x * (1 - mask)
    #     x_rec = recon * (1 - mask)

    #     loss = self.criterion(x_rec, x_init)
    #     return loss

    def new_mask_attr_prediction(self, g, x, conjx, missing_feature_mask):
        # pre_use_g, use_x, mask_nodes = self.encoding_mask_noise(g, x, self._mask_rate)
        pre_use_g, use_x, mask = self.encoding_feature_level_mask(g, x, self._mask_rate)

        use_g = pre_use_g

        if self._encoder_type == "gcn":
            x_true = use_x.clone()
            x_false = use_x.clone()

            x_true[missing_feature_mask] = 0
            x_false[~missing_feature_mask] = 0
            
            enc_rep_true, _ = self.encoder(use_g, x_true, return_hidden=True)
            enc_rep_false, _ = self.encoder(use_g, x_false, return_hidden=True)
            enc_rep_combined = torch.cat((enc_rep_true, enc_rep_false), dim=1)
            enc_rep = self.concat_linear(enc_rep_combined)

        else:
            enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

        # if self._concat_hidden:
        #     enc_rep = torch.cat(all_hidden, dim=1)

        rep = self.encoder_to_decoder(enc_rep)
   
        # rep[mask_nodes] = 0
        normal_dist = torch.randn_like(rep)
        offset = torch.log(torch.tensor(self._mask_rate / (1 - self._mask_rate)))
        mask_probabilities = torch.sigmoid(normal_dist + offset)
        lowdim_mask = torch.bernoulli(mask_probabilities).to(rep.device)
        rep = rep * (1 - lowdim_mask)

        recon = self.decoder(pre_use_g, rep)

        # x_init = conjx[mask_nodes]
        # x_rec = recon[mask_nodes]
        x_init = conjx * (1 - mask)
        x_rec = recon * (1 - mask)

        loss = self.criterion(x_rec, x_init)
        return loss

    ### 得到重建的 embeddings 函数
    ### 这里就不对之前的再次进行 Mask
    def get_reconstraction(self, g, x):
        enc_rep, all_hidden = self.encoder(g, x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        
        normal_dist = torch.randn_like(rep)
        offset = torch.log(torch.tensor(self._mask_rate / (1 - self._mask_rate)))
        mask_probabilities = torch.sigmoid(normal_dist + offset)
        lowdim_mask = torch.bernoulli(mask_probabilities).to(rep.device)
        rep = rep * (1 - lowdim_mask)

        recon = self.decoder(g, rep)

        x_rec = recon

        return x_rec

    def embed(self, g, x, edge_index):
        # rep = self.encoder(g, x)
        rep = x
        row, col = edge_index
        logits = rep[row] * rep[col]
        return logits

    # def missing_embed_zero(self, g, x):
    #     feat = torch.nan_to_num(x, nan=0.0)
    #     rep = self.encoder(g, feat)
    #     return rep

    def missing_embed_mask(self, g, x, missing_feature_mask):
        # feat = torch.where(torch.isnan(x), self.enc_mask_token, x)
        if self._encoder_type == "gcn":
            x_true = x.clone()
            x_false = x.clone()

            x_true[missing_feature_mask] = 0
            x_false[~missing_feature_mask] = 0
            enc_rep_true, _ = self.encoder(g, x_true, return_hidden=True)
            enc_rep_false, _ = self.encoder(g, x_false, return_hidden=True)
            enc_rep_combined = torch.cat((enc_rep_true, enc_rep_false), dim=1)
            rep = self.concat_linear(enc_rep_combined)

        else:
            x = self.get_reconstraction(g,x)
            rep = self.encoder(g, x)

        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
