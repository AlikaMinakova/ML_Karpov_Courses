import torch
import torch.nn as nn

from encoder_decoder import Decoder
from encoder_decoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int, # кол-во слов в словаре языка с которого будем переводить
        output_dim: int, # кол-во слов в словаре языка на который будем переводить
        hidden_dim: int, # размерность эмбендингов (для слова)
        n_layers: int, # кол-во блоков энкодера-декодера
        n_heads: int, # кол-во голов в слое self-attention
        forward_expansion: int,
        dropout: float,
        src_pad_idx: int,
        trg_pad_idx: int,
        device: torch.device,
        max_length: int = 512,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            pad_idx=src_pad_idx,
            max_length=max_length,
        )

        self.decoder = Decoder(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            pad_idx=trg_pad_idx,
            max_length=max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device

    # хотим чтобы модель не обращала внимания на падинг
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

    # не хотим чтобы декодер смотрел вправо, только на то что уже имеем
    def make_trg_mask(self, trg):
        # https://pytorch.org/docs/stable/generated/torch.tril.html
        trg_mask = torch.tril(
            torch.ones((trg.shape[1], trg.shape[1]), device=self.device)
        ).bool()
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) & trg_mask

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        # послед-ость на языке с которого переводим
        src_mask = self.make_src_mask(src)
        # послед-ость на целевом языке
        trg_mask = self.make_trg_mask(trg)

        return self.decoder(trg, self.encoder(src, src_mask), src_mask, trg_mask)
