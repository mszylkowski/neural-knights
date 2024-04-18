import numpy as np
import torch
from torch import nn

from utils.moves import (NUM_POSSIBLE_MOVES, NUM_OF_SQUARES,
    NUM_OF_PIECE_TYPES, PAD_MOVE, START_MOVE)


class Transformer(nn.Module):
    def __init__(self, device: torch.device | None = None,
                 num_heads: int = 2,
                 dim_feedforward: int = 2048,
                 num_layers_enc: int = 2,
                 num_layers_dec: int = 2,
                 dropout: float = 0.2,
                 sequence_length: int = 6,
                 ignore_index: int = PAD_MOVE,
                 start_index: int = START_MOVE):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = NUM_OF_SQUARES * NUM_OF_PIECE_TYPES
        self.dim_feedforward = dim_feedforward
        self.sequence_length = sequence_length
        self.output_size = NUM_POSSIBLE_MOVES
        self.pad_idx = ignore_index
        self.start_idx = start_index

        # Initialize the transformer layer.
        self.transformer = nn.Transformer(d_model=self.hidden_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers_enc,
                                          num_decoder_layers=num_layers_dec,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        # Initialize embeddings.
        self.srcposembeddingL = nn.Embedding(sequence_length, self.hidden_dim)
        self.tgtembeddingL = nn.Embedding(self.output_size, self.hidden_dim)
        self.tgtposembeddingL = nn.Embedding(sequence_length, self.hidden_dim)
        # Initialize final fc layer.
        self.fc_final = nn.Linear(self.hidden_dim, self.output_size)

        if device:
            self.to(device)

    def _shift_tgt(self, tgt):
        """Shifts tgt right by 1 and fills the first index with start_idx."""
        modified_tgt = torch.full_like(tgt, self.start_idx)
        modified_tgt[:, 1:] = tgt[:, :-1]
        return modified_tgt

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, shift_tgt: bool = True):
        """Computes transfomer forward and returns outputs.

        Parameters:
            src: Tensor of shape (N,T,12,8,8), a sequence of chess boards.
            tgt: Tensor of shape (N,T) a sequence of moves.

        Returns:
            The model outputs: move scores of shape (N,T,output_size).
        """
        src = src.type(torch.int)
        tgt = tgt.type(torch.int)

        if shift_tgt:
            tgt = self._shift_tgt(tgt)
            pass

        # Collaps src to a three dim tensor (N,T,12x8x8).
        src = src.view(src.size(0), src.size(1), -1)

        # Add posembed to src and tgt to take into account the sequential
        # nature of positions and moves.
        N, T, _ = src.shape
        src_embed = (src
                     + self.srcposembeddingL(torch.arange(T).repeat((N,1))))
        tgt_embed = (self.tgtembeddingL(tgt)
                     + self.tgtposembeddingL(torch.arange(T).repeat((N,1))))

        # Create target mask and target key padding mask for decoder - Both
        # have boolean values. We want to ignore (i.e. True(s)) lower triangle,
        # but not diagonal.
        tgt_mask = self.transformer.generate_square_subsequent_mask(T)
        tgt_key_padding_mask = torch.zeros((N, T), dtype=torch.bool)
        tgt_key_padding_mask[tgt == self.pad_idx] = True

        out = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.fc_final(out)
        return out
