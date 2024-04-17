import numpy as np
import torch
from torch import nn

from utils.moves import get_all_moves, NUM_OF_SQUARES, NUM_OF_PIECE_TYPES, PAD_MOVE


class Transformer(nn.Module):
    def __init__(self, device: torch.device | None = None,
                 hidden_dim: int = NUM_OF_SQUARES * NUM_OF_PIECE_TYPES,
                 num_heads: int = 2,
                 dim_feedforward: int = 2048,
                 num_layers_enc: int = 2,
                 num_layers_dec: int = 2,
                 dropout: float = 0.2,
                 sequence_length: int = 6,
                 ignore_index: int = PAD_MOVE):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.sequence_length = sequence_length
        self.output_size = len(get_all_moves())
        print('### output size: ', self.output_size)
        self.pad_idx=ignore_index

        # Initialize the transformer layer.
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads,
                                          num_encoder_layers=num_layers_enc,
                                          num_decoder_layers=num_layers_dec,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        # Initialize embeddings.
        self.srcposembeddingL = nn.Embedding(sequence_length, hidden_dim)
        self.tgtembeddingL = nn.Embedding(self.output_size, hidden_dim)
        self.tgtposembeddingL = nn.Embedding(sequence_length, hidden_dim)
        # Initialize final fc layer.
        self.fc_final = nn.Linear(hidden_dim, self.output_size)

        if device:
            self.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Computes transfomer forward and returns outputs.

        Parameters:
            src: Tensor of shape (N,T,12,8,8), a sequence of chess boards.
            tgt: Tensor of shape (N,T) a sequence of moves.

        Returns:
            The model outputs: move scores of shape (N,T,output_size).
        """
        src = src.type(torch.int)
        tgt = tgt.type(torch.int)

        # Collaps src to a three dim tensor (N,T,12x8x8).
        src = src.view(src.size(0), src.size(1), -1)

        # Add posembed to src and tgt to take into account the sequential
        # nature of positions and moves.
        N, T, _ = src.shape
        #print('emb src.shape: ', self.srcembeddingL(src).shape)
        #print('src pose: ',
              #self.srcposembeddingL(torch.arange(T).repeat((N,1))).shape)
        print('tgt.shape: ', self.tgtembeddingL(tgt).shape)
        print('tgt pose shape: ', self.tgtposembeddingL(torch.arange(T).repeat((N,1))).shape)
        src_embed = (src
                     + self.srcposembeddingL(torch.arange(T).repeat((N,1))))
        try:
            tgt_embed = (self.tgtembeddingL(tgt)
                         + self.tgtposembeddingL(torch.arange(T).repeat((N,1))))
        except IndexError:
            print(tgt)
            raise IndexError("index out of range in self")

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

    def generate_move(self, src):
        """
         This function generates the output of the transformer taking src as
         its input it is assumed that the model is trained. The output would be
         the translation of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        N, T = src.shape
        # initially set outputs as a tensor of zeros with dimensions (batch_size, seq_len, output_size)
        outputs = torch.zeros(N, self.sequence_length, self.output_size)

        tgt = torch.full_like(src, self.pad_idx)
        tgt[:,0] = src[:,0]
        for i in range(T):
            t_outputs = self.forward(src, tgt)
            # argmax (N, T) -> get top ith word
            top_word = torch.argmax(t_outputs[:,i,:], dim=-1)
            if i < T - 1:
                tgt[:,i+1] = top_word
            outputs[:,i,:] = t_outputs[:,i,:]
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
