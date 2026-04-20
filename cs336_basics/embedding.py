import torch
from torch import nn
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        embedding_mat = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embedding_mat, mean=0.0, std=1, a=-3, b=3)
        self.embedding_mat = nn.Parameter(embedding_mat)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.embedding_mat[token_ids]





       
