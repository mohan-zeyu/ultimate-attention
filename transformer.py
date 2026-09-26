import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    r"""
    Dimension of input x in [N, n_seq, d_em]
    """
    def __init__(self, 
                 d_k: int = 32,
                 d_em: int = 16,
                 num_heads: int = 4,
                 max_len: int = 128,
                 batch: int = 4
                 ) -> None:
        super().__init__()
        assert d_k % num_heads == 0
        self.d_k = d_k
        self.d_em = d_em
        self.num_heads = num_heads
        self.batch = batch
        self.max_len = max_len
        self.W_q = nn.Linear(d_em, d_k)
        self.W_k = nn.Linear(d_em, d_k)
        self.W_v = nn.Linear(d_em, d_k)
        self.W_o = nn.Linear(d_k, d_em)

        # No, it seems that this line is correct. It revives.
        # --- > Previously:
        #     # NO! It shouldn't be located here. 
        #     # state_dict will take it. Also, the attention layer exists forever, so KV cache won't be offloaded.
        #     # [TODO] Remove it
        # rememeber such self-defined tensors often require manually-handled placing
        self.kv_cache = KVCache(self.d_k, self.num_heads, self.max_len, self.batch)
        # We can separate the training and inference processes now.

        # But there're still problems. 
        # 1. Training doesn't need it. I think there are two ways to do it:
        #   1.  Closure: let's put the kv_cache into a closure called kv_cache_init(). emm... keeping private state..
        # NO, too complex, just import the None
    @property
    def use_cache(self) -> bool:
        return not self.training

    def _convert(self, w: torch.Tensor, num_heads: int) -> torch.Tensor:
        N, n_seq, _ = w.shape
        return w.reshape(N, n_seq, num_heads, -1).transpose(1, 2)

    def _clear_kv(self):
        if self.kv_cache is None:
            raise RuntimeError("KV cache doesn't exist!")
        self.kv_cache.clear()

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        r"""
        Careful! 
        From perspective of forward method, KV cache should manage itself well. 
        All forward() needs to do is getting previous K and V, and also update cache.
        
        So we need external signal to manage the KV cache state, so that KV will be cleaned at proper time 
        though the forward() and Attention block is unaware.
        """

        if x.shape[0] != self.batch:
            raise ValueError
        # Note that the KVCache should be model unaware
        # So we just simply compute the new one and then insert it in.
        Q = self._convert(self.W_q(x), self.num_heads)

        if self.use_cache:
            # For positional embedding techniques like RoPE, we need to add additional operations before update
            new_K = self._convert(self.W_k(x), self.num_heads) 
            new_V = self._convert(self.W_v(x), self.num_heads) 

            if self.kv_cache.update(new_K, new_V) == False:
                raise RuntimeError("Exceeding the maximum length!")
            K, V = self.kv_cache.get()
        else:
            K = self._convert(self.W_k(x), self.num_heads)
            V = self._convert(self.W_v(x), self.num_heads)

        q_len, k_len = Q.size(-2), K.size(-2)
        query_positions = torch.arange(k_len - q_len, k_len, device=Q.device)
        key_positions = torch.arange(k_len, device=Q.device)
        future = query_positions[:, None] < key_positions[None, :]

        scores = Q @ torch.transpose(K, -1, -2) / math.sqrt(self.d_k / self.num_heads)
        ratio = F.softmax(scores.masked_fill(future, float('-inf')), dim=-1) @ V
        N, n_seq, _ = x.shape
        ratio = ratio.transpose(1, 2).reshape(N, n_seq, self.d_k)
        y = self.W_o(ratio)
        return y

class KVCache(nn.Module):
    k: torch.Tensor
    v: torch.Tensor

    def __init__(self,
                 d_k: int,
                 num_heads: int,
                 max_len: int,
                 batch: int = 1,
                 ) -> None:
        super().__init__()
        # it can't mismatch
        self.batch = batch
        self.max_len = max_len
    # we do not need to assert because it has been done in attention
        self.d = d_k // num_heads
        self.num_heads = num_heads
        self.register_buffer('k', torch.empty((batch, num_heads, self.max_len, self.d)), persistent=False)
        self.register_buffer('v', torch.empty_like(self.k), persistent=False)
        # With pre-allocated memory, we could reduce copies.
        self.counter = 0

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, :self.counter, :], \
               self.v[:, :, :self.counter, :]


    def update(self, new_k, new_v) -> bool:
        # Now it seems that we could afford multiple tokens for decoding?
        len = new_k.shape[2]

        # Later maybe we could add some memory pool growing code here?

        if self.counter + len > self.max_len:
            return False

        self.k[:, :, self.counter: self.counter + len, :] = new_k
        self.v[:, :, self.counter: self.counter + len, :] = new_v

        self.counter += len
        return True
    
    def clear(self) -> None:
        self.counter = 0


class Decoder(nn.Module):
    def __init__(self,
                 d_k: int = 32,
                 d_em: int = 16,
                 num_heads: int = 4,
                 max_len: int = 128,
                 batch: int = 4
                 ) -> None:
        super().__init__()
        self.d_hid = 4 * d_em
        self.attention = Attention(d_k, d_em, num_heads, max_len, batch)
        self.ffn = nn.Sequential(
            nn.Linear(d_em, self.d_hid),
            nn.GELU(),
            nn.Linear(self.d_hid, d_em)
        )
        self.ln_att = nn.LayerNorm(d_em)
        self.ln_ffn = nn.LayerNorm(d_em)

    def _clear_kv(self):
        self.attention._clear_kv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_att(x + self.attention(x))
        x = self.ln_ffn(x + self.ffn(x))
        return x
