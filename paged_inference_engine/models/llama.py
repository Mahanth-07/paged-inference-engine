"""Minimal Llama 3.x, written from scratch.

Architecture:
    embed_tokens
    N x DecoderLayer:
        RMSNorm -> Attention(qkv proj, RoPE, PagedAttention, o proj) -> residual
        RMSNorm -> MLP(gate/up fused, SwiGLU, down)                  -> residual
    RMSNorm
    lm_head   (tied to embed_tokens on the 1B model)

Things that silently corrupt output, all handled below:
