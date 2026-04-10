---
name: cv-attention-lstm-image-captioning
description: >
  Show-Attend-Tell decoder: Bahdanau additive attention over CNN spatial features driving a gated LSTMCell for autoregressive image-to-sequence generation.
---
# Attention LSTM Image Captioning

## Overview

Generate sequences (captions, chemical formulas, LaTeX) from images using an attention-based LSTM decoder. At each timestep the decoder attends to CNN spatial features via additive (Bahdanau) attention, applies a learned sigmoid gate, then feeds the context vector + previous token embedding into an LSTMCell. The hidden state is initialized from the mean-pooled encoder output.

## Quick Start

```python
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attn_dim)
        self.decoder_att = nn.Linear(decoder_dim, attn_dim)
        self.full_att = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, h):
        att1 = self.encoder_att(encoder_out)           # (B, pixels, attn_dim)
        att2 = self.decoder_att(h).unsqueeze(1)        # (B, 1, attn_dim)
        alpha = F.softmax(self.full_att(torch.tanh(att1 + att2)).squeeze(2), dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward_step(self, encoder_out, embed_t, h, c):
        context, alpha = self.attention(encoder_out, h)
        gate = torch.sigmoid(self.f_beta(h))
        context = gate * context
        h, c = self.lstm_cell(torch.cat([embed_t, context], dim=1), (h, c))
        return self.fc(h), h, c, alpha
```

## Key Decisions

- **Gate (f_beta)**: Learned sigmoid gate controls how much visual context flows in; improves convergence
- **Init h/c**: Project mean-pooled encoder features through linear layers
- **Teacher forcing**: Use ground truth tokens during training; greedy/beam search at inference
- **Attention dim**: 256 is typical; larger = more expressive but slower

## References

- [InChI / Resnet + LSTM with attention / starter](https://www.kaggle.com/code/yasufuminakama/inchi-resnet-lstm-with-attention-starter)
- [Pytorch ResNet+LSTM with attention](https://www.kaggle.com/code/pasewark/pytorch-resnet-lstm-with-attention)
