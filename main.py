import torch
from transformer import Decoder

def main():
    torch.manual_seed(42)  # same seed -> same weights, same input, every run
    d_em = 8
    d_k = 16
    x = torch.randn(5, 3, d_em)
    att = Decoder(d_k=d_k, d_em=d_em, num_heads=2)
    print(att(x))

if __name__ == '__main__':
    main()
