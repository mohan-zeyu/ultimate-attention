import torch
import torch.nn as nn
from transformer import Decoder

@torch.no_grad()
def test_kv_cache(device: torch.device):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    de = Decoder(d_k=128,
                 d_em=128,
                 num_heads=8,
                 max_len=64,
                 batch=4,
                 )
    de.eval().to(device)
    input = torch.ones(4, 4, 128).to(device)

    output = de(input)
    print(output)

def test_play():
    model = nn.Dropout(p=1.0)
    x = torch.tensor([2.0], requires_grad=True)

    # Training behavior, but no gradient recording
    model.train()
    with torch.no_grad():
        a = model(x)
    print(a, a.requires_grad)
    # tensor([0.]) False

    # Evaluation behavior, but WITH gradient recording
    model.eval()
    b = model(x)
    print(b, b.requires_grad)
    # tensor([2.], grad_fn=...) True

    b.backward()
    print(x.grad)
