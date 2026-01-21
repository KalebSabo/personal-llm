# Load and generate script
from quopri import decode

import torch

from llm import SimpleLLM


model = SimpleLLM().to(torch.device)
model.load_state_dict(torch.load('simple_llm.pth'))
model.eval()

# Generate as much as you want
context = torch.zeros((1, 1), dtype=torch.long, device=torch.device)
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))