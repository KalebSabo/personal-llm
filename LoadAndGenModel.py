# Load and generate script
import torch
import glob
from llm import SimpleLLM, decode

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Find the most recent model file
model_files = glob.glob('simple_llm_iter_*.pth')
if not model_files:
    print("No saved model found. Please train the model first by running llm.py")
    exit(1)

model_file = max(model_files)  # Get the most recent one
print(f"Loading model: {model_file}")

model = SimpleLLM().to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Generate text
print("\nGenerating text...\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))