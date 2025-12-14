import torch
import pickle
from tiny_shakespeare_refactored import GPTLanguageModel, Tokenizer, DataLoader, GPTConfig
from pathlib import Path
from time import sleep

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "model.pkl"

# Load text and tokenizer
text_path = Path("input.txt")
if not text_path.exists():
    raise FileNotFoundError("input.txt not found. Download it first.")
text = text_path.read_text(encoding="utf-8")
tokenizer = Tokenizer(text)

# Build model and load weights
config = GPTConfig(vocab_size=len(tokenizer.chars))

model = GPTLanguageModel(
    config.vocab_size, config.n_embd, config.n_head, config.n_layer, dropout=config.dropout
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Stream endless text generation (Ctrl+C to stop) using a context of block_size tokens
block_size = 32
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Initial context is a single zero token

with torch.no_grad():
    try:
        while True: # Infinite loop
            context = model.generate(context, max_new_tokens=1, block_size=block_size) # Generate a new token
            new_token = context[0, -1].item() # Get the last token generated
            print(tokenizer.decode([new_token]), end="", flush=True) # Print the new token decoded
            context = context[:, -block_size:] # Update the context to the last block_size tokens
            sleep(0.03) # Wait 30 milliseconds before generating the next token
    except KeyboardInterrupt:
        print("\nGeneration stopped by user.")