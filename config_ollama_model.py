from ollama import GemmaModel

config = {
    "num_layers": 12,
    "hidden_size": 768,
    "num_heads": 12,
    "vocab_size": 30522,
    "max_position_embeddings": 512,
}

model = GemmaModel(config)

