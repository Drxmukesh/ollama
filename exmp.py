import pandas as pd
from sklearn.model_selection import train_test_split
from ollama import GemmaModel, Trainer, TrainingArguments

# Load and preprocess data
data = pd.read_csv('data.csv')
train_data, test_data = train_test_split(data, test_size=0.2)

# Model configuration
config = {
    "num_layers": 12,
    "hidden_size": 768,
    "num_heads": 12,
    "vocab_size": 30522,
    "max_position_embeddings": 512,
}

model = GemmaModel(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model
model.save_pretrained('./my_model')
