from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset


# Function to read and format the data
def read_and_format_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    stories = content.split('---')
    stories = [story.strip() for story in stories if story.strip()]
    return stories


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.inputs = [self.tokenizer.encode(text, truncation=True, padding='max_length', max_length=512,
                                             return_tensors='pt').squeeze(0) for text in texts]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx],
                'attention_mask': torch.ones_like(self.inputs[idx])}  # Use ones for attention mask


# Path to your dataset file
file_path = 'custom_dataset.txt'
stories = read_and_format_data(file_path)

# Initialize the tokenizer and prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Add padding token

dataset = CustomDataset(stories, tokenizer)

# Initialize the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings if padding token is added

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to=None  # Disable reporting if using custom setup
)

# Initialize the DataCollator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is not a masked language model
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')
