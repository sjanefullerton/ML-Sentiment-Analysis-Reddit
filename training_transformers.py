import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load data and map sentiment labels to numerical values
data = pd.read_csv("fixed_160redditactual_sentimentlabels.csv").head(158)

# Strip any leading/trailing whitespace from the label columns
data['Title_Label'] = data['Title_Label'].str.strip()
data['Post_Content_Label'] = data['Post_Content_Label'].str.strip()
data['Comments_Label'] = data['Comments_Label'].str.strip()

# Remove rows where any label is missing (NaN, None, empty string)
data = data.dropna(subset=['Title_Label', 'Post_Content_Label', 'Comments_Label'])

# Define the replacement mapping
replacement_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}

# Apply the mapping to all label columns
data[['Title_Label', 'Post_Content_Label', 'Comments_Label']] = data[['Title_Label', 'Post_Content_Label', 'Comments_Label']].replace(replacement_mapping)
# Creating dataset for each type of text to fine-tune on.
# Title sentiment:
title_data = data[['Title', 'Title_Label']].rename(columns={"Title": "text", "Title_Label": "labels"})
title_dataset = Dataset.from_pandas(title_data)
print(title_dataset.column_names)

# Post Content sentiment:
content_data = data[['Post Content', 'Post_Content_Label']].rename(columns={"Post Content": "text", "Post_Content_Label": "labels"})
content_dataset = Dataset.from_pandas(content_data)


# Comments sentiment:
comments_data = data[['Comments', 'Comments_Label']].rename(columns={"Comments": "text", "Comments_Label": "labels"})
comments_dataset = Dataset.from_pandas(comments_data)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    # Tokenize only the text and keep labels unchanged
    tokenized_output = tokenizer(
        text=example["text"],
        padding="max_length",
        truncation=True,
        max_length=512  # or whatever max_length you are using
    )
    tokenized_output["labels"] = example["labels"]  # Ensure labels are untouched
    return tokenized_output

# Apply the tokenization function to the datasets
title_dataset = title_dataset.map(tokenize_function, batched=True)
content_dataset = content_dataset.map(tokenize_function, batched=True)
comments_dataset = comments_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
title_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
content_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
comments_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Set up model for fine-tuning with 3 labels for positive, negative, and neutral
#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
title_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
content_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
comments_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create Trainer instances and train for each dataset

# Title Trainer
title_trainer = Trainer(
    model=title_model,
    args=training_args,
    train_dataset=title_dataset,
    eval_dataset=title_dataset,  # For simplicity, using the same dataset for evaluation
)
title_trainer.train()
title_model.save_pretrained("./title_fine_tuned_distilbert")

# Content Trainer
content_trainer = Trainer(
    model=content_model,
    args=training_args,
    train_dataset=content_dataset,
    eval_dataset=content_dataset,  # For simplicity, using the same dataset for evaluation
)
content_trainer.train()
content_model.save_pretrained("./content_fine_tuned_distilbert")

# Comments Trainer
comments_trainer = Trainer(
    model=comments_model,
    args=training_args,
    train_dataset=comments_dataset,
    eval_dataset=comments_dataset,  # For simplicity, using the same dataset for evaluation
)
comments_trainer.train()
comments_model.save_pretrained("./comments_fine_tuned_distilbert")

# Save the tokenizer once after training is complete
tokenizer.save_pretrained("./tokenizer_fine_tuned_distilbert")

