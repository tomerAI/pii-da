import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import classification_report, f1_score
import numpy as np
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

class DanskBERTTokenClassifier:
    def __init__(self, model_name="vesteinn/DanskBERT", num_labels=19):  # 19 labels for your PII types
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup label mappings
        self.labels = [
            "O",
            "B-Navn", "I-Navn",
            "B-CPR", "I-CPR",
            "B-Adresser", "I-Adresser",
            "B-Religion", "I-Religion",
            "B-Politisk", "I-Politisk",
            "B-Fagforening", "I-Fagforening",
            "B-Genetik", "I-Genetik",
            "B-Seksuel", "I-Seksuel",
            "B-IP", "I-IP",
        ]
        self.index2tag = {idx: tag for idx, tag in enumerate(self.labels)}
        self.tag2index = {tag: idx for idx, tag in enumerate(self.labels)}
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.index2tag,
            label2id=self.tag2index
        )
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=self.config
        ).to(self.device)

    def tokenize_and_align_labels(self, examples):
        """Tokenize inputs and align labels."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for idx, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.index2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.index2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = classification_report(true_labels, true_predictions)
        return {
            "classification_report": results,
            "f1": f1_score(true_labels, true_predictions)
        }

    def train(self, dataset_path, output_dir, num_epochs=3, batch_size=16):
        """Train the model."""
        # Load dataset
        dataset = load_from_disk(dataset_path)
        
        # Encode dataset
        encoded_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

    def predict(self, text):
        """Predict NER tags for a given text."""
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get predictions
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert predictions to tags
        predicted_tags = [
            self.index2tag[p.item()] 
            for p in predictions[0]
        ]
        
        # Align with tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return list(zip(tokens, predicted_tags)) 