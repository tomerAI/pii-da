from datasets import Dataset, DatasetDict
from typing import List, Dict
from src.data_generator import DataGenerator

class DatasetBuilder:
    def __init__(self, generator: DataGenerator):
        self.generator = generator
        
    def create_dataset_split(self, samples_per_type: int) -> Dataset:
        """Create a dataset split with balanced PII types."""
        samples = self.generator.generate_text(samples_per_type)
        
        dataset_examples = {
            "id": [],
            "text": [],
            "tokens": [],
            "ner_tags": [],
            "pii_type": []  # New field to track PII type
        }
        
        for idx, (text, tokens, labels) in enumerate(samples):
            # Calculate which PII type this sample belongs to
            pii_type_idx = idx // samples_per_type
            pii_type = list(self.generator.pii_types.keys())[pii_type_idx]
            
            dataset_examples["id"].append(str(idx))
            dataset_examples["text"].append(text)
            dataset_examples["tokens"].append(tokens)
            dataset_examples["ner_tags"].append(
                [self.generator.labels.index(label) for label in labels]
            )
            dataset_examples["pii_type"].append(pii_type)
            
        return Dataset.from_dict(dataset_examples)
        
    def create_full_dataset(self, train_samples: int, val_samples: int, test_samples: int) -> DatasetDict:
        """
        Create full dataset with splits.
        Note: The actual number of samples will be multiplied by the number of PII types.
        """
        return DatasetDict({
            "train": self.create_dataset_split(train_samples),
            "validation": self.create_dataset_split(val_samples),
            "test": self.create_dataset_split(test_samples)
        })
