from datasets import load_from_disk
import pandas as pd
import json
from collections import Counter

def load_and_explore_dataset():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_from_disk("data/danish_ner_pii_dataset")
    
    # Print basic information about the dataset
    print("\nDataset Structure:")
    print(dataset)
    
    # Explore each split
    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\n{split_name.upper()} Split Statistics:")
        print(f"Number of samples: {len(split)}")
        
        # Count PII types
        pii_type_counts = Counter(split['pii_type'])
        print("\nPII Type Distribution:")
        for pii_type, count in pii_type_counts.items():
            print(f"{pii_type}: {count} samples")
        
        # Sample and display some examples
        print(f"\nSample entries from {split_name} split:")
        for i in range(min(3, len(split))):  # Show up to 3 examples
            print(f"\nExample {i+1}:")
            print(f"Text: {split[i]['text']}")
            print(f"Tokens: {split[i]['tokens']}")
            print(f"NER tags: {split[i]['ner_tags']}")
            print(f"PII type: {split[i]['pii_type']}")
            
        # Calculate average text length
        avg_text_length = sum(len(text.split()) for text in split['text']) / len(split)
        print(f"\nAverage text length (words): {avg_text_length:.2f}")
        
        # Count total entities
        total_entities = sum(1 for tags in split['ner_tags'] for tag in tags if tag != 0)
        print(f"Total number of entities: {total_entities}")
        print(f"Average entities per sample: {total_entities/len(split):.2f}")

if __name__ == "__main__":
    load_and_explore_dataset() 