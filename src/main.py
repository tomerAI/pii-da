import os
from dotenv import load_dotenv
from src.data_generator import DataGenerator
from dataset_builder import DatasetBuilder

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    generator = DataGenerator(api_key=os.getenv("OPENAI_API_KEY"))
    dataset_builder = DatasetBuilder(generator)
    
    # Calculate samples per PII type
    # We have 9 PII types, so total samples will be multiplied by 9:
    samples_per_type = {
        "train": 11,      # 11 samples * 9 PII types = 99 total training samples
        "validation": 2,  # 2 samples * 9 PII types = 18 validation samples
        "test": 2        # 2 samples * 9 PII types = 18 test samples
    }
    
    # Create dataset with specified splits
    dataset = dataset_builder.create_full_dataset(
        train_samples=samples_per_type["train"],
        val_samples=samples_per_type["validation"],
        test_samples=samples_per_type["test"]
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in dataset:
        print(f"{split} split size: {len(dataset[split])} samples")
        print(f"PII types in {split}:", set(dataset[split]["pii_type"]))
    
    # Save the dataset locally
    dataset.save_to_disk("data/danish_ner_pii_dataset")
    print("\nDataset successfully created and saved to 'data/danish_ner_pii_dataset' directory")

if __name__ == "__main__":
    main()
