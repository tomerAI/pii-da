from model_builder import DanskBERTTokenClassifier

def main():
    # Initialize classifier
    classifier = DanskBERTTokenClassifier()
    
    # Train model
    trainer = classifier.train(
        dataset_path="data/danish_ner_pii_dataset",
        output_dir="models/danish_bert_ner",
        num_epochs=3,
        batch_size=16
    )
    
    # Test prediction
    test_text = "Anders Jensen bor på Østergade 52 i København"
    predictions = classifier.predict(test_text)
    
    print("\nTest Prediction:")
    for token, tag in predictions:
        print(f"{token}: {tag}")

if __name__ == "__main__":
    main() 