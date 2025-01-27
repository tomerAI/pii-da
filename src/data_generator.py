import os
from typing import List, Tuple, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class GeneratedSample(BaseModel):
    text: str
    tokens: List[str]
    labels: List[str]

class DataGenerator:
    def __init__(self, api_key: str):
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
        
        # Map PII types to their descriptions and example formats
        self.pii_types = {
            "Navn": {
                "description": "Danish full names",
                "format": "e.g., 'Anders Jensen', 'Marie Hansen'"
            },
            "CPR": {
                "description": "Danish CPR numbers",
                "format": "format: DDMMYY-XXXX"
            },
            "Adresser": {
                "description": "Danish addresses",
                "format": "e.g., 'Østergade 52, 2100 København Ø'"
            },
            "Religion": {
                "description": "religious beliefs or affiliations",
                "format": "e.g., 'muslim', 'buddhist', 'kristen'"
            },
            "Politisk": {
                "description": "political affiliations or views",
                "format": "e.g., 'socialdemokrat', 'konservativ'"
            },
            "Fagforening": {
                "description": "union memberships",
                "format": "e.g., '3F medlem', 'medlem af DJØF'"
            },
            "Genetik": {
                "description": "genetic information",
                "format": "e.g., 'bærer af genmutation'"
            },
            "Seksuel": {
                "description": "sexual orientation",
                "format": "e.g., 'homoseksuel', 'biseksuel'"
            },
            "IP": {
                "description": "IP addresses",
                "format": "format: XXX.XXX.XXX.XXX"
            }
        }
        
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=1.2,
            openai_api_key=api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=GeneratedSample)
        
        self.prompt_template = """
        You are an expert in generating and labeling Danish text containing Personal Identifiable Information (PII).

        For this batch, focus specifically on generating text containing {pii_type}: {pii_description} {pii_format}

        Generate a short, natural-sounding Danish text paragraph that MUST include the specified PII type.
        Make the text sound natural and contextually appropriate.

        Then, tokenize and label the generated text using IOB2 format with these labels:
        {labels}

        Rules:
        1. Each token must be labeled
        2. Use B- prefix for beginning of entity
        3. Use I- prefix for continuation of entity
        4. Use O for non-entity tokens
        5. Maintain proper IOB2 format (I- cannot come after O)
        6. MUST include the specified PII type ({pii_type})

        Output the generated text, tokens, and their labels in this format:
        {format_instructions}
        """
        
        self.prompt = PromptTemplate(
            input_variables=["pii_type", "pii_description", "pii_format", "labels"],
            template=self.prompt_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def generate_batch(self, pii_type: str, batch_size: int) -> List[Tuple[str, List[str], List[str]]]:
        """Generate a batch of samples for a specific PII type."""
        labels_str = "\n".join(f"- {label}" for label in self.labels)
        pii_info = self.pii_types[pii_type]
        
        chain = self.prompt | self.model | self.parser
        
        samples = []
        for _ in range(batch_size):
            try:
                result = chain.invoke({
                    "pii_type": pii_type,
                    "pii_description": pii_info["description"],
                    "pii_format": pii_info["format"],
                    "labels": labels_str
                })
                samples.append((result.text, result.tokens, result.labels))
            except Exception as e:
                print(f"Error in generation for {pii_type}: {e}")
        
        return samples
    
    def generate_text(self, samples_per_type: int = 1) -> List[Tuple[str, List[str], List[str]]]:
        """Generate samples for all PII types."""
        all_samples = []
        
        # Generate batches for each PII type
        for pii_type in self.pii_types.keys():
            print(f"Generating batch for {pii_type}...")
            batch_samples = self.generate_batch(pii_type, samples_per_type)
            all_samples.extend(batch_samples)
            
        return all_samples
