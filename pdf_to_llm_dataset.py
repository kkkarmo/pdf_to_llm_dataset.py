import os
import pdfplumber
import requests
from datasets import Dataset
from transformers import AutoTokenizer, pipeline
from langchain.llms import Groq
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
input_file = os.getenv('INPUT_FILE', 'book.pdf')
output_text_file = os.getenv('OUTPUT_TEXT_FILE', 'extracted_text.txt')
output_dataset_dir = os.getenv('OUTPUT_DATASET_DIR', 'dataset')
output_tokenized_dir = os.getenv('OUTPUT_TOKENIZED_DIR', 'tokenized_dataset')
hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
llama_url = "http://20.20.20.26:11343/generate"  # LLAMA2 endpoint

# Validate environment variables
required_vars = ['INPUT_FILE', 'HUGGINGFACE_API_KEY', 'GROQ_API_KEY']
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Initialize models and pipelines
try:
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", use_auth_token=hf_api_key)
    groq_llm = Groq(groq_api_key=groq_api_key)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token=hf_api_key)
except Exception as e:
    logging.error(f"Error initializing models: {e}")
    raise

def extract_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in tqdm(pdf.pages, desc="Extracting pages"):
                text += page.extract_text() + '\n'
        return text
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while extracting text: {e}")
        raise

def process_text(text):
    cleaned_text = text.replace('\n', ' ').replace('\r', '')
    sentences = cleaned_text.split('. ')
    return [s.strip() for s in sentences if s.strip()]

def enrich_with_ner(sentences):
    enriched_data = []
    batch_size = 32  # Adjust based on your GPU memory
    for i in tqdm(range(0, len(sentences), batch_size), desc="NER Processing"):
        batch = sentences[i:i+batch_size]
        try:
            entities_batch = ner_pipeline(batch)
            for sentence, entities in zip(batch, entities_batch):
                enriched_data.append({"text": sentence, "entities": entities})
        except Exception as e:
            logging.error(f"Error in NER for batch starting at {i}: {e}")
    return enriched_data

def augment_data(examples):
    augmented = []
    for example in tqdm(examples, desc="Data Augmentation"):
        try:
            prompt = f"Generate a paraphrase of the following text: {example['text']}"
            paraphrase = groq_llm(prompt)
            augmented.append({"original": example['text'], "paraphrase": paraphrase, "entities": example['entities']})
        except Exception as e:
            logging.error(f"Error in data augmentation: {e}")
            augmented.append({"original": example['text'], "paraphrase": "", "entities": example['entities']})
    return augmented

def quality_check(examples):
    checked = []
    for example in tqdm(examples, desc="Quality Check"):
        prompt = f"Summarize this text in one sentence: {example['original']}"
        try:
            response = requests.post(llama_url, json={"prompt": prompt}, timeout=10)
            response.raise_for_status()
            summary = response.json().get('generated_text', 'No summary generated')
        except requests.exceptions.RequestException as e:
            logging.error(f"Error in generating summary: {e}")
            summary = "Error in generating summary"
        checked.append({**example, "summary": summary})
    return checked

def tokenize_function(examples):
    return tokenizer(examples['original'], padding='max_length', truncation=True, max_length=512)

def main():
    try:
        # Step 1: Text Extraction
        logging.info("Extracting text from PDF...")
        raw_text = extract_text(input_file)
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        # Step 2: Text Processing
        logging.info("Processing extracted text...")
        processed_text = process_text(raw_text)

        # Step 3: Enrich with Named Entity Recognition
        logging.info("Enriching data with NER...")
        enriched_data = enrich_with_ner(processed_text)

        # Step 4: Create initial dataset
        logging.info("Creating initial dataset...")
        dataset = Dataset.from_list(enriched_data)

        # Step 5: Data Augmentation
        logging.info("Augmenting data...")
        augmented_dataset = Dataset.from_list(augment_data(dataset))

        # Step 6: Quality Check
        logging.info("Performing quality check...")
        checked_dataset = Dataset.from_list(quality_check(augmented_dataset))

        # Step 7: Tokenization
        logging.info("Tokenizing dataset...")
        tokenized_dataset = checked_dataset.map(tokenize_function, batched=True, remove_columns=checked_dataset.column_names)

        # Save the final dataset
        logging.info("Saving final dataset...")
        tokenized_dataset.save_to_disk(output_tokenized_dir)
        checked_dataset.save_to_disk(output_dataset_dir)

        logging.info("Process completed successfully!")
    except Exception as e:
        logging.error(f"An error occurred during the main process: {e}")

if __name__ == "__main__":
    main()