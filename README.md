# PDF to LLM Dataset Converter

This project provides a Python script that converts a PDF book into a training dataset suitable for large language models (LLMs). The script extracts text from the PDF, processes it, enriches it with named entity recognition (NER), augments the data, performs quality checks, and tokenizes the dataset.

## Features

- Extracts text from PDF files using `pdfplumber`.
- Processes and cleans the extracted text.
- Enriches text with named entity recognition using Hugging Face Transformers.
- Augments data by generating paraphrases with Groq LLM.
- Performs quality checks by summarizing the text using a local LLAMA2 model.
- Tokenizes the processed text for LLM training.
- Easy to deploy using Docker and Docker Compose.

## Requirements

- Docker
- Docker Compose
- A Hugging Face API key
- A Groq API key
- Access to the LLAMA2 model (local endpoint)

## Directory Structure


project_directory/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .env
├── pdf_to_llm_dataset.py
│
├── input/
│ └── book.pdf
│
└── output/
├── extracted_text.txt
├── dataset/
└── tokenized_dataset/

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kkkarmo/pdf-to-llm-dataset-converter.git
   cd pdf-to-llm-dataset-converter

Create and Configure the .env File
Create a .env file in the project root directory and add your API keys and configuration:

HUGGINGFACE_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
LLAMA_URL=http://20.20.20.26:11343/generate

Place Your PDF File
Place your PDF book in the input/ directory. Rename it to book.pdf or update the INPUT_FILE environment variable in the .env file accordingly.
Running the Project
Build and Run the Docker Container
Make sure you are in the project directory containing the docker-compose.yml file, then run:

bash
docker-compose up --build

Access Output Files
After the script completes, you can find the output files in the output/ directory. This includes:
extracted_text.txt: The raw extracted text from the PDF.
dataset/: The enriched dataset.
tokenized_dataset/: The tokenized dataset ready for LLM training.
Stopping the Container
To stop the running container, use:

bash
docker-compose down

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
Acknowledgements
pdfplumber for PDF text extraction.
Hugging Face Transformers for pre-trained NLP models.
LangChain for LLM integration.
Docker for containerization.
