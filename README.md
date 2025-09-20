# RAG Bot - PDF Question Answering System

A Retrieval-Augmented Generation (RAG) bot that allows you to ask questions about PDF documents using local AI models. This system uses Ollama for embeddings and language models, ChromaDB for vector storage, and LangChain for document processing.

## 🚀 Features

- **PDF Document Processing**: Automatically loads and processes PDF files from a directory
- **Local AI Models**: Uses Ollama for both embeddings (nomic-embed-text) and language generation (mistral)
- **Vector Database**: Stores document embeddings in ChromaDB for efficient similarity search
- **Smart Chunking**: Splits documents into optimal chunks with overlap for better context
- **Source Tracking**: Returns source references with each answer (file, page, chunk)
- **Incremental Updates**: Only processes new documents, avoiding duplicate work

## 📺 Tutorial

This project is based on the excellent tutorial by Pixegami:
**[RAG Tutorial (with Local LLMs): AI For Your PDFs](https://www.youtube.com/watch?v=2TJxpyO3ei4&t=1060s)**

The tutorial provides a comprehensive walkthrough of building a RAG system from scratch, including:
- Setting up Ollama with local models
- Document processing and chunking strategies
- Vector database implementation with ChromaDB
- Query processing and answer generation

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models: `nomic-embed-text` and `mistral`

## 🛠️ Installation

1. **Clone or download this repository**

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama models:**
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   ```

## 📁 Project Structure

```
RAG-Bot/
├── data/                          # Place your PDF files here
│   └── MA1RA1_2025_Lecture_Note.pdf
├── chroma/                        # Vector database storage (auto-created)
├── get_embedding_function.py      # Embedding configuration
├── populate_database.py           # Document processing and indexing
├── query_data.py                  # Query interface
├── test_embedding.py              # Test embedding functionality
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Usage

### Step 1: Add Your PDF Documents
Place your PDF files in the `data/` directory.

### Step 2: Populate the Database
Process and index your documents:
```bash
python populate_database.py
```

To reset the database and reprocess all documents:
```bash
python populate_database.py --reset
```

### Step 3: Ask Questions
Query your documents:
```bash
python query_data.py "What is the main topic of the lecture?"
```

Example output:
```
====================================================================================================
Response: The main topic of the lecture is machine learning fundamentals, covering supervised and unsupervised learning algorithms.
====================================================================================================
Sources: ['MA1RA1_2025_Lecture_Note.pdf:1:0', 'MA1RA1_2025_Lecture_Note.pdf:1:1', 'MA1RA1_2025_Lecture_Note.pdf:2:0']
```

## 🧪 Testing

Test the embedding function:
```bash
python test_embedding.py
```

Expected output:
```
Embedding functions works! Vector dimension: 768
```

## ⚙️ Configuration

### Embedding Model
The system uses `nomic-embed-text` for embeddings. To change the model, edit `get_embedding_function.py`:
```python
embeddings = OllamaEmbeddings(model="your-preferred-embedding-model")
```

### Language Model
The system uses `mistral` for text generation. To change the model, edit `query_data.py`:
```python
model = Ollama(model="your-preferred-llm-model")
```

### Document Chunking
Adjust chunk size and overlap in `populate_database.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Adjust chunk size
    chunk_overlap=80,      # Adjust overlap
    length_function=len,
    is_separator_regex=False
)
```

### Retrieval Settings
Change the number of similar chunks retrieved in `query_data.py`:
```python
results = db.similarity_search_with_score(query_text, k=5)  # Change k value
```

## 🔧 How It Works

1. **Document Loading**: `PyPDFDirectoryLoader` loads all PDF files from the `data/` directory
2. **Text Splitting**: Documents are split into 800-character chunks with 80-character overlap
3. **Embedding Generation**: Each chunk is converted to a 768-dimensional vector using `nomic-embed-text`
4. **Vector Storage**: Embeddings are stored in ChromaDB with metadata (source, page, chunk ID)
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Context Retrieval**: Top 5 most similar chunks are retrieved
7. **Answer Generation**: Retrieved context is sent to `mistral` model to generate an answer

## 📦 Dependencies

- **pypdf**: PDF document processing
- **langchain**: LLM application framework
- **chromadb**: Vector database for embeddings
- **pytest**: Testing framework
- **boto3**: AWS SDK (for future cloud integrations)

## 🔍 Troubleshooting

### Common Issues

**ImportError with LangChain:**
- Update to newer packages: `pip install langchain-ollama langchain-chroma`

**Ollama connection errors:**
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`

**No results returned:**
- Check if documents were processed: Look for files in `chroma/` directory
- Verify Ollama models are working: `ollama run mistral "Hello"`

**Performance issues:**
- Reduce chunk size or number of retrieved chunks
- Use a smaller/faster language model

## 🚀 Future Enhancements

- [ ] Web interface for easier interaction
- [ ] Support for more document formats (Word, TXT, etc.)
- [ ] Multi-language support
- [ ] Integration with cloud-based LLMs
- [ ] Advanced filtering and search capabilities
- [ ] Conversation memory for follow-up questions

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
