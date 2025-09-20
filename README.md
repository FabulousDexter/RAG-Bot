# RAG Bot - Advanced PDF Question Answering System

A powerful Retrieval-Augmented Generation (RAG) bot that allows you to ask questions about PDF, TXT, and Markdown documents using local AI models. This enhanced system features GPU acceleration, advanced filtering, and intelligent document processing with source attribution.

## 🚀 Enhanced Features

### **Core Capabilities:**
- **Multi-Format Document Processing**: Supports PDF, TXT, and Markdown files
- **GPU-Accelerated Embeddings**: Uses Ollama with GPU support for fast embedding generation
- **Advanced Vector Database**: ChromaDB with intelligent chunking and change detection
- **Smart Source Attribution**: Hierarchical chunk IDs with file, page, and content tracking
- **Incremental Updates**: Content hashing to detect and update only changed documents

### **Advanced Query Features:**
- **Document Filtering**: Filter by filename, folder path, or file type
- **Configurable Retrieval**: Adjust number of chunks retrieved (k parameter)
- **Detailed Source Display**: Show document sources, pages, and similarity scores
- **Organized Results**: Group results by source document for better context
- **Fallback Logic**: Intelligent handling when filters don't match any documents

### **Performance Optimizations:**
- **GPU Acceleration**: Automatic GPU detection and utilization via Ollama
- **Batch Processing**: Efficient chunk processing for large document sets  
- **Change Detection**: Skip processing of unchanged documents
- **Smart Chunking**: Optimized chunk sizes with content overlap for better context

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

## 📁 Enhanced Project Structure

```
RAG-Bot/
├── data/                          # Place your documents here (PDF, TXT, MD)
│   ├── CV/                        # Organized subfolder support
│   │   └── Dat-Tran-CV.pdf
│   ├── MA1RA1_2025_Lecture_Note.pdf
│   └── alice_in_wonderland.md
├── chroma/                        # Vector database storage (auto-created)
│   ├── chroma.sqlite3             # ChromaDB database file
│   └── [embedding files]         # Vector embeddings and metadata
├── venv/                          # Virtual environment (auto-created)
├── get_embedding_function.py      # GPU-accelerated embedding configuration  
├── populate_database.py           # Enhanced document processing with change detection
├── query_data.py                  # Advanced query interface with filtering
├── test_embedding.py              # Embedding functionality testing
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore configuration
└── README.md                      # This file
```

## 🚀 Advanced Usage

### Step 1: Add Your Documents
Place your documents in the `data/` directory. Supports:
- **PDF files**: Technical documents, research papers, manuals
- **Text files**: Notes, documentation, reports  
- **Markdown files**: Documentation, articles, README files
- **Organized folders**: Create subfolders for better organization

### Step 2: Populate the Database
Process and index your documents with enhanced change detection:
```bash
# Process new and changed documents
python populate_database.py

# Reset database and reprocess everything
python populate_database.py --reset
```

**Features:**
- ✅ **Content hashing** to detect file changes
- ✅ **Incremental processing** of only new/modified files
- ✅ **GPU acceleration** for faster embedding generation
- ✅ **Progress tracking** with detailed console output

### Step 3: Query with Advanced Filtering

#### **Basic Queries:**
```bash
# Query all documents
python query_data.py "What is the main topic discussed?"

# Get more context with additional chunks
python query_data.py "Explain the methodology" --k 10
```

#### **Filtered Queries:**
```bash
# Filter by document name/path
python query_data.py "What programming skills are mentioned?" --filter "CV"

# Filter by file type
python query_data.py "What are the research findings?" --file-type "PDF"

# Combine filters
python query_data.py "What topics are covered?" --filter "Lecture" --file-type "PDF" --k 8
```

#### **Detailed Analysis:**
```bash
# Show detailed source information
python query_data.py "Summarize the key points" --show-sources

# Advanced filtering with source details
python query_data.py "What skills are mentioned?" --filter "CV" --show-sources --k 10
```

#### **Example Advanced Output:**
```bash
📚 Using information from 2 document(s):
   - Dat-Tran-CV.pdf (3 chunks)
   - MA1RA1_2025_Lecture_Note.pdf (2 chunks)

🤖 Generating response using Mistral model...

💬 Response:
Based on the CV, the programming skills mentioned include Python, JavaScript, 
machine learning frameworks, and database management systems...

📎 Detailed Sources:
   1. data\CV\Dat-Tran-CV.pdf
      Type: PDF, Page: 0, Distance: 0.23
      Preview: Experience in Python development with frameworks including...

   2. data\CV\Dat-Tran-CV.pdf  
      Type: PDF, Page: 0, Distance: 0.31
      Preview: Technical skills: JavaScript, React, Node.js, MongoDB...
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

### Environment Configuration
The system uses Ollama for both embeddings and language model:

```python
# get_embedding_function.py
EMBEDDING_MODEL = "nomic-embed-text"  # High-quality embeddings
LLM_MODEL = "mistral"                 # Local language model
```

### Advanced Parameters

#### **Database Population (populate_database.py)**
- `--reset`: Clear existing database and reprocess all documents
- `--debug`: Enable verbose logging for troubleshooting
- Automatic GPU acceleration when available
- Content hashing for intelligent change detection

#### **Query Interface (query_data.py)**
```bash
python query_data.py "your question" [options]

Options:
  --k INTEGER         Number of relevant chunks to retrieve (default: 5)
  --filter TEXT       Filter documents by name/path substring
  --file-type TEXT    Filter by file type (PDF, TXT, MD)
  --show-sources      Display detailed source information
  --model TEXT        Override default LLM model
  --help              Show all available options
```

### Performance Tuning
- **GPU Acceleration**: Automatically detected and enabled for Ollama
- **Chunk Size**: Optimized at 1000 characters with 200 character overlap
- **Embedding Model**: `nomic-embed-text` provides excellent semantic understanding
- **Similarity Search**: ChromaDB's cosine similarity with configurable k value

### Document Processing
- **Supported Formats**: PDF, TXT, MD with automatic format detection
- **Change Detection**: SHA-256 hashing prevents unnecessary reprocessing
- **Error Handling**: Graceful handling of corrupted or inaccessible files
- **Progress Tracking**: Real-time feedback during processing

### Model Configuration
To change embedding model, edit `get_embedding_function.py`:
```python
embeddings = OllamaEmbeddings(model="your-preferred-embedding-model")
```

To change language model, edit `query_data.py`:
```python
model = Ollama(model="your-preferred-llm-model")
```

### Document Chunking
Adjust chunk size and overlap in `populate_database.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Optimized chunk size
    chunk_overlap=200,     # Balanced overlap
    length_function=len,
)
```

### Retrieval Settings
Change the number of similar chunks retrieved in `query_data.py`:
```python
results = db.similarity_search_with_score(query_text, k=5)  # Change k value
```

## 🔧 How It Works

1. **Document Loading**: Multi-format loaders handle PDF, TXT, and MD files from the `data/` directory
2. **Content Hashing**: SHA-256 hashing detects file changes to avoid unnecessary reprocessing
3. **Text Splitting**: Documents are split into 1000-character chunks with 200-character overlap for optimal context
4. **GPU-Accelerated Embedding**: Each chunk is converted to a 768-dimensional vector using `nomic-embed-text` with GPU acceleration
5. **Vector Storage**: Embeddings are stored in ChromaDB with rich metadata (source, page, chunk ID, file type)
6. **Advanced Query Processing**: User questions support filtering by document name, file type, and configurable retrieval parameters
7. **Smart Context Retrieval**: Configurable number of most similar chunks with custom filtering logic
8. **Enhanced Answer Generation**: Retrieved context is sent to `mistral` model with detailed source attribution and organized results

## 📊 Performance Features

- **🚀 GPU Acceleration**: Automatic GPU detection and utilization for embedding generation
- **⚡ Incremental Processing**: Only processes new or modified documents
- **🎯 Smart Filtering**: Advanced document filtering by name, path, and file type
- **📈 Scalable Architecture**: Handles large document collections efficiently
- **🔍 Rich Metadata**: Comprehensive source attribution with page numbers and relevance scores
- **💾 Persistent Storage**: ChromaDB ensures fast startup and query performance

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
