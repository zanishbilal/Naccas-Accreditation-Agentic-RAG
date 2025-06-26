# NACCAS Policy Assistant

A Streamlit-based chatbot application that helps users query NACCAS policies using retrieval-augmented generation (RAG) with LangGraph workflow.

## Features

- **Conversational Interface**: Chat-based interface for querying NACCAS policies
- **Document Upload**: Support for PDF and DOCX file uploads
- **Retrieval-Augmented Generation**: Uses Pinecone vector database for context retrieval
- **Multi-session Support**: Maintain multiple chat sessions
- **Public Access**: Integrated with ngrok for public URL access

## Project Structure

```
├── app.py              # Main Streamlit application
├── config.py           # Configuration and environment variables
├── tools.py            # Pinecone retriever and tools setup
├── utils.py            # Utility functions
├── workflow.py         # LangGraph workflow definition
├── run.py              # App runner with ngrok integration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md          # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd naccas-policy-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   PINECONE_API_KEY=your_pinecone_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   NGROK_AUTH_TOKEN=your_ngrok_auth_token_here
   ```

### 4. Configure Your Pinecone Index

Make sure you have a Pinecone index named `clara-qwen2` with the appropriate embeddings loaded. Update the index name in `config.py` if different.

## Usage

### Option 1: Run with ngrok (Recommended)

```bash
python run.py
```

This will:
- Set up ngrok tunnel
- Start the Streamlit app
- Provide a public URL for access

### Option 2: Run Locally

```bash
streamlit run app.py
```

This will run the app locally at `http://localhost:8501`

## Configuration

You can modify the following settings in `config.py`:

- **Model Settings**: Change the embedding model or response model
- **Pinecone Settings**: Update index name or retrieval parameters
- **Generation Parameters**: Adjust temperature, max tokens, etc.

## File Descriptions

- **`app.py`**: Main Streamlit application with UI and chat logic
- **`config.py`**: Centralized configuration management
- **`tools.py`**: Pinecone vector store and retriever setup
- **`utils.py`**: Helper functions for chat management and file processing
- **`workflow.py`**: LangGraph workflow with RAG pipeline
- **`run.py`**: Application runner with ngrok integration

## Dependencies

Key dependencies include:
- `streamlit`: Web application framework
- `langchain`: LLM framework and tools
- `langgraph`: Workflow orchestration
- `pinecone-client`: Vector database client
- `groq`: LLM API client
- `transformers`: Hugging Face transformers
- `pyngrok`: ngrok Python wrapper

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors, you may need to:
1. Install appropriate PyTorch version for your system
2. Modify the device setting in `tools.py` from "cuda" to "cpu"

### API Key Issues
Make sure all API keys are properly set in your `.env` file and that you have:
- Valid Pinecone API key with access to your index
- Valid Groq API key
- Valid ngrok auth token (if using public access)

### Memory Issues
If you experience memory issues, try:
- Reducing the embedding model size
- Lowering the `max_tokens` parameter
- Using CPU instead of GPU for embeddings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
to be decided
