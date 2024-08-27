# Reranked RAG over My Personal Website

This Streamlit app showcases a Retrieval-Augmented Generation (RAG) system with reranking, designed to answer questions about me and my work based on the content of my personal website.

## Project Overview

I built this app to demonstrate:
1. The power of RAG systems for information retrieval and question answering
2. The effectiveness of Cohere's reranking in improving retrieval accuracy
3. The integration of various AI technologies in a practical application

## Key Components

- **Web Scraping**: Automatically extracts content from [colemcintosh.io](https://colemcintosh.io)
- **RAG System**: Implemented using LangChain for flexible and powerful retrieval
- **Reranking**: Utilizes Cohere Rerank to enhance retrieval precision
- **LLM Integration**: Leverages Groq's LLM for generating human-like responses
- **Vector Storage**: Uses FAISS for efficient similarity search
- **User Interface**: Built with Streamlit for an interactive Q&A experience

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Cohere Rerank](https://cohere.com/rerank)
- [Groq](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up a `.env` file with necessary API keys (Cohere, Groq)
4. Run the app: `streamlit run app.py`

## How It Works

Users can ask questions about me or my work in the text input field. The app then:
1. Retrieves relevant information from the vectorized content of my website
2. Reranks the retrieved documents for improved relevance
3. Generates a response using Llama 3.1 8B  from Groq
4. Displays the answer and source documents

Feel free to reach out if you have any questions or suggestions!