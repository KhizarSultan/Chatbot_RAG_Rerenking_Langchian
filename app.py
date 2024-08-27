import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

st.set_page_config(page_title="Cole McIntosh RAG", page_icon="🌐")

# Load and process the website content
@st.cache_resource
def load_and_process_website():
    loader = WebBaseLoader("https://colemcintosh.io")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=756,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        user_agent="ColeMcintosh.io Q&A",
        cohere_api_key=st.secrets['COHERE_API_KEY']
    )
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Set up the Streamlit app
st.title("Reranked RAG over my website 🌐")
st.caption("Powered by [Cole McIntosh](https://colemcintosh.io), [Cohere Rerank](https://cohere.com/rerank), [Groq](https://groq.com), and [LangChain](https://langchain.com)")
st.info("""
Cohere Rerank enhances search quality by reordering documents based on their relevance to a query. Unlike traditional RAG, it uses a two-stage approach: initial retrieval followed by semantic reranking. This allows for better semantic understanding and improved handling of nuanced queries.
""", icon="🔍")

# Add a visual separator
st.markdown("---")

# Load the processed website content
vectorstore = load_and_process_website()

# Initialize ChatGroq model
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=st.secrets['GROQ_API_KEY'])

# Define the base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Set up the Cohere reranker
compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
    cohere_api_key=st.secrets['COHERE_API_KEY']
)
# Create the contextual compression retriever
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Load the prompt from LangChain Hub
rag_prompt = hub.pull("rlm/rag-prompt", api_key=st.secrets['LANGCHAIN_API_KEY'])

# Define the LCEL chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Create a form for user input
with st.form(key='question_form'):
    user_question = st.text_input("Ask a question about Cole")
    submit_button = st.form_submit_button(label='Submit')

if submit_button and user_question:
    # Get the answer
    answer = chain.invoke(user_question)
    
    # Display the answer
    st.subheader("Answer")
    st.write(answer)
    
    # Display the source documents in an expandable section
    with st.expander("View Sources"):
        docs = retriever.get_relevant_documents(user_question)
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content)
            st.markdown("---")