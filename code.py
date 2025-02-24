from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from config import embedding_name,model_name,prompt
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os

embed_name = embedding_name

pdfs_directory = 'utils/temp/'

#house keeping functions
def create_folder_if_not_exists(folder_path):
    """
    Creates a folder if it does not exist.

    Args:
        folder_path: The path to the folder to create.
    """
    os.makedirs(folder_path, exist_ok=True)


def extract_text(text: str) -> str:
    " we have output with resaoning text and this function will remove that"
    index = text.find("</think>")
    if index != -1:
        return text[index + len("</think>"):].lstrip()  # lstrip() avoids extra space scanning
    return ""


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def pdfloader(folderpath):
    # loader = PyMuPDFLoader("/Users/andraju/Desktop/expr/MAD/pdfexp/pdf loader/glue-dg.pdf")
    loader = PyMuPDFLoader(folderpath)
    data = loader.load()
    return data



#splitter
def get_text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=300,
    )

    return  text_splitter.split_documents(docs)

#vector store
def vector_store_setup(docs_split):
    embeddings = OllamaEmbeddings(model=embed_name)
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=docs_split)
    return vector_store

# Formatting documents for RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#RAG Chaing
def rag_chain(retriver_docs):
    model = OllamaLLM(model=model_name)
    prompt_template = ChatPromptTemplate.from_template(prompt)
    return (
            {"context": retriver_docs | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
    )




uploaded_file = st.file_uploader(
    "Upload PDF",
    type="PDF",
    accept_multiple_files=False
)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#upload file
if uploaded_file:
    create_folder_if_not_exists(pdfs_directory)
    upload_pdf(uploaded_file)
    documents = pdfloader(pdfs_directory + uploaded_file.name)
    chunked_documents = get_text_splitter(documents)
    vector_store = vector_store_setup(chunked_documents)

    question = st.chat_input("Ask your questions....")


    if question:
        # Append user's question to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

        # Get model response
        rag_chain_output = rag_chain(retriever)

        # Ensure output is gathered correctly
        output = ''.join(rag_chain_output.stream(question))  # Output must be a string
        if output:  # Ensure that the output is not empty
            st.session_state.chat_history.append({"role": "assistant", "content": output})
        else:
            st.error("No response generated from the assistant.")

# Display full chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

