This script will remove leading and trailing spaces, convert the filename to lowercase, replace spaces with underscores, and remove special characters and other junk things (except digits and hyphens) in one line. You can test it with the example provided or replace "My FiLe -WiTh 123 !@#$%^&.txt" with your desired filename.

import re

def clean_filename(filename):
    return re.sub(r'[^\w\d\-.]', '', filename.strip().lower().replace(" ", "_"))

# Example usage
original_filename = "  My FiLe -WiTh 123 !@#$%^&.txt  "
cleaned_filename = clean_filename(original_filename)
print("Original Filename:", original_filename)
print("Cleaned Filename:", cleaned_filename)




messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]


=================

# Install dependencies
!pip install huggingface_hub
!pip install chromadb
!pip install langchain
!pip install pypdf
!pip install sentence-transformers

# import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
# Load the pdf file and split it into smaller chunks
loader = PyPDFLoader('report.pdf')
documents = loader.load()

# Split the documents into smaller chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# We will use HuggingFace embeddings 
embeddings = HuggingFaceEmbeddings()
#Using Chroma vector database to store and retrieve embeddings of our text
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={'k': 2})
# We are using Mistral-7B for this question answering 
repo_id = "mistralai/Mistral-7B-v0.1"
llm = HuggingFaceHub(huggingfacehub_api_token='your huggingface access token here', 
                     repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})
# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)
#We will run an infinite loop to ask questions to LLM and retrieve answers untill the user wants to quit
import sys
chat_history = []
while True:
    query = input('Prompt: ')
    #To exit: use 'exit', 'quit', 'q', or Ctrl-D.",
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
=================================================
https://github.com/SonicWarrior1/pdfchat/blob/master/app.py
=================================================
=================================================
=================================================
=================================================
=================================================
=================================================
from fastapi import HTTPException, status
from datetime import datetime
import os
import shutil
from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/uploadfile")
def upload_file(session_id: str, author_nbkid: str, bucket_name: str, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No upload file sent")
    elif not file.filename.endswith("-pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed")
    else:
        try:
            # Get the file metadata
            file_metadata = {
                "author_nbkid": author_nbkid,
                "bucket_name": bucket_name,
                "filename": file.filename,
                "file_size": file.size,
                "Upload_datetine": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            
            # Save the file to the server
            model_data_load_path = get_setting("model_data_load_path")
            bucket_path = os.path.join(model_data_load_path, author_nbkid, bucket_name)
            if not os.path.exists(bucket_path):
                os.makedirs(bucket_path)
            file_path = os.path.join(bucket_path, file.filename)
            with open(file_path, "wb") as fp:
                shutil.copyfileobj(file.file, fp)
                
            return {"message": f"Successfully uploaded in bucket: {bucket_name}", "file_metadata": file_metadata}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))





import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class FileDeleteRequest(BaseModel):
    file_path: str

@app.delete("/delete_file/")
async def delete_file(request_data: FileDeleteRequest):
    file_path = request_data.file_path
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Attempt to delete the file
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
    
    return {"message": f"File {file_path} deleted successfully"}
    
    
    
=============

from fastapi import FastAPI, UploadFile, File
import PyPDF2

app = FastAPI()

def extract_pdf_metadata(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        metadata = {}
        metadata['title'] = pdf_reader.getDocumentInfo().title
        metadata['author'] = pdf_reader.getDocumentInfo().author
        metadata['subject'] = pdf_reader.getDocumentInfo().subject
        metadata['producer'] = pdf_reader.getDocumentInfo().producer
        metadata['creator'] = pdf_reader.getDocumentInfo().creator
        metadata['creation_date'] = pdf_reader.getDocumentInfo().creationDate
        metadata['num_pages'] = pdf_reader.numPages
    return metadata

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Check if the uploaded file is a PDF
    if file.filename.endswith('.pdf'):
        with open(file.filename, "wb") as buffer:
            # Write the contents of the uploaded file to the server
            buffer.write(await file.read())
        
        # Extract metadata from the uploaded PDF file
        metadata = extract_pdf_metadata(file.filename)
        
        return {"filename": file.filename, "metadata": metadata, "message": "File uploaded successfully"}
    else:
        return {"error": "Only PDF files are allowed"}



@app.post('/upload')
def upload_file(uploaded_file: UploadFile = File(...)):
    path = f"files/{uploaded_file.filename}"
    with open(path, 'w+b') as file:
        shutil.copyfileobj(uploaded_file.file, file)

    return {
        'file': uploaded_file.filename,
        'content': uploaded_file.content_type,
        'path': path,
    }

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Get the file metadata
    file_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": file.file_size,
    }

    # Save the file to the server
    with open(file.filename, "wb") as f:
        f.write(file.file)

    # Return the file metadata
    return file_metadata

# temp

import streamlit as st

from fileingestor import FileIngestor

# Set the title for the Streamlit app
st.title("Chat with PDF - 🦙 🔗")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

if uploaded_file:
    file_ingestor = FileIngestor(uploaded_file)
    file_ingestor.handlefileandingest()


from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_path = 'PATH_TO_/models/llama-2-7b-chat.Q4_K_M.gguf'


class Loadllm:
    @staticmethod
    def load_llm():
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # Prepare the LLM

        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=40,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
        )

        return llm

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from loadllm import Loadllm
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'


class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create a FAISS vector store and save embeddings
        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        # Load the language model
        llm = Loadllm.load_llm()

        # Create a conversational chain
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        # Function for conversational chat
        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + self.uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to PDF data 🧮", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
