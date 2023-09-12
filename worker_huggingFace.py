import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings
    # Set up the environment variable for HuggingFace and initialize the desired model.
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your HuggingFace API"

    # Insert the name of repo model
    model_id = "tiiuae/falcon-7b-instruct"
    
    # load the model into the HuggingFaceHub
    llm_hub = # specify hugging face hub object with (repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})
    

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = # create object of Hugging Face Instruct Embeddings with (model_name,  model_kwargs={"device": DEVICE} )
    

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain
    # Load the document
    loader =   # ---> use PyPDFLoader and document_path from the function input parameter <---
    
    documents = loader.load()
    # Split the document into chunks, set chunk_size=1024, and chunk_overlap=64. assign it to variable text_splitter
    text_splitter = # ---> use Recursive Character TextSplitter and specify the input parameters <---
    
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)
    
    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever= db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False
    )


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    # Pass the prompt and the chat history to the conversation_retrieval_chain object
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    
    answer =  output["result"]
    
    # Update the chat history
    # TODO: Append the prompt and the bot's response to the chat history using chat_history.append and pass `prompt` `answer` as arguments
    # --> write your code here <--	
    
    # Return the model's response
    return result['answer']
    

# Initialize the language model
init_llm()
