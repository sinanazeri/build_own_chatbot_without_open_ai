import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoModelForCausalLM

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []


# Function to initialize the language model and its embeddings
def init_llm():
    global llm, tokenizer, embeddings

    # Load models locally
    #tiiuae/falcon-7b-instruct
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b",trust_remote_code=True)

    #tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b",trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b",trust_remote_code=True)
    #google/flan-t5-xl
    #model = AutoModelForSeq2SeqLM.from_pretrained("tiiuae/falcon-7b",trust_remote_code=True
                                                #load_in_8bit=True,
                                            #    device_map='auto',
                                                #   torch_dtype=torch.float16,
                                            #    low_cpu_mem_usage=True)
    
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                          model_kwargs={"device": DEVICE}
                                                          )
    
    pipe = pipeline("text-generation", model=model, trust_remote_code=True,repetition_penalty=1.15,max_length=2048,tokenizer=tokenizer)

                                                  
    # pipe = pipeline(
    #     "text2text-generation",
    #     model=model, 
    #     tokenizer=tokenizer, 
    #     max_length=2048,
    #     temperature=0,
    #     top_p=0.95,
    #     repetition_penalty=1.15
    #     )

    llm = HuggingFacePipeline(pipeline=pipe)
    #return llm, tokenizer, embeddings

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)

    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'lambda_mult': 0.25})
    
    # Build the QA chain, which utilizes the model and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever= retriever,
        return_source_documents=False
    )

# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    # Query the model
    output = conversation_retrieval_chain(prompt)
    answer = output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer

# Initialize the language model
init_llm()
