from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()
def get_pdf_text(pdf_docs):
    print("Extracting text from PDF files...")
    text = ""
    for pdf in pdf_docs : 
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def create_chunks(text, chunk_size=1000, overlap=200):
    print("Creating chunks from text...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks 


def get_vectores(text_chunks):
    print("test embeddings for chunks...")
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token : 
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    print(f"Using HuggingFace API key: {token[:4]}...")  

    embeddings = HuggingFaceEndpointEmbeddings(   
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=token
    )
    vectorestore = FAISS.from_texts(text_chunks, embeddings)
    vectorestore.save_local("faiss_index")
    print("Vector store saved locally.", vectorestore)
    return vectorestore


def get_conversation_chain(vectorstore):
    print("Creating conversation chain...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    # Pour chaque question tu chercher les chunks plus proches sematiquement 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Vous êtes un assistant qui répond aux questions basées sur le contexte fourni.
        Utilisez les informations suivantes pour répondre à la question.
        Si vous ne trouvez pas la réponse dans le contexte, dites-le clairement.
        
        Contexte: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}")
    ])
    print('Conversation chain created successfully.', retriever)
    def format_doc(docs) : 
        return "\n\n".join([doc.page_content for doc in docs])
    chain = (
        {
            "context" : RunnableLambda(lambda x : x["question"]) | retriever | format_doc,
            "question" : RunnableLambda(lambda x : x["question"]), 
            "chat_history" : RunnableLambda(lambda x : x.get('chat_history', []))
        } | prompt | llm
        | StrOutputParser()
    )
    print("Conversation chain created successfully.", chain)
    return chain