from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from PyPDF2 import PdfReader
import re
import io
import shutil
import json
from typing import List, Union


def create_vector_store_from_jsonl(jsonl_input: Union[str, io.BytesIO, object], collection_name: str = "document_collection"):
    """
    Crée ou met à jour un vector store à partir d'un fichier JSONL.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = f"./database/{collection_name}"

    # Vérifier/créer le dossier database
    os.makedirs("./database", exist_ok=True)
    
    def read_lines(input_obj) -> List[str]:
        if hasattr(input_obj, "getvalue"):
            raw = input_obj.getvalue()
            if isinstance(raw, bytes):
                text = raw.decode("utf-8")
            else:
                text = str(raw)
            return text.splitlines()
        if isinstance(input_obj, (str, os.PathLike)):
            with open(input_obj, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        try:
            data = input_obj.read()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return data.splitlines()
        except Exception as e:
            print(f"Erreur lecture fichier: {e}")
            return []

    print(f"Lecture du fichier JSONL: {jsonl_input}")
    lines = read_lines(jsonl_input)
    print(f"Nombre de lignes lues: {len(lines)}")
    
    source_name = getattr(jsonl_input, "name", None) or (os.path.basename(str(jsonl_input)) if isinstance(jsonl_input, (str, os.PathLike)) else "uploaded.jsonl")

    documents: List[Document] = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Extraction du contenu
            content = None
            # Chercher d'abord dans question/réponse
            q = obj.get("question", "")
            a = obj.get("answer", "")
            if q and a:
                content = f"Question: {q}\nRéponse: {a}"
            
            if not content:
                for key in ("content", "text", "body"):
                    if key in obj and obj[key]:
                        content = obj[key]
                        break

            if content:
                metadata = {"source": source_name, "line": idx + 1}
                documents.append(Document(page_content=content, metadata=metadata))
                
        except json.JSONDecodeError as e:
            print(f"Erreur JSON ligne {idx}: {e}")
            continue
        except Exception as e:
            print(f"Erreur traitement ligne {idx}: {e}")
            continue

    print(f"Documents créés: {len(documents)}")
    
    if not documents:
        print("Aucun document créé!")
        return None

    print(f"Création vector store dans: {db_location}")
    vector_store = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        persist_directory=db_location,
        embedding=embeddings
    )

    class RetrieverWrapper:
        def __init__(self, vector_store):
            self._vector_store = vector_store

        def get_relevant_documents(self, query: str) -> List[Document]:
            return self._vector_store.similarity_search(query, k=100)  # augmenter k

    retriever = RetrieverWrapper(vector_store)
    return retriever



def create_vector_store_from_pdf(pdf_file, collection_name="document_collection"):
    """
    Create or update a vector store from an uploaded PDF file.
    
    Args:
        pdf_file: The uploaded PDF file from Streamlit
        collection_name: Name of the collection to store the vectors
        
    Returns:
        retriever: A retriever object that can be used for similarity search
    """
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Create a unique database location based on collection name
    db_location = f"./database/{collection_name}"
    
    def extract_text_from_pdf(pdf_file):
        """Extract text from PDF and split into meaningful chunks."""
        # Support Streamlit UploadedFile from st.chat_input(accept_file=True)
        if hasattr(pdf_file, "getvalue"):
            file_stream = io.BytesIO(pdf_file.getvalue())
            source_name = getattr(pdf_file, "name", "uploaded.pdf")
        else:
            file_stream = pdf_file
            source_name = getattr(pdf_file, "name", str(pdf_file))

        reader = PdfReader(file_stream)
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # Split text into smaller chunks (e.g., by sections or paragraphs)
            chunks = re.split(r'\n\s*\n', text or "")
            
            for chunk_num, chunk in enumerate(chunks):
                if chunk.strip():  # Only process non-empty chunks
                    doc = Document(
                        page_content=chunk.strip(),
                        metadata={
                            "source": source_name,
                            "page": page_num + 1,
                            "chunk": chunk_num + 1
                        }
                    )
                    documents.append(doc)
        
        return documents

    # Always rebuild the collection for a new uploaded file to avoid stale data
    if os.path.exists(db_location):
        shutil.rmtree(db_location, ignore_errors=True)

    # Extract documents from PDF
    documents = extract_text_from_pdf(pdf_file)

    # Create vector store with the documents
    vector_store = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        persist_directory=db_location,
        embedding=embeddings
    )

    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )
    
    return retriever