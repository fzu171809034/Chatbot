from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_file(file_path):
    ext = os.path.splitext(file_path)[-1]
    if ext == ".pdf":
        return PyPDFLoader(file_path).load_and_split()
    elif ext == ".txt":
        return TextLoader(file_path).load_and_split()
    else:
        raise ValueError("Unsupported file type")
