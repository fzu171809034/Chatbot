from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.file_loader import load_file
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader

def load_file_to_vectorstore(file_path):
    pages = load_file(file_path)
    # print("[DEBUG] 文档页数：", len(pages))
    # print("[DEBUG] 第1页内容预览：", pages[0].page_content[:300])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)
    # print("[DEBUG] FAISS 构建完成")
    return db

def load_corp_files():
    corpus_dir = "company_knowledge"  # 知识库目录，自行创建
    all_docs = []

    for file in Path(corpus_dir).glob("*.txt"):
        loader = UnstructuredFileLoader(str(file))
        docs = loader.load()
        all_docs.extend(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(all_docs, embeddings)
    return db