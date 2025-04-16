from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from memory.memory import get_memory
from retrieval.rag_engine import load_corp_files, load_file_to_vectorstore  # 修改点
from database.storage import save_message, load_history
import gradio as gr
import os
from config import OPENAI_API_KEY, DB_PATH
from langchain_core.messages import AIMessage

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
memory = get_memory()

retriever = None
qa_chain = None
uploaded_files = []


def update_vectorstore():
    global retriever, qa_chain

    print("[INFO] 加载企业知识库...")
    base_vs = load_corp_files()

    docs = [base_vs]
    for f in uploaded_files:
        print(f"[INFO] 加载用户文件: {f.name}")
        file_vs = load_file_to_vectorstore(f.name)
        docs.append(file_vs)

    merged_vs = docs[0]
    for other in docs[1:]:
        merged_vs.merge_from(other)

    retriever = merged_vs.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


update_vectorstore()


def chatbot_interface(message, file):
    global uploaded_files

    if file is not None:
        if not isinstance(file, list):
            file = [file]
        uploaded_files.extend(file)
        update_vectorstore()
        return f"📎 共加载 {len(uploaded_files)} 个文件（含企业知识库）"

    if qa_chain:
        result = qa_chain.invoke({"question": message})
        answer = result["answer"]
    else:
        result = llm.invoke(message)
        answer = result.content if isinstance(result, AIMessage) else str(result)

    save_message(message, answer)
    return answer


chat_history = []

with gr.Blocks(css=".chatbox {height: 500px; overflow-y: auto;}") as demo:
    gr.Markdown("## 🤖 企业知识库问答 + 文件 Q&A")

    with gr.Row():
        with gr.Column(scale=2):
            history_output = gr.Textbox(label="📜 最近历史记录", lines=25, interactive=False)
            history_btn = gr.Button("🕘 查看历史记录", scale=1)

        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="对话记录", elem_classes="chatbox", type="messages")
            with gr.Row():
                file_input = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="📎 上传文件")
            with gr.Row():
                message = gr.Textbox(placeholder="输入你的问题...", lines=1, scale=8, show_label=False)
                submit_btn = gr.Button("🚀 发送")

    def chat_handler(msg, files):
        global chat_history
        file_msg = ""

        if files is not None:
            result = chatbot_interface("", files)
            file_msg = result

        if msg.strip() != "":
            result = chatbot_interface(msg, None)
            chat_history.append({"role": "user", "content": msg})
            chat_history.append({"role": "assistant", "content": result})
            return "", chat_history

        if file_msg:
            chat_history.append({"role": "user", "content": "📎 上传文件"})
            chat_history.append({"role": "assistant", "content": file_msg})
            return "", chat_history

        return "", chat_history

    submit_btn.click(fn=chat_handler, inputs=[message, file_input], outputs=[message, chatbot])
    history_btn.click(fn=load_history, outputs=history_output)

if __name__ == "__main__":
    demo.launch()
