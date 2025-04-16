from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI  # 需要用一个 LLM 来做摘要

def get_memory():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # 用于摘要的 LLM，可与主 LLM 分开
    return ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
