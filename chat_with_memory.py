from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given the chat history and the latest user question,
reformulate the question to be standalone – meaning it can be understood
without the chat history. Do NOT answer the question, just reformulate
it if needed. If it's already standalone, return it as is.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer based on the provided context. Cite sources
using [Source N] notation. If the context is insufficient, say so.

Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


class RAGChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.chat_history = []

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            collection_name="rag_chatbot",
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )
        self.retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
        )

    def _contextualize_question(self, question: str) -> str:
        """Reformulate question using chat history."""
        if not self.chat_history:
            return question

        chain = CONTEXTUALIZE_PROMPT | self.fast_llm | StrOutputParser()

        return chain.invoke({"chat_history": self.chat_history, "question": question})

    def ask(self, question: str) -> str:
        """Process a question through the RAG pipeline"""
        standalon_q = self._contextualize_question(question)

        docs = self.retriever.invoke(standalon_q)
        context = "\n\n---\n\n".join(
            f"[Source {i}] {d.page_content}" for i, d in enumerate(docs, 1)
        )

        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": context,
                "chat_history": self.chat_history,
                "question": question,
            }
        )

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return answer
