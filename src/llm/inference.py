from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant
from src.helpers import get_secrets
from llama_index.retrievers import BaseRetriever
from llama_index.vector_stores import VectorStoreQuery
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.prompts import PromptTemplate
from llama_index.llms import ChatMessage, MessageRole, AzureOpenAI
from llama_index import ServiceContext
from llama_index import get_response_synthesizer
import llama_index

llama_index.set_global_handler("simple")

condense_question_prompt = PromptTemplate("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

# TODO where to override the system prompt for chat engine?
system_prompt = '''You are a chat bot on our website ki-campus.org helping and answering its user. We are a learning platform for artificial intelligence with free online courses, videos and podcasts in various topics of AI and data literacy. \
As an research and development project, the AI Campus is funded by the German Federal Ministry of Education and Research (BMBF). \
You audience speaks german or english, answer in the language the user asked the question in.

You will be provided with a pre-selection of context to answer the question. You don't need to use all of it, choose what you think is relevant. 
Dont add any additional information that is not in the context. Don't make up an answer! If you don't know the answer, just say that you don't know. 

Keep your answers short and simple, you are a chat bot!'''

class KiCampusRetriever(BaseRetriever):
    def __init__(self, embedder: MultilingualE5LargeEmbedder):    
        self.embedder = embedder
        self.vector_store = VectorDBQdrant('disk').as_llama_vector_store(collection_name="assistant")
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = self.embedder.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=3)
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: list[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


class KICampusAssistant():
    def __init__(self):
        self.embedder = MultilingualE5LargeEmbedder()
        self.retriever = KiCampusRetriever(self.embedder)

        #print(retriever.retrieve("what do you offer about ethical AI?"))

        secrets = get_secrets()
        llm = AzureOpenAI(
            model="gpt-35-turbo",
            deployment_name="gpt-3_5",
            api_key=secrets['AZURE']['OPENAI_KEY'],
            azure_endpoint=secrets['AZURE']['OPENAI_ENDPOINT'],
            api_version='2023-05-15',
        )
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=self.embedder)

        response_synthesizer = get_response_synthesizer(service_context=service_context, response_mode='compact')

        query_engine = RetrieverQueryEngine(retriever=self.retriever, response_synthesizer=response_synthesizer)
        
        # Example list of `ChatMessage` objects
        chat_history = [
            ChatMessage(role=MessageRole.USER, content="Hello assistant."),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hey there."),
        ]
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=condense_question_prompt,
            verbose=True,
            service_context=service_context,
            chat_history=chat_history,
        )

    def answer(self, query: str) -> str:
        response = self.chat_engine.chat(query)
        print(response)
        return response

    def reset(self):
        self.chat_engine.reset()

if __name__ == '__main__':
    assistant = KICampusAssistant()
    assistant.answer(query='Welche Kurse zu ethischer KI habt ihr im Angebot?')