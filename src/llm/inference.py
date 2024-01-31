from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from src.vectordb.qdrant import VectorDBQdrant
from env import EnvHelper
from llama_index.retrievers import BaseRetriever
from llama_index.vector_stores import VectorStoreQuery
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from src.embedder.multilingual_e5_large import MultilingualE5LargeEmbedder
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.llms import ChatMessage, MessageRole, AzureOpenAI
from llama_index import ServiceContext
from llama_index import get_response_synthesizer
import llama_index
from llama_index.embeddings.base import BaseEmbedding
from src.llm.prompts import condense_question, text_qa_template

llama_index.set_global_handler('simple')

class KiCampusRetriever(BaseRetriever):
    def __init__(self, embedder:BaseEmbedding=None):
        if embedder is None:
            embedder = MultilingualE5LargeEmbedder()
        self.embedder = embedder
        self.vector_store = VectorDBQdrant('disk').as_llama_vector_store(collection_name='assistant')
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = self.embedder.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=3)
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            # how node gets rendered as context for the llm
            # alternatively create a custom node post-processor and pass to query engine https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html
            node.text_template = '{metadata_str}\nContent: {content}'

            score: list[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

class KICampusAssistant():
    def __init__(self, verbose:bool=False):
        self.retriever = KiCampusRetriever()

        secrets = EnvHelper()
        llm = AzureOpenAI(
            model='gpt-35-turbo',
            deployment_name='gpt-3_5',
            api_key=secrets.AZURE_OPENAI_KEY,
            azure_endpoint=secrets.AZURE_OPENAI_URL,
            api_version='2023-05-15',
        )

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=None)
        response_synthesizer = get_response_synthesizer(service_context=service_context, 
                                                        response_mode='compact', text_qa_template=text_qa_template)
        query_engine = RetrieverQueryEngine(retriever=self.retriever, response_synthesizer=response_synthesizer)
        
        if verbose:
            # the refine prompt templater is only used when compact prompting is too long for the llm
            for k, p in query_engine.get_prompts().items():
                print(f'\n**Prompt Key**: {k}\n**Text:**:')
                print(p.get_template())
            print('*** Prompt Examples End ***')

        # Wrap query_engine: Fits the chat history and the new query into a single query containing all previous context
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=condense_question,
            verbose=True,
            service_context=service_context,
            chat_history=None,
        )

    def chat(self, query: str, chat_history: list[ChatMessage] = None) -> str:
        response = self.chat_engine.chat(query, chat_history=chat_history)
        print(response)
        return response

    def reset(self):
        self.chat_engine.reset()

if __name__ == '__main__':
    assistant = KICampusAssistant(verbose=True)
    assistant.answer(query='Welche Kurse zu ethischer KI habt ihr im Angebot?')
    print('wait')
