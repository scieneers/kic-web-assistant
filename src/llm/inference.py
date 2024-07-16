from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import QueryBundle, Settings, get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from src.env import EnvHelper
from src.LlamaIndex.qdrant import VectorDBQdrant
from src.llm.prompts import condense_question, system, text_qa_template

Settings.global_handler = "simple"


class KiCampusRetriever(BaseRetriever):
    def __init__(self, embedder: BaseEmbedding = None):
        secrets = EnvHelper()

        self.embedder = AzureOpenAIEmbedding(
            model=secrets.AZURE_OPENAI_EMBEDDER_MODEL,
            deployment_name=secrets.AZURE_OPENAI_EMBEDDER_DEPLOYMENT,
            api_key=secrets.AZURE_OPENAI_EMBEDDER_API_KEY,
            azure_endpoint=secrets.AZURE_OPENAI_EMBEDDER_ENDPOINT,
            api_version=secrets.AZURE_OPENAI_EMBEDDER_API_VERSION,
        )

        self.vector_store = VectorDBQdrant("remote").as_llama_vector_store(collection_name="web_assistant")
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        embedding = self.embedder.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=3)
        query_result = self.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            # how node gets rendered as context for the llm
            # alternatively create a custom node post-processor and pass to query engine https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html
            node.text_template = "{metadata_str}\nContent: {content}"

            score: list[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


class KICampusAssistant:
    def __init__(self, verbose: bool = False):
        self.retriever = KiCampusRetriever()
        secrets = EnvHelper()

        self.langfuse = Langfuse(
            secret_key=secrets.LANGFUSE_SECRET_KEY, public_key=secrets.LANGFUSE_PUBLIC_KEY, host=secrets.LANGFUSE_HOST
        )
        self.langfuse_callback_handler = LlamaIndexCallbackHandler(
            secret_key=secrets.LANGFUSE_SECRET_KEY, public_key=secrets.LANGFUSE_PUBLIC_KEY
        )

        self.llm_gpt = AzureOpenAI(
            temperature=0.0,
            model=secrets.AZURE_OPENAI_GPT4_MODEL,
            api_version=secrets.AZURE_OPENAI_GPT4_API_VERSION,
            engine=secrets.AZURE_OPENAI_GPT4_DEPLOYMENT,
            api_key=secrets.AZURE_OPENAI_GPT4_KEY,
            azure_endpoint=secrets.AZURE_OPENAI_GPT4_URL,
        )

        print("CHAT MODEL IS READY -------------------------------------------------------------------------")

        Settings.llm = self.llm_gpt
        Settings.embed_model = None

        response_synthesizer = get_response_synthesizer(
            response_mode="compact", text_qa_template=text_qa_template, verbose=True
        )
        query_engine = RetrieverQueryEngine(retriever=self.retriever, response_synthesizer=response_synthesizer)

        if verbose:
            # the refine prompt templater is only used when compact prompting is too long for the llm
            for k, p in query_engine.get_prompts().items():
                print(f"\n**Prompt Key**: {k}\n**Text:**:")
                print(p.get_template())
            print("*** Prompt Examples End ***")

        # Wrap query_engine: Fits the chat history and the new query into a single query containing all previous context
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=condense_question,
            verbose=True,
            chat_history=None,
        )

    def chat(self, query: str, chat_history: list[ChatMessage] = None) -> str:
        root_trace = self.langfuse.trace(name="llamaindex-rag-chat")
        trace_id = root_trace.trace_id
        self.langfuse_callback_handler.set_root(root_trace)

        response = self.chat_engine.chat(query, chat_history=chat_history)

        return response, trace_id

    def submit_feedback(self, feedback_response: dict, trace_id: str):
        value = 1 if feedback_response["score"] == "👍" else -1
        comment = feedback_response["text"]

        self.langfuse.score(trace_id=trace_id, name="user-explicit-feedback", value=value, comment=comment)

    def reset(self):
        self.chat_engine.reset()


if __name__ == "__main__":
    assistant = KICampusAssistant(verbose=True)
    assistant.chat(query="Eklär über den Kurs Deep Learning mit Tensorflow, Keras und Tensorflow.js")
