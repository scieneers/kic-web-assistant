import os

import torch.nn.functional as F
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class MultilingualE5LargeEmbedder(BaseEmbedding):
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModel = PrivateAttr()

    def __init__(self, **kwargs):
        # tokenizer is not thread-safe, chainlit uses multiple threads
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self._tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        self._model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

        super().__init__(**kwargs)

    def tokenize(self, input_text: str) -> dict[str, Tensor]:
        token_dict = self._tokenizer(input_text, max_length=513, padding=True, truncation=True, return_tensors="pt")

        token_len = token_dict["input_ids"].shape[1]
        if token_len > 512:
            raise ValueError(f"input text length of {len(input_text)} exceeds max length of 512 tokens.")
        return token_dict

    def embed(self, input_text: str, type: str = "query") -> list[float]:
        """type can be either 'query' or 'passage' and will result in different embeddings.
        When used for anything other than retrieval, you can simply only use the 'query' prefix."""

        def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            """How much was each token influenced by the others + normalize by number of real tokens from attention_mask"""
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        if type == "query":
            input_text = f"query: {input_text}"
        elif type == "passage":
            input_text = f"passage: {input_text}"
        else:
            raise ValueError(f"Unknown value for 'type': {type}")

        tokens = self.tokenize(input_text)
        outputs = self._model(**tokens)
        embeddings = average_pool(outputs.last_hidden_state, tokens["attention_mask"])
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        return embeddings_norm.tolist()[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        """Implementation for llama_index wrapper"""
        return self.embed(query, type="query")

    def _aget_query_embedding(self, query: str) -> list[float]:
        """Implementation for llama_index wrapper"""
        return self.embed(query, type="query")

    def _get_text_embedding(self, text: str) -> list[float]:
        """Implementation for llama_index wrapper"""
        return self.embed(text, type="passage")


if __name__ == "__main__":
    # Each input text should start with 'query: ' or 'passage: ', even for non-English texts.
    # For tasks other than retrieval, you can simply use the 'query: ' prefix.
    input_texts = [
        "how much protein should a female eat",
        "南瓜的家常做法",
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅",
    ]

    embedder = MultilingualE5LargeEmbedder()
    # embeddings = [embedder.embed(text) for text in input_texts]
    # print(embeddings)

    input_texts = "<h2>KI-Lectures: Lernen und Bildung mit KI</h2>\n\n<p>eine Veranstaltungsreihe vom <em>AI.EDU Research Lab</em> des Forschungsschwerpunkts D²L² der FernUniversität in Hagen und dem Projekt <em>tech4comp</em> mit der Universität Leipzig und der TU Dresden, zusammen mit dem DFKI, die im Sommersemester 2021 auf dem KI-Campus stattfindet.</p>\n\n<p><strong>Inhalt</strong>  </p>\n\n<p>“<em>AIEd is […] a powerful tool to open up what is sometimes called the ‘black box of learning,’ giving us more fine-grained understandings of how learning actually happens</em>” (Luckin et al., 2016, p.18)  </p>\n\n<p>KI-Lectures „beleuchten“ ausgehend von dieser These interdisziplinäre Blickwinkel, unterschiedliche theoretische Positionen und Ansätze zu den Möglichkeiten von Künstlicher Intelligenz für Lernen und für Bildung. Gegenwärtige Diskurse zu Künstlicher Intelligenz in bildungswissenschaftlichen Kontexten weisen mitunter frappierende Parallelen zu früheren Entwicklungen auf wie intelligente tutorielle Systeme oder VR/AR-System nach dem Vorbild der 1990er Jahre. Mit dem Anknüpfen an diese Entwicklungen kehren auch die damit verbundenen theoretischen Fragen und Probleme zurück, wie sich auch an den weiteren Beispielen der anthropologischen Perspektiven auf das Verhältnis von Mensch und Maschine, der Fragen zu Diskursen über Kybernetisierungstendenzen in der Didaktik oder zur Individualisierung des Lernens zeigt.  </p>\n\n<p>Die Lectures greifen thematische Schwerpunkte aus einer theoretischen Perspektive auf und demonstrieren dazugehörige aktuelle Anwendungen anhand von Projektbeispielen. Da ethische Richtlinien in allen KI-Anwendungen eine Grundlage bilden sollten, wird dieser Thematik eine eigene Session gewidmet. Ein Theoriegespräch über die Bedeutung von Künstlicher Intelligenz für Bildung und Lernen schließt die KI-Lectures für das Sommersemester ab.</p>\n"
    embedder.tokenize(input_texts)
