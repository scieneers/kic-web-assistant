import re

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore


class OutputParserTool:
    """This class is responsible for parsing the document references from the return string."""

    def __init__(self) -> None:
        self.name = "OutputParser"

    def _clean_up_answer(self, answer) -> str:
        """Remove double spaces and remove doc references in false format.
        Keep only references with allowed format [docN] where N=integer. E.g. [doc5] instead of [doc10.2] or doc[2.3.1]
        """
        answer = answer.replace("  ", " ")
        # Anything where N is a float
        answer = re.sub(r"\[doc\d+(?:\.\d+)+\]", "", answer)
        # Anything where N is not a digit
        answer = re.sub(r"\[doc\D\]", "", answer)
        return answer

    def _get_source_docs_from_answer(self, answer) -> list[int]:
        """Extract all [docN] from answer and extract N, and just return the N's as a list of ints"""
        results = re.findall(r"\[doc(\d+)\]", answer)
        doc_ids = [int(i) for i in results]
        # remove duplicates while preserving order
        doc_ids = list(dict.fromkeys(doc_ids))
        return doc_ids

    def _make_doc_references_sequential(self, answer, doc_ids) -> str:
        """Make references consecutive, e.g. for two references [doc1][doc3] -> [doc1][doc2] or a single ref [doc4] -> [doc1]"""
        sorted_doc_ids = sorted(doc_ids)
        target_doc_ids = [i for i in range(1, len(doc_ids) + 1)]

        if sorted_doc_ids == target_doc_ids:
            return answer

        for current, target in zip(sorted_doc_ids, target_doc_ids):
            answer = answer.replace(f"[doc{current}]", f"[doc{target}]")

        return answer

    def _remove_doc_ids_from_answer(self, answer, doc_ids) -> str:
        """Remove all [docN] from answer where N is in doc_ids list. Used for hallucinated references."""
        for idx in doc_ids:
            answer = answer.replace(f"[doc{idx}]", "")
        return answer

    def parse(
        self,
        answer: str,
        source_documents: list[NodeWithScore],
        **kwargs: dict,
    ) -> ChatMessage:
        answer = self._clean_up_answer(answer)
        doc_ids = self._get_source_docs_from_answer(answer)

        # we track doc ids which exist in case of hallucination
        real_doc_ids = []
        fake_doc_ids = []

        for i in doc_ids:
            idx = i - 1
            try:
                doc = source_documents[idx]

                if doc.metadata.get("url") in answer:
                    answer = re.sub(rf"(, )?\[doc{i}\]", "", answer)
                else:
                    answer = re.sub(rf"(, )?\[doc{i}\]", rf"\1({doc.metadata.get('url')})", answer)

                real_doc_ids.append(i)
            except IndexError:
                print(f"Could not find doc{i} in source documents")
                fake_doc_ids.append(i)

        answer = self._remove_doc_ids_from_answer(answer, fake_doc_ids)
        answer = self._make_doc_references_sequential(answer, real_doc_ids)
        return answer
