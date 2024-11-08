from typing import Any

import pytest
from llama_index.core.schema import TextNode

from src.llm.parser.citation_parser import CITATION_TEXT, CitationParser

examples: list[Any] = [
    (
        "Bei Bedarf kann diese Summe auf 500.000 Euro erhöht werden. [doc1][doc2][doc3]",
        f'Bei Bedarf kann diese Summe auf 500.000 Euro erhöht werden. {CITATION_TEXT.format(url="fake_1.url", index=1)}{CITATION_TEXT.format(url="fake_2.url", index=2)}{CITATION_TEXT.format(url="fake_3.url", index=3)}',
    ),
    ("Wrong [docX][doc1.1], Right [doc1].", f'Wrong , Right {CITATION_TEXT.format(url="fake_1.url", index=1)}.'),
    (
        "[doc1][doc2] beginning.",
        f'{CITATION_TEXT.format(url="fake_1.url", index=1)}{CITATION_TEXT.format(url="fake_2.url", index=2)} beginning.',
    ),
    ("Not existing [doc100]", "Not existing "),
    (
        "Unusual [doc2] order [doc4][doc1][doc2][doc1]",
        f'Unusual {CITATION_TEXT.format(url="fake_2.url", index=2)} order {CITATION_TEXT.format(url="fake_4.url", index=3)}{CITATION_TEXT.format(url="fake_1.url", index=1)}{CITATION_TEXT.format(url="fake_2.url", index=2)}{CITATION_TEXT.format(url="fake_1.url", index=1)}',
    ),
    (
        "Made consecutive [doc2][doc3]",
        f'Made consecutive {CITATION_TEXT.format(url="fake_2.url", index=1)}{CITATION_TEXT.format(url="fake_3.url", index=2)}',
    ),
    (
        "[doc3] high refs [doc2]",
        f'{CITATION_TEXT.format(url="fake_3.url", index=2)} high refs {CITATION_TEXT.format(url="fake_2.url", index=1)}',
    ),
    (
        "Dies wird in [doc2] und [doc4] angegeben.",
        f'Dies wird in {CITATION_TEXT.format(url="fake_2.url", index=1)} und {CITATION_TEXT.format(url="fake_4.url", index=2)} angegeben.',
    ),
    (
        "Dies wird in [doc4] und [doc1] angegeben.",
        f'Dies wird in {CITATION_TEXT.format(url="fake_4.url", index=2)} und {CITATION_TEXT.format(url="fake_1.url", index=1)} angegeben.',
    ),
    (
        "Dies wird in [doc4] und [doc1], [doc3] angegeben.",
        f'Dies wird in {CITATION_TEXT.format(url="fake_4.url", index=3)} und {CITATION_TEXT.format(url="fake_1.url", index=1)}, {CITATION_TEXT.format(url="fake_3.url", index=2)} angegeben.',
    ),
]


@pytest.mark.parametrize("answer, processed", examples)
def test_answer_parsing(answer: str, processed: str):
    sources = [TextNode(text=f"doc_{i}", metadata={"url": f"fake_{i}.url"}) for i in range(1, 5)]

    answer_parsed = CitationParser().parse(answer=answer, source_documents=sources)
    assert answer_parsed == processed
