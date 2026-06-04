
# Metrics via AspectCritic :
#   - context_precision
#   - context_recall
#   - faithfulness
#   - answer_relevancy
#   - context_relevance

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import AspectCritic

# Uses local chatbot implementation
from rag import ExampleRAG, SimpleKeywordRetriever

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "kicampus")

# === Questions to evaluate ===
QUESTIONS: List[str] = [
    "what is the history of AI",
    "how can I reset my password on KI Campus?",
    "how can I see my course plan on KI Campus?",
]

# A concise reference to help the LLM critics 
REFERENCE = (
    "AI began as a field in 1956 at the Dartmouth workshop. Early optimism led to an AI winter in the 1970s. "
    "Expert systems revived interest in the 1980s. Deep learning advances around 2012 (ImageNet) sparked a resurgence. "
    "Transformers (2017) enabled large language models such as ChatGPT (2022)."
)

# ------------------------ helper: load docs ------------------------
def load_docs() -> List[str]:
    docs: List[str] = []
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.lower().endswith((".txt", ".md")):
                p = os.path.join(DATA_DIR, fname)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        t = f.read().strip()
                        if t:
                            docs.append(t)
                except Exception as e:
                    print(f"Warn: could not read {p}: {e}")
    if not docs:
        docs = [
            "Artificial intelligence (AI) emerged as a discipline in 1956 at the Dartmouth workshop. "
            "Early optimism gave way to funding cuts in the 1970s 'AI winter'. Expert systems in the 1980s revived "
            "interest before another slowdown. The deep learning renaissance began around 2012 with ImageNet. "
            "Transformers in 2017 led to large language models like ChatGPT (2022).",
            "Auf KI Campus finden Lernende unter 'Meine Kurse' ihre aktiv belegten Lehrangebote. "
            "Ein Passwort lässt sich in der Regel über 'Passwort vergessen' auf der Login-Seite zurücksetzen.",
        ]
    return docs

# ------------------------ helper: call your chatbot ------------------------
def get_answer_and_contexts(rag: ExampleRAG, question: str, top_k: int = 3) -> tuple[str, List[str]]:
    """
    Uses your ExampleRAG to:
      1) retrieve top-k docs (for contexts)
      2) generate the answer
    Returns (answer, contexts_as_text_list)
    """
    # 1) retrieve contexts (explicitly)
    retrieved = rag.retrieve_documents(question, top_k=top_k)
    contexts = [d["content"] for d in retrieved] if retrieved else []

    # 2) generate the answer (your pipeline)
    answer = rag.generate_response(question, top_k=top_k)

    return answer.strip(), contexts

# ------------------------ metrics ------------------------
def build_metrics(async_llm) -> List[AspectCritic]:
    return [
        AspectCritic(
            name="context_precision",
            definition="Evaluate whether the top retrieved contexts are directly relevant to the question and ranked with the most relevant first.",
            llm=async_llm,
        ),
        AspectCritic(
            name="context_recall",
            definition="Evaluate whether the retrieved contexts collectively contain the key facts needed to answer the question.",
            llm=async_llm,
        ),
        AspectCritic(
            name="faithfulness",
            definition="Evaluate whether the answer is fully supported by the retrieved contexts; penalize any unsupported claims.",
            llm=async_llm,
        ),
        AspectCritic(
            name="answer_relevancy",
            definition="Evaluate whether the answer directly addresses the question without going off-topic.",
            llm=async_llm,
        ),
        AspectCritic(
            name="context_relevance",
            definition="Evaluate whether the selected contexts are on-topic and useful for answering the question.",
            llm=async_llm,
        ),
    ]

# ------------------------ main ------------------------
async def main():
    # 0) Spin up your local chatbot (ExampleRAG)
    llm_sync = OpenAI()  # uses OPENAI_API_KEY from env
    retriever = SimpleKeywordRetriever()
    bot = ExampleRAG(llm_client=llm_sync, retriever=retriever, logdir="logs")
    bot.set_documents(load_docs())

    # 1) LLM for Ragas metrics (async)
    metrics_client = AsyncOpenAI()
    async_llm = llm_factory("gpt-5.4", client=metrics_client)
    critics = build_metrics(async_llm)

    all_results: List[Dict[str, Any]] = []

    for q in QUESTIONS:
        print(f"\n💬 Question: {q}")
        answer, contexts = get_answer_and_contexts(bot, q, top_k=3)
        print(f"🤖 Answer:\n{answer}\n")
        print(f"📚 Retrieved contexts ({len(contexts)}):")
        for i, c in enumerate(contexts, 1):
            preview = (c[:180] + "…") if len(c) > 180 else c
            print(f"  [{i}] {preview}")

        # 2) Score all five metrics
        metric_results: Dict[str, Any] = {}
        print("\n📏 Ragas (AspectCritic) metrics")
        for m in critics:
            s = await m.ascore(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=REFERENCE,  # optional but improves judgments
            )
            metric_results[m.name] = {
                "score": float(s.value),
                "reason": getattr(s, "reason", None),
            }
            print(f"   - {m.name}: {float(s.value):.3f}")
            if getattr(s, "reason", None):
                print(f"     reason: {s.reason}")

        all_results.append(
            {
                "question": q,
                "answer": answer,
                "retrieved_contexts": contexts,
                "reference": REFERENCE,
                "metrics": metric_results,
            }
        )

    # 3) Save results
    out_fn = f"kicampus_eval_min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(
            {"created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "results": all_results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n✅ Saved: {out_fn}")

if __name__ == "__main__":
    asyncio.run(main())
