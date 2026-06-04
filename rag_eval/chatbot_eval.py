import sys
import os
import asyncio
import json
import logging

from datetime import datetime
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =========================
# RAGAS
# =========================
from ragas import SingleTurnSample
from ragas.metrics import AnswerRelevancy, ContextRelevance, Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =========================
# Backend
# =========================
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import QuestionAnswerer
from src.llm.tools.contextualize import get_contextualizer
from src.llm.tools.retrieve import get_retriever
from src.llm.objects.reranker import Reranker
from eval_decompose import decompose_query_eval
from rag_eval.eval_retrieve_multi import retrieve_multi_parallel_eval
from src.llm.objects.citation_parser import CitationParser
from rag_eval.eval_synthesizer import synthesize_answer_eval

course_ids = [1, 6, 10, 19, 27, 28, 29, 36, 37, 38, 41, 43, 44, 46, 48, 49, 50, 51, 53, 55, 56, 57, 58, 59, 60, 63, 64, 66, 67, 74, 75, 76, 78, 79, 80, 88, 92, 93, 95, 98, 99, 100, 103, 106, 107, 109, 111, 121, 124, 127, 128, 134, 136, 140, 141, 142, 143, 144, 145, 151, 152, 153, 158, 160, 161, 163, 164, 165, 168, 171, 176, 177, 180, 185, 186, 187, 190, 191, 192, 197, 198, 199, 202, 203, 213, 214, 215, 221, 223, 224, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 241, 242, 243, 245, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 266, 267, 268, 270, 271, 272, 274, 275, 276, 279, 282, 283, 284, 285, 286, 287, 289, 290, 291, 293, 294, 295, 296, 297, 301, 304, 305, 311, 313, 316, 317, 318, 319, 320, 321, 322, 323, 325, 329, 330, 331, 332, 334, 335, 337, 338, 340, 341, 342, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 363, 364, 365, 367, 369, 370, 372, 373, 374, 385, 386, 387, 390, 397]

# =====================================================
# CONTEXT RETRIEVAL
# =====================================================

def answer_question(
    question: str,
    course_id: Optional[List[int]]
) -> Tuple[tuple, str]:

    retrieve_top_n = 10
    contextualizer = get_contextualizer()
    retriever = get_retriever(use_hybrid=True, n_chunks=retrieve_top_n)
    reranker = Reranker(top_n=10)
    question_answerer = QuestionAnswerer()
    citation_parser = CitationParser()

    is_moodle = course_id is not None
    mode = contextualizer.classify_scenario(query=question, model=Models.AZURE_FALLBACK)

    if mode == "multi_hop":
        decomposed = decompose_query_eval(model=Models.AZURE_FALLBACK, query=question)
        retrieved_nodes = retrieve_multi_parallel_eval(
            subqueries=decomposed,
            course_id=course_id,
            module_id=None,
            retrieve_top_n=retrieve_top_n
        )
        combined_nodes = synthesize_answer_eval(retrieved_nodes=retrieved_nodes)
        reranked_nodes = reranker.rerank(
            query=question,
            nodes=combined_nodes,
            model=Models.AZURE_FALLBACK
        )
    else:
        retrieved_nodes = retriever.retrieve(
            query=question,
            course_id=course_id,
            module_id=None
        )
        reranked_nodes = reranker.rerank(
            query=question,
            nodes=retrieved_nodes,
            model=Models.AZURE_FALLBACK
        )

    contexts = [node.text for node in reranked_nodes]

    response = question_answerer.answer_question(
        query=question,
        chat_history=[],
        sources=reranked_nodes,
        model=Models.AZURE_FALLBACK,
        language="German",
        is_moodle=is_moodle,
        course_id=course_id,
    )

    answer = citation_parser.parse(
        response.content,
        source_documents=reranked_nodes
    )

    return tuple(contexts), answer


# =====================================================
# MAIN EVALUATION
# =====================================================

async def main():
    print("\nStarting evaluation...\n")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-5.4", temperature=0)
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    metrics = [
        AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings),
        ContextRelevance(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
    ]

    # --------------------------------------------------
    # Evaluation Questions
    # --------------------------------------------------
    moodle_questions = [
        # 1. WISSEN ALLGEMEIN & KURSBEZOGEN
        #simple_hop_rag
        "Was ist erklärbarer KI (XAI)?",
        "Was bedeutet der Begriff Prompt in KI-Systemen?",
        "Welche Vorteile bietet der Einsatz von KI in der öffentlichen Verwaltung?",
        "Warum ist Data Awareness wichtig?",
        "Welche Podcasts zu Künstlicher Intelligenz werden auf dem KI-Campus angeboten?",
        "Wie arbeitet die Lernmethode Artificial Neural Networks (ANN)?",
        "Was ist der Unterschied zwischen Supervised und Unsupervised Learning?",
        "Was ist KI und Ethik?",
        "Welche Arten des Lernens werden im Maschinellen Lernen unterschieden?",
        "Wie funktionieren neuronale Netze?",
        "Warum ist Ethik bei KI wichtig?",
        "Welche Ziele verfolgt der EU AI Act?",
        "Was bedeutet Clustering?",
        "Was ist Reinforcement Learning?",
        "Was sind Trainingsdaten bei KI-Modellen?",
        "Was sind die Hauptphasen des Datenlebenszyklus?",
        "Was versteht man unter Learning Analytics?",
       
        #multi_hop_rag
        "Was ist „Generative KI & Chatbots“ und wie wird dieses Thema in den KI-Campus-Videos zur Reihe „ChatGPT – kurz erklärt“ erläutert?",
        "Was ist GANs und wie funktionieren Neuronale Netze?",
        "Wie beeinflusst Data Science den medizinischen Bereich?",
        "Wie beeinflusst Data Awareness die Qualität von Machine-Learning-Modellen?",
        "Welche Rolle spielt KI-Didaktik in der Hochschullehre?",
        "Welche Grundkonzepte des Maschinellen Lernens werden in KI-Campus-Kursen vermittelt?",
        "Wie hängen LLMs und Chatbots zusammen?",
        "Was sind k-NN und Regression?",
        "Gibt es einen Unterschied zwischen Automated Machine Learning und Machine Learning?",
        "Wie erklären die Lernmaterialien des KI-Campus den Unterschied zwischen generativer KI und klassischer KI?",
        "Welche Faktoren beeinflussen die Antwortqualität großer Sprachmodelle?",
        "Wie kann KI zur Erreichung der Ziele für nachhaltige Entwicklung beitragen?",
        "Welche Datenschutzprobleme können beim Einsatz von KI in der Medizin auftreten?"
        ]
    
    questions = [

        # 1. WISSEN ALLGEMEIN & KURSBEZOGEN
        #simple_hop_rag
        "Was ist erklärbarer KI (XAI)?",
        "Was bedeutet der Begriff Prompt in KI-Systemen?",
        "Welche Vorteile bietet der Einsatz von KI in der öffentlichen Verwaltung?",
        "Warum ist Data Awareness wichtig?",
        "Welche Podcasts zu Künstlicher Intelligenz werden auf dem KI-Campus angeboten?",
        "Wie arbeitet die Lernmethode Artificial Neural Networks (ANN)?",
        "Was ist der Unterschied zwischen Supervised und Unsupervised Learning?",
        "Was ist KI und Ethik?",
        "Welche Arten des Lernens werden im Maschinellen Lernen unterschieden?",
        "Wie funktionieren neuronale Netze?",
        "Warum ist Ethik bei KI wichtig?",
        "Welche Ziele verfolgt der EU AI Act?",
        "Was bedeutet Clustering?",
        "Was ist Reinforcement Learning?",
        "Was sind Trainingsdaten bei KI-Modellen?",
        "Was sind die Hauptphasen des Datenlebenszyklus?",
        "Was versteht man unter Learning Analytics?",

        #multi_hop_rag
        "Was ist „Generative KI & Chatbots“ und wie wird dieses Thema in den KI-Campus-Videos zur Reihe „ChatGPT – kurz erklärt“ erläutert?",
        "Was ist GANs und wie funktionieren Neuronale Netze?",
        "Wie beeinflusst Data Science den medizinischen Bereich?",
        "Wie beeinflusst Data Awareness die Qualität von Machine-Learning-Modellen?",
        "Welche Rolle spielt KI-Didaktik in der Hochschullehre?",
        "Welche Grundkonzepte des Maschinellen Lernens werden in KI-Campus-Kursen vermittelt?",
        "Wie hängen LLMs und Chatbots zusammen?",
        "Was sind k-NN und Regression?",
        "Gibt es einen Unterschied zwischen Automated Machine Learning und Machine Learning?",
        "Wie erklären die Lernmaterialien des KI-Campus den Unterschied zwischen generativer KI und klassischer KI?",
        "Welche Faktoren beeinflussen die Antwortqualität großer Sprachmodelle?",
        "Wie kann KI zur Erreichung der Ziele für nachhaltige Entwicklung beitragen?",
        "Welche Datenschutzprobleme können beim Einsatz von KI in der Medizin auftreten?",

        # 2. TECHNISCHER SUPPORT 
        #simple_hop_rag
        "Welche Filteroptionen stehen im Lernkatalog des KI-Campus zur Verfügung, um Kurse einzugrenzen?",
        "Wie funktioniert die Registrierung auf dem KI-Campus?",
        "Wie kann ich Informationen in meinem Profil ändern?",
        "In welchen Sprachen sind die Kurse im Lernkatalog des KI-Campus verfügbar?",
        "Welche Voraussetzungen muss ich erfüllen, um alle Plattforminhalte nutzen zu können?",
        "Ich habe mein Passwort vergessen, was kann ich tun?",
        "Wo finde ich den Bereich 'Meine Kurse'?",
        "Wie kann ich ein Konto auf der KI-Campus-Website erstellen?",
        "Was kann ich tun, wenn ich trotz eines Referenzlinks keinen Zugriff auf Inhalte habe?",
        "Wie kann ich Lernangebote nach Level (z. B. Einsteiger:innen) sortieren?",
       
        # 3. KURSMODALITÄTEN 
        #simple_hop_rag
        "Ist der KI-Campus kostenlos?",
        "Welche Funktionen haben Podcasts auf dem KI-Campus?",
        "Welche Informationen enthält eine Teilnahmebestätigung?",
        "Für welche Kurse wird ein Micro-Degree angeboten?",
        "Welche Meisterungskriterien gibt es für einen Kursabschluss auf dem KI-Campus?",
        "Welche Arten von Lernformaten bietet der KI-Campus an (z. B. Kurse, Videos, Podcasts)?",

        #multi_hop_rag
        "Was sind die Voraussetzungen für den Abschluss eines KI-Campus-Kurses?",
        "Welche Fristen gibt es in KI-Campus-Kursen und welche Bedeutung haben sie?",
        "Worin unterscheiden sich Teilnahmebestätigung, Leistungsnachweis und Zertifikat auf dem KI-Campus, und wann bekommt man welches Dokument?",
        "Wie hängen Pflichtaufgaben und Bewertung zusammen, wenn man einen KI-Campus-Kurs erfolgreich abschließen möchte?",
        
        # 4. ANFRAGEN ZUR CHATBOTFUNKTION 
        #simple_hop_rag
        "Stellt der KI-Campus Podcasts zur Verfügung?",
        "Gibt es Quizformate auf der Plattform zur Selbstüberprüfung?",
        "Beantwortet der Chatbot ausschließlich Fragen zu KI-Themen?",
        "Was ist der Zweck der KI-Campus-Plattform?",
        "Welche funktionalen Einschränkungen hat der Chatbot?",
        "Wie kann der Chatbot Informationen zu Leistungsnachweisen in Kursen bereitstellen?",

        #multi_hop_rag
        "Wie nutzt der Chatbot bereitgestellte Dokumente zur Beantwortung von Fragen?",
        "Welche Informationen erhalten Teilnehmende nach Abschluss eines KI-Campus-Kurses?",
        "Wie ist der KI-Campus-Chatbot technisch aufgebaut?",
        "Welche Zielgruppen spricht der KI-Campus laut Plattformbeschreibung an?",
    ]

    N_REPEATS = 5

    all_results = []
    global_metric_sums = {m.name: 0.0 for m in metrics}
    global_metric_count = 0

    # ungrounded tracking
    ungrounded_count = 0
    ungrounded_by_question = {}

    for q in questions:
        print(f"\n============================\nQuestion: {q}")
        per_question_metric_sums = {m.name: 0.0 for m in metrics}
        runs = []

        for run_idx in range(N_REPEATS):
            print(f"\n--- Run {run_idx + 1}/{N_REPEATS} ---")

            if q in moodle_questions:
                contexts, answer = answer_question(q, course_id=course_ids)
            else:
                contexts, answer = answer_question(q, course_id=None)

            print("\nAnswer:")
            print(answer)

            sample = SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=list(contexts),
            )

            run_scores = {}
            for m in metrics:
                score = float(await m.single_turn_ascore(sample))
                run_scores[m.name] = score
                per_question_metric_sums[m.name] += score
                global_metric_sums[m.name] += score
                print(f"  - {m.name}: {score:.3f}")

            # ungrounded detection
            answer_relevancy = run_scores.get("answer_relevancy", 0.0)
            context_relevance = run_scores.get("nv_context_relevance", 0.0)

            if context_relevance < 0.3 and answer_relevancy > 0.7:
                ungrounded_count += 1
                ungrounded_by_question[q] = ungrounded_by_question.get(q, 0) + 1

            runs.append({
                "run": run_idx + 1,
                "answer": answer,
                "scores": run_scores,
            })

            global_metric_count += 1

        avg_metrics = {
            k: v / N_REPEATS for k, v in per_question_metric_sums.items()
        }

        all_results.append({
            "question": q,
            "runs": runs,
            "average_metrics": avg_metrics
        })

    global_avg = {
        name: global_metric_sums[name] / global_metric_count
        for name in global_metric_sums
    }

    ungrounded_rate = ungrounded_count / global_metric_count

    print("\n============================")
    print("GLOBAL AVERAGE METRICS:")
    for name, avg in global_avg.items():
        print(f"  - {name}: {avg:.3f}")

    print("\nUNGROUNDED ANSWERS:")
    print(f"  - Ungrounded answers: {ungrounded_count}")
    print(f"  - Ungrounded answer rate: {ungrounded_rate:.3f}")

    output = {
        "results": all_results,
        "global_average_metrics": global_avg,
        "repetitions": N_REPEATS,
        "ungrounded_answers": {
            "count": ungrounded_count,
            "rate": ungrounded_rate,
            "by_question": ungrounded_by_question,
        },
    }

    outfile = f"chatbot_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {outfile}")


if __name__ == "__main__":
    asyncio.run(main())
