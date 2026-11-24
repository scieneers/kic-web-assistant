"""Mini-Benchmark: Sokratischer Lernassistent v1 vs. v2.

Simulierte Lernende (Persona-LLM) spielen echte Sessions gegen beide
Varianten (echte LLM-Calls, echter Index) und die Transkripte werden mit
deterministischen Metriken verglichen — kein LLM-Judge, keine Wissenschaft:
schnelle, reproduzierbare Indizien für die v1-vs-v2-Entscheidung.

Ablauf pro Session:
  1. Start über das API-Flag (start_socratic / start_socratic_v2)
  2. max_turns Runden: Persona-LLM antwortet auf die letzte Tutor-Nachricht
  3. Weicher Ausstieg ("Danke, lass gut sein ...") → misst, ob die Variante
     auch nicht-wörtliche Exits erkennt

Metriken (alle ohne LLM, direkt aus Transkript + Graph-State):
  - completed:        Konsolidierung/Abschluss von der Variante selbst erreicht
  - question_rate:    Anteil Tutor-Nachrichten mit Frage
  - max_q_streak:     längste Folge reiner Frage-Turns (Frage-Monokultur)
  - repeated_qs:      nahezu identische Tutorfragen (difflib > 0.85)
  - talk_ratio:       Wortanteil Lernende/Tutor (höher = aktivere Lernende)
  - quiz_leaks:       Quiz gestellt & Lösung im selben Tutor-Text verraten (v2)
  - soft_exit_ok:     weicher Ausstieg hat die Session beendet
  - moves:            Verteilung der ausgeführten Moves (v2, aus v2_last_move)
  - guard_overrides:  Turns, in denen ein Guard den Policy-Move überstimmt hat (v2)
  - final_learner_model: letzter Lernstand vor dem State-Reset (v2)

Beispiel:
  uv run python scripts/llms/socratic_benchmark.py \
      --course-id 42 --module-id 123 --runs 1 --max-turns 8

Benötigt dieselben Env-Variablen wie scripts/interactive_chat.py
(Search-Index + LLM-Zugänge). Ergebnisse landen unter
outputs/socratic_benchmark/<zeitstempel>/ (Transkripte + report.md).
"""

import argparse
import difflib
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.llm.assistant import KICampusAssistant
from src.llm.objects.LLMs import LLM, Models

# ---------------------------------------------------------------------------
# Personas — simulierte Lernende. Bewusst wenige, dafür unterscheidbar.
# ---------------------------------------------------------------------------

PERSONA_COMMON = (
    "Du simulierst eine lernende Person im Chat mit einem Lern-Tutor eines Online-Kurses. "
    "Antworte IMMER nur als diese Person, in 1-3 kurzen Sätzen, auf Deutsch. "
    "Stelle dich nie als KI vor, erkläre nie deine Rolle, übernimm nie die Tutor-Rolle. "
    "Wenn der Tutor Themen zur Auswahl anbietet, wähle das erste. "
    "Wenn der Tutor fragt, was du lernen willst, sage: die Inhalte dieses Moduls verstehen."
)

PERSONAS = {
    "novize": PERSONA_COMMON + (
        " Dein Charakter: absolute:r Anfänger:in. Du weißt fast nichts über das Thema, "
        "antwortest zögerlich und rätst manchmal falsch, gibst dir aber Mühe."
    ),
    "fortgeschritten": PERSONA_COMMON + (
        " Dein Charakter: fortgeschritten. Du beantwortest die meisten Fragen korrekt und "
        "präzise und stellst gelegentlich eine vertiefende Rückfrage."
    ),
    "frustriert": PERSONA_COMMON + (
        " Dein Charakter: schnell frustriert. Du antwortest oft mit 'weiß nicht', 'keine Ahnung' "
        "oder 'das bringt doch nichts', bleibst aber im Gespräch."
    ),
    "abkuerzer": PERSONA_COMMON + (
        " Dein Charakter: ungeduldig. Du willst keine Gegenfragen, sondern forderst regelmäßig "
        "die Lösung ein ('sag mir einfach die Antwort'). Zwischendurch antwortest du knapp."
    ),
}

START_QUERY = "Ich möchte etwas lernen."
SOFT_EXIT_QUERY = "Danke, das reicht mir für heute. Lass gut sein."

# ---------------------------------------------------------------------------
# Session-Runner
# ---------------------------------------------------------------------------

_persona_llm = LLM()


def _persona_reply(persona_prompt: str, transcript: list[dict], persona_model: Models) -> str:
    """Nächste Lernenden-Antwort aus den letzten Turns generieren."""
    tail = transcript[-6:]
    lines = ["Bisheriger Dialog (Ausschnitt):"]
    for turn in tail:
        speaker = "TUTOR" if turn["role"] == "tutor" else "DU"
        lines.append(f"{speaker}: {turn['text']}")
    lines.append("\nDeine nächste Antwort als lernende Person:")
    response = _persona_llm.chat(
        query="\n".join(lines),
        chat_history=[],
        model=persona_model,
        system_prompt=persona_prompt,
    )
    reply = (response.content or "").strip()
    return reply or "Hm, kannst du das anders formulieren?"


def _state_snapshot(assistant: KICampusAssistant, thread_id: str) -> dict:
    values = assistant.graph.get_state({"configurable": {"thread_id": thread_id}}).values
    pending = values.get("v2_pending_quiz") or {}
    return {
        "socratic_mode": values.get("socratic_mode"),
        "socratic_v2_phase": values.get("socratic_v2_phase"),
        "pending_quiz_solutions": list(pending.get("correct_answers") or []),
        # Move-Tracking (v2): jeder v2-Node setzt die Felder pro Turn explizit,
        # d. h. der Snapshot zeigt den Move DIESES Turns (None im Opening).
        "last_move": values.get("v2_last_move"),
        "last_policy_move": values.get("v2_last_policy_move"),
        "target_concept": values.get("v2_target_concept"),
        "learner_model": values.get("v2_learner_model"),
    }


def run_session(
    variant: str,
    persona_name: str,
    course_id: int,
    module_id,
    tutor_model: Models,
    persona_model: Models,
    max_turns: int,
    reranker_type: str,
) -> dict:
    """Eine komplette Session einer Persona gegen eine Variante."""
    assistant = KICampusAssistant(
        enable_socratic=(variant == "v1"),
        enable_socratic_v2=(variant == "v2"),
        reranker_type=reranker_type,
    )

    transcript: list[dict] = []
    states: list[dict] = []
    session_self_ended = False

    def send(query: str) -> str:
        transcript.append({"role": "learner", "text": query})
        message, thread_id = assistant.chat_with_course(
            query=query,
            model=tutor_model,
            course_id=course_id,
            module_id=module_id,
            thread_id=send.thread_id,
            start_socratic=(variant == "v1" and send.thread_id is None),
            start_socratic_v2=(variant == "v2" and send.thread_id is None),
        )
        send.thread_id = thread_id
        text = message.content or ""
        transcript.append({"role": "tutor", "text": text})
        states.append(_state_snapshot(assistant, thread_id))
        return text

    send.thread_id = None

    send(START_QUERY)
    persona_prompt = PERSONAS[persona_name]

    for _ in range(max_turns):
        state = states[-1]
        if state["socratic_mode"] is None and state["socratic_v2_phase"] is None:
            session_self_ended = True
            break
        reply = _persona_reply(persona_prompt, transcript, persona_model)
        send(reply)

    soft_exit_ok = None
    if not session_self_ended:
        send(SOFT_EXIT_QUERY)
        final = states[-1]
        soft_exit_ok = final["socratic_mode"] is None and final["socratic_v2_phase"] is None

    return {
        "variant": variant,
        "persona": persona_name,
        "transcript": transcript,
        "states": states,
        "session_self_ended": session_self_ended,
        "soft_exit_ok": soft_exit_ok,
    }


# ---------------------------------------------------------------------------
# Deterministische Metriken
# ---------------------------------------------------------------------------

def _words(text: str) -> int:
    return len(text.split())


def _tutor_turns(transcript: list[dict]) -> list[str]:
    return [t["text"] for t in transcript if t["role"] == "tutor"]


def _questions(tutor_texts: list[str]) -> list[str]:
    """Letzte Frage je Tutor-Turn (grob: letzter '?'-Satz)."""
    questions = []
    for text in tutor_texts:
        sentences = [s.strip() for s in re.split(r"(?<=[?])\s+", text) if s.strip().endswith("?")]
        if sentences:
            questions.append(sentences[-1].lower())
    return questions


def compute_metrics(session: dict) -> dict:
    transcript = session["transcript"]
    tutor_texts = _tutor_turns(transcript)
    learner_texts = [t["text"] for t in transcript if t["role"] == "learner"]

    has_question = ["?" in text for text in tutor_texts]
    max_streak = streak = 0
    for is_q in has_question:
        streak = streak + 1 if is_q else 0
        max_streak = max(max_streak, streak)

    questions = _questions(tutor_texts)
    repeated = sum(
        1
        for i in range(len(questions))
        for j in range(i + 1, len(questions))
        if difflib.SequenceMatcher(None, questions[i], questions[j]).ratio() > 0.85
    )

    tutor_words = sum(_words(t) for t in tutor_texts) or 1
    learner_words = sum(_words(t) for t in learner_texts)

    # Quiz-Leak: Tutor-Turn, der ein Quiz stellt (pending danach), enthält die Lösung.
    quiz_posed = 0
    quiz_leaks = 0
    for state, text in zip(session["states"], tutor_texts):
        solutions = [s for s in state["pending_quiz_solutions"] if len(s) > 3]
        if state["pending_quiz_solutions"]:
            quiz_posed += 1
        lowered = text.lower()
        if any(
            re.search(rf"korrekte? antwort\w*\s*(ist|sind|:).{{0,40}}{re.escape(sol.lower())}", lowered)
            for sol in solutions
        ):
            quiz_leaks += 1

    # Move-Tracking (nur v2; v1-Sessions haben keine v2_last_move-Felder).
    # Pro Snapshot = pro Tutor-Turn; Opening setzt None und fällt raus.
    states = session["states"]
    moves = [s["last_move"] for s in states if s.get("last_move")]
    guard_overrides = sum(
        1
        for s in states
        if s.get("last_move") and s.get("last_policy_move") and s["last_move"] != s["last_policy_move"]
    )
    # Letzter Lernstand VOR dem State-Reset (EXIT/Konsolidierung setzen das
    # Learner-Model auf None, daher rückwärts das letzte gefüllte nehmen).
    final_learner_model = next((s["learner_model"] for s in reversed(states) if s.get("learner_model")), None)

    return {
        "turns": len(tutor_texts),
        "completed": session["session_self_ended"],
        "question_rate": round(sum(has_question) / max(len(tutor_texts), 1), 2),
        "max_q_streak": max_streak,
        "repeated_qs": repeated,
        "talk_ratio": round(learner_words / tutor_words, 2),
        "quiz_posed": quiz_posed,
        "quiz_leaks": quiz_leaks,
        "soft_exit_ok": session["soft_exit_ok"],
        "moves": dict(Counter(moves)),
        "guard_overrides": guard_overrides,
        "final_learner_model": final_learner_model,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _format_moves(moves: dict) -> str:
    return ", ".join(f"{move} ×{count}" for move, count in sorted(moves.items(), key=lambda kv: -kv[1])) or "—"


def _format_learner_model(learner_model: dict | None) -> str:
    if not learner_model:
        return "—"
    concepts = learner_model.get("konzepte") or {}
    misconceptions = learner_model.get("missverstaendnisse") or []
    affect = learner_model.get("affekt") or "neutral"
    concept_str = ", ".join(f"{name}: {status}" for name, status in concepts.items()) or "keine Konzepte erfasst"
    parts = [concept_str]
    if misconceptions:
        parts.append(f"Missverständnisse: {'; '.join(misconceptions)}")
    parts.append(f"Affekt: {affect}")
    return " · ".join(parts)


def build_report(results: list[dict]) -> str:
    lines = [
        "# Socratic Benchmark v1 vs. v2",
        "",
        f"Sessions: {len(results)} · erzeugt: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Variante | Persona | Turns | Abschluss erreicht | Frage-Rate | Max. Frage-Streak | Wiederholte Fragen | Talk-Ratio | Quiz gestellt | Quiz-Leaks | Weicher Exit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        soft = "—" if m["soft_exit_ok"] is None else ("✅" if m["soft_exit_ok"] else "❌")
        lines.append(
            f"| {r['variant']} | {r['persona']} | {m['turns']} | {'✅' if m['completed'] else '❌'} "
            f"| {m['question_rate']} | {m['max_q_streak']} | {m['repeated_qs']} | {m['talk_ratio']} "
            f"| {m['quiz_posed']} | {m['quiz_leaks']} | {soft} |"
        )

    lines += ["", "## Mittelwerte pro Variante", ""]
    lines.append("| Variante | Sessions | Abschlussquote | Frage-Rate | Max. Streak | Wiederholte Fragen | Talk-Ratio | Weicher Exit ok |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for variant in sorted({r["variant"] for r in results}):
        group = [r["metrics"] for r in results if r["variant"] == variant]
        soft_known = [m["soft_exit_ok"] for m in group if m["soft_exit_ok"] is not None]
        lines.append(
            f"| {variant} | {len(group)} "
            f"| {sum(m['completed'] for m in group)}/{len(group)} "
            f"| {round(statistics.mean(m['question_rate'] for m in group), 2)} "
            f"| {round(statistics.mean(m['max_q_streak'] for m in group), 1)} "
            f"| {sum(m['repeated_qs'] for m in group)} "
            f"| {round(statistics.mean(m['talk_ratio'] for m in group), 2)} "
            f"| {sum(soft_known)}/{len(soft_known) if soft_known else 0} |"
        )
    # Move-Tracking gibt es nur in v2 — v1 taucht hier bewusst nicht auf.
    v2_results = [r for r in results if r["variant"] == "v2"]
    if v2_results:
        lines += [
            "",
            "## Moves & Lernstand (v2)",
            "",
            "| Persona | Lauf | Moves (ausgeführt) | Guard-Overrides | Lernstand am Ende |",
            "|---|---|---|---|---|",
        ]
        run_counter: dict = {}
        for r in v2_results:
            m = r["metrics"]
            run_counter[r["persona"]] = run_counter.get(r["persona"], 0) + 1
            lines.append(
                f"| {r['persona']} | {run_counter[r['persona']]} | {_format_moves(m['moves'])} "
                f"| {m['guard_overrides']} | {_format_learner_model(m['final_learner_model'])} |"
            )
        lines += [
            "",
            "Guard-Overrides = Turns, in denen die deterministischen Leitplanken den Policy-Vorschlag "
            "überstimmt haben (Frage-Streak → HINT, Hint-Limit → MICRO_EXPLAIN, zu frühes CONSOLIDATE → FRAGE). "
            "Viele Overrides deuten auf eine driftende Policy hin.",
        ]

    lines += [
        "",
        "Lesart: höhere Talk-Ratio und Abschlussquote sowie weniger Streaks/Wiederholungen sprechen "
        "für besseres Tutoring; Quiz-Leaks müssen 0 sein. Kein Ersatz für die Testphase mit Menschen — "
        "ein schneller, reproduzierbarer Indikator.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Mini-Benchmark Sokratik v1 vs. v2 (simulierte Lernende).")
    p.add_argument("--course-id", type=int, required=True)
    p.add_argument("--module-id", type=int, default=None, help="Modul-ID (für v2 empfohlen)")
    p.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    p.add_argument("--personas", nargs="+", default=list(PERSONAS), choices=list(PERSONAS))
    p.add_argument("--runs", type=int, default=1, help="Wiederholungen pro Persona/Variante")
    p.add_argument("--max-turns", type=int, default=8, help="Lernenden-Antworten pro Session")
    p.add_argument("--model", default="AZURE_FALLBACK", help="Tutor-Modell (Models-Enum-Name)")
    p.add_argument("--persona-model", default="MINI", help="Persona-Modell (Models-Enum-Name)")
    p.add_argument("--reranker", default="azure_semantic", help="Reranker-Backend")
    p.add_argument("--out", default=None, help="Ausgabeverzeichnis (Default: outputs/socratic_benchmark/<ts>)")
    return p.parse_args()


def main():
    args = parse_args()
    tutor_model = getattr(Models, args.model)
    persona_model = getattr(Models, args.persona_model)

    out_dir = Path(args.out) if args.out else Path("outputs/socratic_benchmark") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(args.variants) * len(args.personas) * args.runs
    done = 0
    for variant in args.variants:
        for persona in args.personas:
            for run in range(args.runs):
                done += 1
                print(f"[{done}/{total}] {variant} · {persona} · Lauf {run + 1} ...", flush=True)
                try:
                    session = run_session(
                        variant=variant,
                        persona_name=persona,
                        course_id=args.course_id,
                        module_id=args.module_id,
                        tutor_model=tutor_model,
                        persona_model=persona_model,
                        max_turns=args.max_turns,
                        reranker_type=args.reranker,
                    )
                except Exception as exc:  # eine kaputte Session soll den Lauf nicht beenden
                    print(f"    FEHLER: {exc}")
                    continue
                session["metrics"] = compute_metrics(session)
                results.append(session)
                name = f"{variant}_{persona}_run{run + 1}.json"
                (out_dir / name).write_text(json.dumps(session, ensure_ascii=False, indent=2))
                print(f"    -> {session['metrics']}")

    report = build_report(results)
    (out_dir / "report.md").write_text(report)
    print("\n" + report)
    print(f"\nTranskripte + Report: {out_dir}")


if __name__ == "__main__":
    main()
