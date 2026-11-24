"""
Teste die Extraktion aller Interaktionstypen aus H5P Interactive Videos.
Verwendet die bereits heruntergeladenen JSON-Files aus test_quiz_extraction.py
"""

import json
import sys
from pathlib import Path

# Füge src zum Path hinzu
sys.path.insert(0, str(Path(__file__).parents[4]))

from src.loaders.models.hp5activities import (
    QuizQuestion,
    TrueFalseQuestion,
    FillInBlanksQuestion,
    DragDropQuestion,
    TextBanner,
)


def parse_interaction(interaction: dict):
    """Parst eine einzelne Interaktion und gibt das entsprechende Objekt zurück."""
    action = interaction.get("action", {})
    library = action.get("library", "")
    params = action.get("params", {})
    
    # MultiChoice
    if "H5P.MultiChoice" in library:
        question_text = params.get("question", "").strip()
        
        correct = []
        incorrect = []
        for answer in params.get("answers", []):
            text = answer.get("text", "").strip()
            if text:
                if answer.get("correct"):
                    correct.append(text)
                else:
                    incorrect.append(text)
        
        if question_text and correct:
            return QuizQuestion(
                type=library,
                question=question_text,
                correct_answers=correct,
                incorrect_answers=incorrect
            )
    
    # SingleChoiceSet (different structure)
    elif "H5P.SingleChoiceSet" in library:
        choices = params.get("choices", [])
        all_questions = []
        
        for choice in choices:
            question_text = choice.get("question", "").strip()
            answers = choice.get("answers", [])
            
            if question_text and answers:
                # First answer is always correct in SingleChoiceSet
                correct = [answers[0].strip()] if answers else []
                incorrect = [a.strip() for a in answers[1:] if a.strip()]
                
                if correct:
                    all_questions.append(QuizQuestion(
                        type=library,
                        question=question_text,
                        correct_answers=correct,
                        incorrect_answers=incorrect
                    ))
        
        # Return first question (we'll handle multiple later if needed)
        return all_questions[0] if all_questions else None
    
    # Wahr/Falsch
    elif "H5P.TrueFalse" in library:
        question_text = params.get("question", "").strip()
        correct_str = params.get("correct", "").lower()
        
        if question_text and correct_str in ["true", "false"]:
            correct_answer = (correct_str == "true")
            return TrueFalseQuestion(
                type=library,
                question=question_text,
                correct_answer=correct_answer
            )
    
    # Lückentext
    elif "H5P.Blanks" in library:
        intro_text = params.get("text", "").strip()
        questions = params.get("questions", [])
        
        if questions:
            text_with_blanks = questions[0].strip() if questions else ""
            
            if intro_text or text_with_blanks:
                question_text = intro_text if intro_text else "Lückentext"
                return FillInBlanksQuestion(
                    type=library,
                    question=question_text,
                    text_with_blanks=text_with_blanks
                )
    
    # Drag & Drop
    elif "H5P.DragQuestion" in library:
        task = params.get("question", {}).get("task", {})
        dropzones = task.get("dropZones", [])
        elements = task.get("elements", [])
        
        question_text = "Ordne die Elemente den Kategorien zu:"
        
        # Extrahiere Kategorien (Dropzones) mit Index
        categories = []
        category_map = {}  # Index -> Label
        for idx, dz in enumerate(dropzones):
            label = dz.get("label", "").strip()
            if label:
                categories.append(label)
                category_map[str(idx)] = label
        
        # Extrahiere ziehbare Elemente mit Index
        draggable_items = []
        element_map = {}  # Index -> Text
        for idx, elem in enumerate(elements):
            text = elem.get("type", {}).get("params", {}).get("text", "").strip()
            if text:
                draggable_items.append(text)
                element_map[str(idx)] = text
        
        # Erstelle korrekte Zuordnungen
        correct_mappings = {}
        for idx, dz in enumerate(dropzones):
            label = dz.get("label", "").strip()
            correct_elem_ids = dz.get("correctElements", [])
            
            if label and correct_elem_ids:
                correct_items = []
                for elem_id in correct_elem_ids:
                    elem_text = element_map.get(str(elem_id))
                    if elem_text:
                        correct_items.append(elem_text)
                
                if correct_items:
                    correct_mappings[label] = correct_items
        
        if categories and draggable_items and correct_mappings:
            return DragDropQuestion(
                type=library,
                question=question_text,
                categories=categories,
                draggable_items=draggable_items,
                correct_mappings=correct_mappings
            )
    
    # Text
    elif "H5P.Text" in library:
        text = params.get("text", "").strip()
        if text:
            return TextBanner(
                type=library,
                text=text
            )
    
    return None


def main():
    output_dir = Path(__file__).parent / "output"
    
    # Zähler für Interaktionstypen
    stats = {
        "QuizQuestion": 0,
        "TrueFalseQuestion": 0,
        "FillInBlanksQuestion": 0,
        "DragDropQuestion": 0,
        "TextBanner": 0,
        "Unknown": 0,
    }
    
    print("=== TESTE INTERAKTIONS-EXTRAKTION ===\n")
    
    # Durchlaufe alle content_*.json Files
    for json_file in sorted(output_dir.glob("content_module_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        # Extrahiere Interaktionen
        iv = content.get("interactiveVideo", {})
        interaction_list = []
        
        if "assets" in iv and "interactions" in iv["assets"]:
            interaction_list = iv["assets"]["interactions"]
        elif "interactions" in iv:
            interaction_list = iv["interactions"]
        
        if not interaction_list:
            continue
        
        print(f"\n--- {json_file.name} ---")
        print(f"Gefunden: {len(interaction_list)} Interaktionen")
        
        for idx, interaction in enumerate(interaction_list):
            parsed = parse_interaction(interaction)
            
            if parsed:
                class_name = type(parsed).__name__
                stats[class_name] += 1
                
                print(f"\n  [{idx+1}] {class_name}")
                # Zeige den kompletten Text mit Einrückung
                full_text = parsed.to_text()
                for line in full_text.split('\n'):
                    print(f"      {line}")
            else:
                stats["Unknown"] += 1
                library = interaction.get("action", {}).get("library", "UNKNOWN")
                print(f"\n  [{idx+1}] NICHT ERKANNT: {library}")
    
    # Zusammenfassung
    print("\n\n=== ZUSAMMENFASSUNG ===")
    total = sum(stats.values())
    print(f"Total: {total} Interaktionen")
    for key, count in stats.items():
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {key}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Test abgeschlossen!")


if __name__ == "__main__":
    main()
