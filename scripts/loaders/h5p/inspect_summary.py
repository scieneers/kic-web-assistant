"""Test Summary extraction from H5P Interactive Videos."""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders.models.hp5activities import Summary

def test_summary_extraction():
    """Test extracting summary from content_module_2195.json"""
    content_file = Path(__file__).parent / "output" / "content_module_2195.json"
    
    with open(content_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    # Extract summary
    summary_data = content.get("interactiveVideo", {}).get("summary", {})
    
    if not summary_data:
        print("❌ No summary found in content")
        return
    
    task_params = summary_data.get("task", {}).get("params", {})
    intro = task_params.get("intro", "").strip()
    summaries = task_params.get("summaries", [])
    
    print("=== Summary Data ===")
    print(f"Intro: {intro[:100]}..." if len(intro) > 100 else f"Intro: {intro}")
    print(f"Statement Groups: {len(summaries)}")
    print()
    
    statement_groups = []
    for i, summary_group in enumerate(summaries, 1):
        statements = summary_group.get("summary", [])
        print(f"Group {i}: {len(statements)} statements")
        for j, stmt in enumerate(statements):
            clean = stmt.strip()
            print(f"  [{j+1}] {clean[:80]}..." if len(clean) > 80 else f"  [{j+1}] {clean}")
        print()
        
        if statements:
            clean_statements = [s.strip() for s in statements if s.strip()]
            statement_groups.append(clean_statements)
    
    # Create Summary object
    summary_obj = Summary(
        type="H5P.Summary",
        intro=intro if intro else "Zusammenfassung",
        statement_groups=statement_groups
    )
    
    print("\n=== Summary Object to_text() ===")
    print(summary_obj.to_text())

if __name__ == "__main__":
    test_summary_extraction()
