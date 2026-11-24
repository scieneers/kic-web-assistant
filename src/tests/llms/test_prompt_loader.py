"""Unit tests for the prompt loader.

Covers:
- Loading existing prompts by name (without .txt extension)
- FileNotFoundError for missing prompts
- Return type is a non-empty string
"""

import pytest
from pathlib import Path

from src.llm.prompts.prompt_loader import load_prompt, BASE_PROMPT_DIR


KNOWN_PROMPTS = [
    "system_prompt",
    "router_prompt",
    "socratic_core",
    "socratic_diagnose",
    "socratic_explain",
    "socratic_hinting",
    "socratic_stuckness_goal_achievement",
    "language_detector_prompt",
    "contextualizer_prompt",
]


class TestLoadPrompt:
    @pytest.mark.parametrize("prompt_name", KNOWN_PROMPTS)
    def test_known_prompt_returns_string(self, prompt_name):
        result = load_prompt(prompt_name)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_missing_prompt_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("this_prompt_does_not_exist_xyz")

    def test_error_message_contains_path(self):
        with pytest.raises(FileNotFoundError, match="this_prompt_does_not_exist_xyz"):
            load_prompt("this_prompt_does_not_exist_xyz")

    def test_system_prompt_is_non_trivial(self):
        result = load_prompt("system_prompt")
        assert len(result) > 50

    def test_base_prompt_dir_points_to_prompts_folder(self):
        assert BASE_PROMPT_DIR.is_dir()
        assert (BASE_PROMPT_DIR / "system_prompt.txt").is_file()
