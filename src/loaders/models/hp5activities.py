from pathlib import Path
import re
import json
import zipfile
from typing import Optional

from pydantic import BaseModel, HttpUrl, model_validator


def extract_library_from_h5p(h5p_zip_path: str) -> Optional[str]:
	"""Extrahiert mainLibrary aus h5p.json eines H5P-Packages.
	
	Args:
		h5p_zip_path: Pfad zum H5P ZIP-File
		
	Returns:
		Library-Name (z.B. "H5P.Flashcards") oder None bei Fehler
	"""
	try:
		with zipfile.ZipFile(h5p_zip_path, "r") as zip_ref:
			with zip_ref.open("h5p.json") as f:
				h5p_data = json.load(f)
				return h5p_data.get("mainLibrary", "")
	except Exception:
		return None


def strip_html(text: str) -> str:
    """Entfernt HTML-Tags und dekodiert HTML-Entities."""
    if not text:
        return ""
    # Entferne HTML-Tags
    text = re.sub(r'<[^>]+>', '', text)
    # Ersetze HTML-Entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    # Entferne übermäßige Whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class H5PActivities(BaseModel):
    id: int
    coursemodule: int
    fileurl: HttpUrl
    filename: Path
    intro: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_fileurl(cls, values):
        values["fileurl"] = values["package"][0]["fileurl"]
        values["filename"] = values["package"][0]["filename"]
        return values