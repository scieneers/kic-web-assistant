"""
Datenmodell für Moodle Resource Module.

Ein Resource-Modul repräsentiert eine herunterladbare Datei (PDF, DOCX, etc.).
"""

import logging
import tempfile
import unicodedata
import zipfile
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from src.env import env


class ResourceFileType(StrEnum):
    """Unterstützte Dateitypen für Resource-Module."""
    PDF = "pdf"
    HTML = "html"
    ZIP = "zip"
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    TXT = "txt"
    # Erweiterbar für andere Formate:
    # DOCX = "docx"
    # PPTX = "pptx"
    

class Resource(BaseModel):
    """
    Repräsentiert eine herunterladbare Ressource aus einem Moodle Resource-Modul.
    
    Attributes:
        filename: Name der Datei (z.B. "document.pdf")
        fileurl: Download-URL der Datei
        mimetype: Dateityp/Extension (z.B. "pdf", "docx") - aus DownloadableContent.type
        filesize: Größe der Datei in Bytes
        extracted_text: Extrahierter Textinhalt (nur bei unterstützten Formaten)
    """
    
    filename: str
    fileurl: str
    mimetype: str  # Dateiendung wie "pdf", "docx", etc. (von DownloadableContent.type)
    filesize: int | None = None
    extracted_text: str | None = None  # Text aus PDF oder anderen Formaten
    
    @property
    def is_pdf(self) -> bool:
        """Prüft, ob die Resource eine PDF-Datei ist."""
        return self.mimetype.lower() == ResourceFileType.PDF
    
    @property
    def is_html(self) -> bool:
        """Prüft, ob die Resource eine HTML-Datei ist."""
        return self.mimetype.lower() == ResourceFileType.HTML
    
    @property
    def is_zip(self) -> bool:
        """Prüft, ob die Resource eine ZIP-Datei ist."""
        return self.mimetype.lower() == ResourceFileType.ZIP
    
    @property
    def is_audio(self) -> bool:
        """Prüft, ob die Resource eine Audio-Datei ist."""
        return self.mimetype.lower() in [ResourceFileType.WAV, ResourceFileType.MP3, ResourceFileType.M4A]
    
    @property
    def is_txt(self) -> bool:
        """Prüft, ob die Resource eine TXT-Datei ist."""
        return self.mimetype.lower() == ResourceFileType.TXT
    
    @property
    def is_supported(self) -> bool:
        """
        Prüft, ob der Dateityp für Textextraktion unterstützt wird.
        
        Unterstützt: PDF, HTML, ZIP, Audio (WAV, MP3, M4A), TXT
        """
        return self.is_pdf or self.is_html or self.is_zip or self.is_audio or self.is_txt
    
    def __str__(self) -> str:
        """String-Repräsentation für Document-Generierung."""
        if self.extracted_text:
            return f"Datei: {self.filename}\n\n{self.extracted_text}"
        return f"Datei: {self.filename}"
    
    def extract_from_bytes(self, file_bytes: bytes, logger: logging.Logger) -> str:
        """
        Extrahiert Text aus Datei-Bytes basierend auf Dateityp.
        
        Args:
            file_bytes: Dateiinhalt als Bytes
            logger: Logger für Statusmeldungen
            
        Returns:
            str: Extrahierter Text oder leerer String
        """
        if self.is_pdf:
            return self._extract_pdf(file_bytes, logger)
        elif self.is_html:
            return self._extract_html(file_bytes, logger)
        elif self.is_audio:
            if not env.AUDIO_TRANSCRIPTION_ENABLED:
                logger.debug(f"Audio-Transkription deaktiviert, überspringe {self.filename}")
                return ""
            return self._extract_audio(file_bytes, logger)
        elif self.is_txt:
            return self._extract_txt(file_bytes, logger)
        elif self.is_zip:
            return self._extract_zip(file_bytes, logger)
        else:
            logger.warning(f"Dateityp {self.mimetype} wird nicht unterstützt")
            return ""
    
    def _extract_pdf(self, pdf_bytes: bytes, logger: logging.Logger) -> str:
        """Extrahiert Text aus PDF."""
        from src.loaders.pdf import PDF
        
        try:
            pdf = PDF()
            text = pdf.extract_text_from_bytes(pdf_bytes, filename=self.filename)
            if text:
                logger.info(f"PDF {self.filename}: {len(text)} Zeichen extrahiert")
                return text
            else:
                logger.warning(f"PDF {self.filename} enthält keinen extrahierbaren Text")
                return ""
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren von PDF {self.filename}: {e}")
            return f"[PDF-Extraktionsfehler: {str(e)}]"
    
    def _extract_html(self, html_bytes: bytes, logger: logging.Logger) -> str:
        """Extrahiert Text aus HTML."""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_bytes, "html.parser")
            text = soup.get_text("\n")
            text = unicodedata.normalize("NFKD", text)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if text:
                logger.info(f"HTML {self.filename}: {len(text)} Zeichen extrahiert")
                return text
            else:
                logger.warning(f"HTML {self.filename} enthält keinen Text")
                return ""
        except Exception as e:
            logger.error(f"Fehler beim Parsen von HTML {self.filename}: {e}")
            return f"[HTML-Parse-Fehler: {str(e)}]"
    
    def _extract_audio(self, audio_bytes: bytes, logger: logging.Logger) -> str:
        """Extrahiert Text aus Audio via Vosk-Transkription."""
        from src.loaders.audio import Audio
        
        try:
            audio = Audio()
            text = audio.extract_text_from_bytes(
                audio_bytes,
                filename=self.filename,
                mimetype=self.mimetype
            )
            # Drop heavy Vosk model ASAP
            try:
                del audio
            except Exception:
                pass
            if text:
                logger.info(f"Audio {self.filename} transkribiert")
                return f"Transkribiertes Audio:\n{text}"
            else:
                logger.warning(f"Audio {self.filename} ergab leere Transkription")
                return ""
        except Exception as e:
            logger.error(f"Fehler beim Transkribieren von Audio {self.filename}: {e}")
            return f"[Audio-Transkriptionsfehler: {str(e)}]"
    
    def _extract_txt(self, txt_bytes: bytes, logger: logging.Logger) -> str:
        """Extrahiert Text aus TXT-Datei."""
        try:
            # UTF-8 Dekodierung
            text = txt_bytes.decode('utf-8')
            text = unicodedata.normalize("NFKD", text)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if text:
                logger.info(f"TXT {self.filename}: {len(text)} Zeichen extrahiert")
                return text
            else:
                logger.warning(f"TXT {self.filename} ist leer")
                return ""
        except UnicodeDecodeError:
            # Fallback zu Latin-1
            try:
                text = txt_bytes.decode('latin-1')
                text = unicodedata.normalize("NFKD", text)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text = '\n'.join(lines)
                logger.info(f"TXT {self.filename} mit Latin-1 dekodiert")
                return text
            except Exception as e:
                logger.error(f"Fehler beim Dekodieren von TXT {self.filename}: {e}")
                return f"[TXT-Dekodierungsfehler: {str(e)}]"
    
    def _extract_zip(self, zip_bytes: bytes, logger: logging.Logger) -> str:
        """Extrahiert Text aus ZIP-Archiv (rekursiv)."""
        from src.loaders.audio import Audio
        from src.loaders.pdf import PDF
        from bs4 import BeautifulSoup
        
        zip_texts = [f"ZIP-Archiv: {self.filename}"]
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / self.filename
                with open(zip_path, 'wb') as f:
                    f.write(zip_bytes)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                for file_path in Path(tmpdir).rglob('*'):
                    if file_path.is_file() and file_path != zip_path:
                        file_ext = file_path.suffix.lower().lstrip('.')
                        relative_path = file_path.relative_to(tmpdir)
                        
                        # PDF in ZIP
                        if file_ext == 'pdf':
                            try:
                                pdf = PDF()
                                with open(file_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                text = pdf.extract_text_from_bytes(pdf_bytes, filename=str(relative_path))
                                if text:
                                    zip_texts.append(f"\n--- PDF: {relative_path} ---\n{text}")
                            except Exception as e:
                                logger.warning(f"PDF-Fehler in ZIP {relative_path}: {e}")
                        
                        # HTML in ZIP
                        elif file_ext in ['html', 'htm']:
                            try:
                                with open(file_path, 'rb') as f:
                                    html_content = f.read()
                                soup = BeautifulSoup(html_content, "html.parser")
                                text = soup.get_text("\n")
                                text = unicodedata.normalize("NFKD", text)
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                text = '\n'.join(lines)
                                if text:
                                    zip_texts.append(f"\n--- HTML: {relative_path} ---\n{text}")
                            except Exception as e:
                                logger.warning(f"HTML-Fehler in ZIP {relative_path}: {e}")
                        
                        # Audio in ZIP
                        elif file_ext in ['wav', 'mp3', 'm4a']:
                            if not env.AUDIO_TRANSCRIPTION_ENABLED:
                                logger.debug(f"Audio-Transkription deaktiviert, überspringe {relative_path}")
                                continue
                            try:
                                audio = Audio()
                                with open(file_path, 'rb') as f:
                                    audio_bytes = f.read()
                                text = audio.extract_text_from_bytes(audio_bytes, filename=str(relative_path), mimetype=f"audio/{file_ext}")
                                if text:
                                    zip_texts.append(f"\n--- Audio: {relative_path} ---\n{text}")
                            except Exception as e:
                                logger.warning(f"Audio-Fehler in ZIP {relative_path}: {e}")
                        
                        # TXT in ZIP
                        elif file_ext == 'txt':
                            try:
                                with open(file_path, 'rb') as f:
                                    txt_content = f.read()
                                text = txt_content.decode('utf-8')
                                text = unicodedata.normalize("NFKD", text)
                                lines = [line.strip() for line in text.split('\n') if line.strip()]
                                text = '\n'.join(lines)
                                if text:
                                    zip_texts.append(f"\n--- TXT: {relative_path} ---\n{text}")
                            except Exception as e:
                                logger.warning(f"TXT-Fehler in ZIP {relative_path}: {e}")
                        
                        # Nicht-unterstützte Dateitypen
                        else:
                            zip_texts.append(f"\nDatei {relative_path} nicht unterstützt und daher nicht verarbeitet.")
            
            result = '\n'.join(zip_texts)
            logger.info(f"ZIP {self.filename}: {len(zip_texts)-1} Dateien extrahiert")
            return result
            
        except Exception as e:
            logger.error(f"ZIP-Verarbeitungsfehler {self.filename}: {e}")
            return f"[ZIP-Verarbeitungsfehler: {str(e)}]"
