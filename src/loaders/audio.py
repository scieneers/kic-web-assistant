# DEPENDENCIES: download vosk models from https://alphacephei.com/vosk/models (vosk-model-de-0.21)
# DEPENDENCIES: requires ffmpeg installed and accessible in PATH

"""
Audio transcription extractor for WAV, MP3, M4A files.

Uses Vosk offline speech recognition for extracting text from audio files.
Vosk requires pre-downloaded language models stored locally.
"""

import json
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path


class Audio:
    """
    Audio transcription extractor for extracting text from audio files.
    
    Supports WAV, MP3, M4A formats using Vosk offline speech recognition.
    Requires Vosk language models to be downloaded and stored locally.
    
    Model Setup:
    1. Download German model from: https://alphacephei.com/vosk/models
       Recommended: vosk-model-de-0.21 (small, ~44MB) or vosk-model-small-de-0.15
    2. Extract to a folder (e.g., ./vosk-models/vosk-model-de-0.21)
    3. Set MODEL_PATH to the extracted model directory
    
    Returns transcribed text that can be integrated into Module.to_document().
    """

    # Path to the Vosk model directory
    # You can download models from: https://alphacephei.com/vosk/models
    # Recommended for German: vosk-model-de-0.21 or vosk-model-small-de-0.15
    MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "src/loaders/models/vosk_model/vosk-model-small-de-0.15")

    def __init__(self) -> None:
        self.logger = logging.getLogger("loader.audio")

        # Imported lazily: vosk is a Linux-only dependency (see pyproject.toml),
        # so importing this module must not require it. It is only needed once an
        # Audio instance is actually constructed to transcribe a file.
        from vosk import Model

        # Load Vosk model
        if not os.path.exists(self.MODEL_PATH):
            self.logger.error(
                f"Vosk model not found at {self.MODEL_PATH}. "
                "Please download a model from https://alphacephei.com/vosk/models "
                "and extract it to the MODEL_PATH location."
            )
            raise FileNotFoundError(f"Vosk model not found: {self.MODEL_PATH}")
        
        self.logger.info(f"Loading Vosk model from {self.MODEL_PATH}")
        self.model = Model(self.MODEL_PATH)

    def _convert_to_wav(self, audio_bytes: bytes, source_filename: str) -> bytes:
        """
        Convert audio file to WAV format (required by Vosk).
        Uses ffmpeg for conversion.
        
        Args:
            audio_bytes: Original audio file bytes
            source_filename: Original filename (for extension detection)
            
        Returns:
            bytes: WAV file bytes (16kHz, mono, 16-bit PCM)
        """
        with tempfile.NamedTemporaryFile(suffix=Path(source_filename).suffix, delete=False) as input_file:
            input_file.write(audio_bytes)
            input_path = input_file.name
        
        output_path = input_path.replace(Path(source_filename).suffix, ".wav")
        
        try:
            # Convert to WAV with ffmpeg: 16kHz, mono, 16-bit PCM
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",       # mono
                "-f", "wav",      # WAV format
                "-y",             # overwrite
                output_path
            ], check=True, capture_output=True)
            
            # Read converted WAV
            with open(output_path, 'rb') as f:
                wav_bytes = f.read()
            
            return wav_bytes
            
        finally:
            # Cleanup temp files
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def extract_text_from_bytes(self, audio_bytes: bytes, filename: str, mimetype: str) -> str:
        """
        Extract text from audio file bytes using Vosk speech recognition.
        
        Args:
            audio_bytes: Audio file content as bytes
            filename: Audio filename for logging
            mimetype: MIME type (audio/wav, audio/mp3, etc.)
            
        Returns:
            str: Transcribed text from the audio file
        """
        self.logger.info(f"Transcribing audio file: {filename} ({mimetype})")
        
        try:
            # Convert to WAV if necessary (Vosk requires WAV)
            if not (mimetype == "audio/wav" or filename.lower().endswith('.wav')):
                self.logger.info(f"Converting {filename} to WAV format")
                wav_bytes = self._convert_to_wav(audio_bytes, filename)
            else:
                wav_bytes = audio_bytes
            
            # Write to temp file for wave module
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(wav_bytes)
                temp_wav_path = temp_wav.name
            
            try:
                # Open WAV file
                with wave.open(temp_wav_path, "rb") as wf:
                    # Verify format
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
                        self.logger.warning(
                            f"Audio format may not be optimal: "
                            f"channels={wf.getnchannels()}, "
                            f"samplewidth={wf.getsampwidth()}, "
                            f"framerate={wf.getframerate()}"
                        )
                    
                    # Create recognizer (vosk is Linux-only; import lazily)
                    from vosk import KaldiRecognizer

                    rec = KaldiRecognizer(self.model, wf.getframerate())
                    rec.SetWords(True)
                    
                    # Process audio
                    transcription_parts = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            if result.get("text"):
                                transcription_parts.append(result["text"])
                    
                    # Get final result
                    final_result = json.loads(rec.FinalResult())
                    if final_result.get("text"):
                        transcription_parts.append(final_result["text"])
                    
                    transcribed_text = " ".join(transcription_parts).strip()
                    
                    if transcribed_text:
                        self.logger.info(
                            f"Successfully transcribed {filename}: "
                            f"{len(transcribed_text)} characters"
                        )
                        return transcribed_text
                    else:
                        self.logger.warning(f"Vosk returned empty transcription for {filename}")
                        return ""
                        
            finally:
                # Cleanup temp file
                Path(temp_wav_path).unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Error transcribing audio {filename}: {e}")
            return f"[Fehler bei Audio-Transkription: {str(e)}]"

    def extract_text(self, audio_path: str | Path) -> str:
        """
        Extract text from an audio file path using Vosk speech recognition.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Read file
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Determine mimetype from extension
        ext = audio_path.suffix.lower()
        mimetype_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mp3',
            '.m4a': 'audio/m4a',
            '.mpeg': 'audio/mpeg'
        }
        mimetype = mimetype_map.get(ext, 'audio/unknown')
        
        return self.extract_text_from_bytes(audio_bytes, audio_path.name, mimetype)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    audio = Audio()
    # Sample bytes for demonstration
    sample_bytes = b"fake audio data" * 1000
    
    # NOTE: keep this as a simple manual demo only; do not use print() in ingestion.
    # The correct method is extract_text_from_bytes (extract_metadata_from_bytes does not exist).
    text = audio.extract_text_from_bytes(sample_bytes, "podcast.mp3", "audio/mp3")
    logging.getLogger("loader").info("Demo transcription length=%s", len(text))
