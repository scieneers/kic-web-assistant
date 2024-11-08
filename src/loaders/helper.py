import re
import unicodedata
from io import StringIO
from pathlib import Path

import webvtt
from bs4 import BeautifulSoup

TMP_DIR = Path(__file__).parent.parent.parent.joinpath("tmp").resolve()


def convert_vtt_to_text(vtt_buffer: StringIO) -> str:
    try:
        vtt = webvtt.read_buffer(vtt_buffer)
    except webvtt.MalformedFileError as err:
        raise err
    transcript = ""

    lines = []
    for line in vtt:
        # Strip the newlines from the end of the text.
        # Split the string if it has a newline in the middle
        # Add the lines to an array
        lines.extend(line.text.strip().splitlines())

    # Remove repeated lines
    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line

    return transcript


def process_html_summaries(text: str) -> str:
    """remove html tags from summaries, beautify poorly formatted texts"""
    if "<" not in text:
        new_text = text.strip()
    else:
        soup = BeautifulSoup(text, "html.parser")
        for br_tag in soup.find_all("br"):
            br_tag.replace_with(" ")
        new_text = soup.get_text().strip()
    new_text = new_text.replace("\n", " ").replace("\r", "")
    # \r = carriage return character
    new_text = re.sub(r"\s{3,}", "  ", new_text)

    # Normalize parsed text (remove \xa0 from str)
    new_text = unicodedata.normalize("NFKD", new_text)
    return new_text
