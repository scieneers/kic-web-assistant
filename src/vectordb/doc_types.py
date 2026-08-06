"""Shared document-class constants for the Azure AI Search index.

The `type` field is the document-class discriminator. Besides the classic
classes (module, EmptyModule, Kurs, Drupal page types) the index holds
structured ITEM documents — one per quiz question, glossary entry, flashcard,
video transcript or book chapter — emitted by Module.to_item_documents().

Defined here (not in loaders or llm) because both sides depend on vectordb:
the loaders WRITE these types, the retriever FILTERS on them.
"""

QUIZ_ITEM = "QuizItem"
GLOSSARY_ENTRY = "GlossaryEntry"
FLASHCARD = "Flashcard"
TRANSCRIPT = "Transcript"
BOOK_CHAPTER = "BookChapter"

# Every structured item class (used e.g. to keep them out of the frontend
# course/module tree scan).
ITEM_DOC_TYPES = (QUIZ_ITEM, GLOSSARY_ENTRY, FLASHCARD, TRANSCRIPT, BOOK_CHAPTER)

# Types that must NEVER reach normal RAG retrieval. QuizItem documents carry
# the correct answers in their metadata payload — retrieving them would leak
# assessment solutions into the LLM prompt (and thus into chat answers).
# ModuleFingerprint is a legacy bookkeeping type kept for defensive exclusion.
# GlossaryEntry/Transcript/BookChapter/Flashcard deliberately DO participate
# in normal retrieval: they are precise, deep-linkable content sources.
RETRIEVAL_EXCLUDED_TYPES = ("ModuleFingerprint", QUIZ_ITEM)
