# class RecursiveCharacterTextSplitter():
#     ''' Langchains recursive splitter.
#     Splitting text by recursively look at characters.

#     Recursively tries to split by different characters to find one
#     that works.
#     '''

#     def __init__(
#         self,
#         separators: Optional[List[str]] = None,
#         keep_separator: bool = True,
#         is_separator_regex: bool = False,
#         **kwargs: Any,
#     ) -> None:
#         '''Create a new TextSplitter.'''
#         super().__init__(keep_separator=keep_separator, **kwargs)
#         self._separators = separators or ['\n\n', '\n', ' ', '']
#         self._is_separator_regex = is_separator_regex

#     def _split_text(self, text: str, separators: List[str]) -> List[str]:
#         '''Split incoming text and return chunks.'''
#         final_chunks = []
#         # Get appropriate separator to use
#         separator = separators[-1]
#         new_separators = []
#         for i, _s in enumerate(separators):
#             _separator = _s if self._is_separator_regex else re.escape(_s)
#             if _s == '':
#                 separator = _s
#                 break
#             if re.search(_separator, text):
#                 separator = _s
#                 new_separators = separators[i + 1 :]
#                 break

#         _separator = separator if self._is_separator_regex else re.escape(separator)
#         splits = _split_text_with_regex(text, _separator, self._keep_separator)

#         # Now go merging things, recursively splitting longer texts.
#         _good_splits = []
#         _separator = '' if self._keep_separator else separator
#         for s in splits:
#             if self._length_function(s) < self._chunk_size:
#                 _good_splits.append(s)
#             else:
#                 if _good_splits:
#                     merged_text = self._merge_splits(_good_splits, _separator)
#                     final_chunks.extend(merged_text)
#                     _good_splits = []
#                 if not new_separators:
#                     final_chunks.append(s)
#                 else:
#                     other_info = self._split_text(s, new_separators)
#                     final_chunks.extend(other_info)
#         if _good_splits:
#             merged_text = self._merge_splits(_good_splits, _separator)
#             final_chunks.extend(merged_text)
#         return final_chunks

#     def split_text(self, text: str) -> List[str]:
#         return self._split_text(text, self._separators)
