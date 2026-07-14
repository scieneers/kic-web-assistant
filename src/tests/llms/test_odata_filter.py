"""Tests for the OData filter builder used by KiCampusRetriever."""

from src.llm.objects.retriever import _build_odata_filter


class TestModuleIdFilter:
    def test_single_module_id_uses_eq(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=42)
        assert "module_id eq 42" in odata_filter
        assert "search.in(module_id" not in odata_filter

    def test_module_id_list_uses_search_in(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=[1, 33, 102])
        assert "search.in(module_id, '1,33,102', ',')" in odata_filter

    def test_none_module_id_excludes_empty_module_marker(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=None)
        assert "type ne 'EmptyModule'" in odata_filter
        assert "module_id eq" not in odata_filter
        assert "search.in(module_id" not in odata_filter

    def test_empty_module_id_list_behaves_like_none(self):
        with_none = _build_odata_filter(course_id=79, module_id=None)
        with_empty_list = _build_odata_filter(course_id=79, module_id=[])
        assert with_none == with_empty_list

    def test_module_id_list_does_not_exclude_empty_module_marker(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=[1, 33])
        assert "type ne 'EmptyModule'" not in odata_filter


class TestCourseIdFilter:
    def test_single_course_id_uses_eq(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=None)
        assert "course_id eq 79" in odata_filter

    def test_course_id_list_uses_search_in(self):
        odata_filter = _build_odata_filter(course_id=[79, 102], module_id=None)
        assert "search.in(course_id, '79,102', ',')" in odata_filter


class TestCombinedFilter:
    def test_course_list_and_module_list_together(self):
        odata_filter = _build_odata_filter(course_id=[79, 102], module_id=[1, 33])
        assert "search.in(course_id, '79,102', ',')" in odata_filter
        assert "search.in(module_id, '1,33', ',')" in odata_filter


class TestTypeExclusion:
    """QuizItem docs carry correct answers in metadata — the exclusion must be
    UNCONDITIONAL (also with module scope), or the answers leak into the prompt."""

    def test_exclusion_without_scope(self):
        odata_filter = _build_odata_filter(course_id=None, module_id=None)
        assert "not search.in(type, 'ModuleFingerprint,QuizItem', ',')" in odata_filter

    def test_exclusion_with_module_scope(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=42)
        assert "not search.in(type, 'ModuleFingerprint,QuizItem', ',')" in odata_filter

    def test_exclusion_with_module_list_scope(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=[1, 33])
        assert "not search.in(type, 'ModuleFingerprint,QuizItem', ',')" in odata_filter


class TestDocTypeFilter:
    """Typed access path: only the requested document class, scope still applies."""

    def test_doc_type_filters_to_exact_type(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=42, doc_type="QuizItem")
        assert "type eq 'QuizItem'" in odata_filter
        assert "course_id eq 79" in odata_filter
        assert "module_id eq 42" in odata_filter

    def test_doc_type_skips_exclusion_and_empty_module_clause(self):
        odata_filter = _build_odata_filter(course_id=79, module_id=None, doc_type="GlossaryEntry")
        assert "not search.in" not in odata_filter
        assert "type ne 'EmptyModule'" not in odata_filter

    def test_doc_type_without_scope_does_not_restrict_to_drupal(self):
        odata_filter = _build_odata_filter(course_id=None, module_id=None, doc_type="GlossaryEntry")
        assert "source eq 'Drupal'" not in odata_filter
        assert "type eq 'GlossaryEntry'" in odata_filter
