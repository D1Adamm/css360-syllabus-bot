import unittest

from app.syllabus_chunking import (
    MAX_CHUNK_CHARS,
    OVERLAP_CHARS,
    chunk_syllabus_text,
    is_likely_heading,
)


class SyllabusChunkingTests(unittest.TestCase):
    def test_heading_preserving_chunks(self) -> None:
        text = (
            "Attendance\n\n"
            "Students who miss class should submit the absence form before the session begins. "
            "Repeated absences may affect participation marks and project progress.\n\n"
            "Office Hours\n\n"
            "Office hours are available on Monday and Wednesday afternoons for project support "
            "and clarifying syllabus questions that were not covered in class."
        )
        chunks = chunk_syllabus_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_title, "Attendance")
        self.assertTrue(chunks[0].text.startswith("Attendance"))
        office = next(chunk for chunk in chunks if chunk.section_title == "Office Hours")
        self.assertIn("Office Hours", office.text)

    def test_long_section_splitting_and_overlap(self) -> None:
        sentence = (
            "Students must follow course policies carefully when submitting assignments late. "
        )
        # Build a long section without extra headings.
        body = sentence * 40
        text = f"Late Policy\n\n{body}"
        chunks = chunk_syllabus_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), MAX_CHUNK_CHARS + 50)
            self.assertEqual(chunk.section_title, "Late Policy")
            self.assertTrue(chunk.text.strip())

        # Adjacent long-section chunks should share overlap content.
        if len(chunks) >= 2:
            first_tail = chunks[0].text[-OVERLAP_CHARS:]
            # Overlap is approximate; ensure later chunk still starts with heading + shared words.
            self.assertTrue(chunks[1].text.startswith("Late Policy"))
            shared_token = "submitting assignments late"
            self.assertIn(shared_token, chunks[0].text)
            self.assertIn(shared_token, chunks[1].text)
            self.assertTrue(first_tail)

    def test_no_empty_chunks_and_stable_ordering(self) -> None:
        text = (
            "Course Information\n\n"
            "This course surveys core topics with enough detail for retrieval quality checks.\n\n"
            "Grading\n\n"
            "Grades combine projects, quizzes, and participation with published weights.\n\n"
            "General paragraph without a detected heading keeps flowing under General or prior "
            "section and remains ordered with the rest of the syllabus content consistently."
        )
        chunks = chunk_syllabus_text(text)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.text.strip() for chunk in chunks))
        orders = [chunk.order for chunk in chunks]
        self.assertEqual(orders, sorted(orders))
        ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(ids, [f"chunk-{index:03d}" for index in orders])

    def test_chunk_size_limits(self) -> None:
        paragraph = ("Policy details continue with clear sentences for chunk boundaries. ") * 30
        text = f"Policies\n\n{paragraph}"
        chunks = chunk_syllabus_text(text)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= MAX_CHUNK_CHARS + 80 for chunk in chunks))

    def test_detects_likely_headings(self) -> None:
        self.assertTrue(is_likely_heading("Attendance"))
        self.assertTrue(is_likely_heading("1. Course Goals"))
        self.assertFalse(is_likely_heading("Students must attend every required lab session."))


if __name__ == "__main__":
    unittest.main()
