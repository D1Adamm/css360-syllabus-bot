import unittest

from app.course_index import _embedding_input
from app.syllabus_chunking import (
    INDEX_VERSION,
    MAX_CHUNK_CHARS,
    OVERLAP_CHARS,
    SyllabusChunk,
    chunk_syllabus_text,
    embedding_input_for_chunk,
    is_likely_heading,
    try_split_inline_heading,
    validate_chunks,
)

# Representative wiki-export style syllabus covering the CSS 360 policy sections.
CSS360_STYLE_SYLLABUS = """
Software Engineering (Fall 2025)

Jump to:navigation, search
CSS360: Software Engineering
Contents
\t•\t1
\t•\tOverview and Learning Objectives
\t•\t2
\t•\tClass format and structure
\t•\t4
\t•\tAssignments
\t•\t5.7
\t•\tAdministrative Notes
\t•\tCredit and Notes
Instructor:\xa0Kaylea Champion
Contact:\xa0Your messages are important to me! I prefer, in this order:
\t1\tpublic messages via Discord (others might have the same question or have the answer)
\t2\tprivate messages via Discord
\t3\tprivate messages via Canvas\xa0Note: all grade-related discussion must happen via Canvas
\t4\temail only if absolutely necessary to kaylea@uw.edu
If you need to have a discussion with me, I would be happy to meet with you via an office hours appointment.

Textbook:
We do not have a required textbook. You will instead engage with material from a wide range of sources.

Course Meetings:
Tuesday / Thursday
3:30 PM to 5:30 PM

Course Websites
\t•\tWe will use Canvas for announcements and turning in assignments.
\t•\tWe will use Discord for chat, including to ask questions.

Overview and Learning Objectives
Software engineering is a collection of practices, philosophies, skills, and strategies for building and maintaining software.
As students of software engineering in the twenty-first century, I expect that many of you taking this course will work in technology roles after graduation.

Class format and structure
In general, the organization of the course adopts a flipped approach where you engage with instructional materials on your own.

Note about this Syllabus
You should expect this syllabus to be a dynamic document (not a 'contract').

Assignments
The assignments in this class are designed to give you an opportunity to try your hand at using the conceptual material taught in the class. We will do one project and have one short writing assignment reflecting on that project and course topics. Any quizzes will be short and low-stakes. There will be no exams.
Unless otherwise noted, all assignments are due at the end of the day.

Grading
\t•\tIn-class activities: 20%
\t•\tProjects: 50%
\t•\tReflection: 5%
Impact of Missing Class\xa0If you must miss class, must be late, or must leave early, file the class absence form to alert me. Not filing this form ('no call, no show') will impact your in-class activities grade. Advance notice lets you avoid a penalty, but it does not serve as a makeup. There is no direct way to make up for missing in-class activities. Instead, you can expect to be called upon more often in subsequent classes to balance out your participation.
Standup participation is equivalent to answering one question in class. You can also receive credit for serving as scrum master.
Grade Questions\xa0Everyone makes mistakes and I want to fix mine as quickly as possible. If you have questions about a grade, book a private timeslot on my calendar within 1 week of the grade being released. In a grade consultation session, I will ask you to take the lead.
Mapping Percentage to the 4.0 Scale\xa0Instructors have discretion in how we make use of the 4.0 grading scale; we select the weight for each element of the course.
Late Policy\xa0With respect to Bot Project 1-7, note that the assignments are due Fridays at 11:59 p.m. Pacific time. Among these 7 projects, you may choose one project for which you want to use one 48-hour extension per quarter, no questions asked. Use Canvas to make your request so that it is private and trackable. I suggest not using your extension unless you cannot avoid it (you might need it later!).
No extension is possible for the Demo and Feedback assignments as those involve coordinating the entire class community and have very tight timing. No extension is possible for the Reflection assignment because it sits at the very end of the quarter and I must submit grades on time.

Administrative Notes
UW Bothell STEM has a set of standard policies for undergraduates.

Office Hours
The best way to get in touch with me about issues in class will in the Discord server via asynchronous messages.

Credit and Notes
This course was inspired by course notes shared by colleagues.
""".strip()


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
        body = sentence * 40
        text = f"Late Policy\n\n{body}"
        chunks = chunk_syllabus_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), MAX_CHUNK_CHARS + 50)
            self.assertEqual(chunk.section_title, "Late Policy")
            self.assertTrue(chunk.text.strip())

        if len(chunks) >= 2:
            first_tail = chunks[0].text[-OVERLAP_CHARS:]
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
        self.assertTrue(is_likely_heading("Late Policy"))
        self.assertTrue(is_likely_heading("Class format and structure"))
        self.assertFalse(is_likely_heading("Students must attend every required lab session."))
        self.assertFalse(is_likely_heading("Jump to:navigation, search"))

    def test_splits_glued_policy_headings(self) -> None:
        self.assertEqual(
            try_split_inline_heading(
                "Late Policy With respect to Bot Project 1-7, note that the assignments are due."
            ),
            (
                "Late Policy",
                "With respect to Bot Project 1-7, note that the assignments are due.",
            ),
        )
        self.assertEqual(
            try_split_inline_heading(
                "Grade Questions Everyone makes mistakes and I want to fix mine quickly."
            )[0],
            "Grade Questions",
        )
        self.assertEqual(
            try_split_inline_heading(
                "Impact of Missing Class If you must miss class, file the absence form."
            )[0],
            "Impact of Missing Class",
        )
        self.assertEqual(
            try_split_inline_heading(
                "Contact: Your messages are important to me! I prefer, in this order:"
            ),
            (
                "Contact",
                "Your messages are important to me! I prefer, in this order:",
            ),
        )


class Css360StyleChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = chunk_syllabus_text(CSS360_STYLE_SYLLABUS)
        self.by_section: dict[str, list[SyllabusChunk]] = {}
        for chunk in self.chunks:
            self.by_section.setdefault(chunk.section_title, []).append(chunk)

    def test_contact_is_distinct_section(self) -> None:
        self.assertIn("Contact", self.by_section)
        contact_text = "\n".join(chunk.text for chunk in self.by_section["Contact"])
        self.assertIn("public messages via Discord", contact_text)
        self.assertIn("grade-related discussion must happen via Canvas", contact_text)

    def test_contact_order_and_canvas_grade_rule_stay_together(self) -> None:
        contact_chunks = self.by_section["Contact"]
        self.assertTrue(
            any(
                "public messages via Discord" in chunk.text
                and "grade-related discussion must happen via Canvas" in chunk.text
                for chunk in contact_chunks
            )
        )

    def test_grade_questions_detected_separately(self) -> None:
        self.assertIn("Grade Questions", self.by_section)
        text = "\n".join(chunk.text for chunk in self.by_section["Grade Questions"])
        self.assertIn("within 1 week", text)

    def test_late_policy_detected_separately(self) -> None:
        self.assertIn("Late Policy", self.by_section)
        text = "\n".join(chunk.text for chunk in self.by_section["Late Policy"])
        self.assertIn("48-hour extension", text)
        self.assertIn("Use Canvas to make your request", text)
        self.assertIn("No extension is possible for the Demo", text)
        self.assertIn("No extension is possible for the Reflection", text)

    def test_impact_of_missing_class_detected_separately(self) -> None:
        self.assertIn("Impact of Missing Class", self.by_section)
        text = "\n".join(chunk.text for chunk in self.by_section["Impact of Missing Class"])
        self.assertIn("absence form", text)
        self.assertIn("Advance notice", text)
        self.assertIn("no direct way to make up", text.lower())
        self.assertIn("Standup participation", text)

    def test_no_exams_retrievable_under_assignments(self) -> None:
        self.assertIn("Assignments", self.by_section)
        text = "\n".join(chunk.text for chunk in self.by_section["Assignments"])
        self.assertIn("There will be no exams", text)

    def test_toc_and_navigation_noise_excluded(self) -> None:
        joined = "\n".join(chunk.text for chunk in self.chunks)
        self.assertNotIn("Jump to:navigation", joined)
        # TOC-only chrome should not dominate early chunks.
        early = "\n".join(chunk.text for chunk in self.chunks[:3])
        self.assertNotIn("•\t1", early)
        self.assertNotIn("•\t4", early)

    def test_document_title_not_assigned_to_nearly_every_chunk(self) -> None:
        document_title = self.chunks[0].document_title
        self.assertEqual(document_title, "Software Engineering (Fall 2025)")
        titled = sum(1 for chunk in self.chunks if chunk.section_title == document_title)
        self.assertEqual(titled, 0)
        self.assertGreaterEqual(len(self.by_section), 8)

    def test_chunk_metadata_includes_document_and_path(self) -> None:
        late = self.by_section["Late Policy"][0]
        payload = late.to_dict()
        self.assertEqual(payload["documentTitle"], "Software Engineering (Fall 2025)")
        self.assertEqual(payload["sectionTitle"], "Late Policy")
        self.assertIn("headingPath", payload)
        self.assertEqual(payload["headingPath"][0], "Late Policy")
        self.assertIn("startOffset", payload)
        self.assertIn("endOffset", payload)

    def test_embeddings_include_meaningful_heading_context(self) -> None:
        late = self.by_section["Late Policy"][0]
        embedded = embedding_input_for_chunk(
            section_title=late.section_title,
            text=late.text,
        )
        self.assertTrue(embedded.startswith("Section: Late Policy\n\n"))
        self.assertIn("48-hour", embedded)
        # course_index helper stays aligned with the shared embedding formatter.
        self.assertEqual(_embedding_input(late), embedded)

    def test_long_sections_split_on_clean_boundaries(self) -> None:
        sentence = "Policy details continue with clear sentences for chunk boundaries. "
        text = f"Policies\n\n{sentence * 40}"
        chunks = chunk_syllabus_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks[:-1]:
            body = chunk.text
            if body.startswith("Policies"):
                body = body[len("Policies") :].lstrip("\n")
            self.assertFalse(body[:1].islower(), msg=chunk.text[:80])
            self.assertTrue(
                chunk.text.rstrip().endswith((".", "!", "?", ":"))
                or "\n\n" in chunk.text[-40:]
                or len(chunk.text) <= MAX_CHUNK_CHARS
            )

    def test_validation_passes_for_css360_style_fixture(self) -> None:
        warnings = validate_chunks(
            self.chunks,
            document_title=self.chunks[0].document_title,
            source_char_count=len(CSS360_STYLE_SYLLABUS),
        )
        self.assertFalse(
            any("document title as sectionTitle" in warning for warning in warnings)
        )
        self.assertEqual(INDEX_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
