"""Tokens, password hashes, classroom codes, and the Principal rules.

Pure functions; no database, no HTTP. Each guarantee here is something the
routes and guards assume rather than check again.
"""

from __future__ import annotations

import re
import unittest

from app.auth import codes, passwords, tokens
from app.auth.principal import ANONYMOUS, Participant, Principal, StaffUser

COURSE_A = "css-360-winter-2026-a7rp"
COURSE_B = "css-350-spring-2026-n3h9"


class TokenTests(unittest.TestCase):
    def test_tokens_are_long_random_and_url_safe(self) -> None:
        first, second = tokens.generate_token(), tokens.generate_token()
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(len(first), 43)
        self.assertTrue(tokens.is_plausible_token(first))
        self.assertRegex(first, r"^[A-Za-z0-9_-]+$")

    def test_hash_is_sha256_hex_and_deterministic(self) -> None:
        token = tokens.generate_token()
        self.assertEqual(tokens.hash_token(token), tokens.hash_token(token))
        self.assertRegex(tokens.hash_token(token), r"^[0-9a-f]{64}$")
        self.assertNotEqual(tokens.hash_token(token), tokens.hash_token(token + "x"))

    def test_implausible_values_are_rejected_before_any_lookup(self) -> None:
        for value in ("", "short", "has spaces " * 4, None, 42, "a" * 200, "bad/char" * 5):
            with self.subTest(value=value):
                self.assertFalse(tokens.is_plausible_token(value))

    def test_hashes_match_is_a_comparison(self) -> None:
        digest = tokens.hash_token("t")
        self.assertTrue(tokens.hashes_match(digest, digest))
        self.assertFalse(tokens.hashes_match(digest, digest[::-1]))


class PasswordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        stored = passwords.hash_password("correct horse battery")
        self.assertTrue(stored.startswith("scrypt$14$8$5$"))
        self.assertTrue(passwords.verify_password("correct horse battery", stored))
        self.assertFalse(passwords.verify_password("correct horse batterx", stored))

    def test_same_password_hashes_differently_each_time(self) -> None:
        self.assertNotEqual(
            passwords.hash_password("same password here"),
            passwords.hash_password("same password here"),
        )

    def test_malformed_or_foreign_hashes_never_verify(self) -> None:
        for stored in ("", "plaintext", "scrypt$x$8$5$abc$def", "bcrypt$…", "scrypt$14$8$5$$"):
            with self.subTest(stored=stored):
                self.assertFalse(passwords.verify_password("anything at all", stored))
                self.assertTrue(passwords.needs_rehash(stored))

    def test_current_parameters_do_not_need_rehash_but_weaker_ones_do(self) -> None:
        fresh = passwords.hash_password("a perfectly fine password")
        self.assertFalse(passwords.needs_rehash(fresh))
        weaker = fresh.replace("scrypt$14$", "scrypt$13$", 1)
        self.assertTrue(passwords.needs_rehash(weaker))

    def test_oversized_passwords_are_refused_cheaply(self) -> None:
        stored = passwords.hash_password("a perfectly fine password")
        self.assertFalse(passwords.verify_password("x" * 10_000, stored))

    def test_password_problem_is_about_length_only(self) -> None:
        self.assertIsNotNone(passwords.password_problem("short"))
        self.assertIsNotNone(passwords.password_problem(" padded password "))
        self.assertIsNotNone(passwords.password_problem("x" * 300))
        self.assertIsNotNone(passwords.password_problem(None))
        self.assertIsNone(passwords.password_problem("twelve chars!"))
        self.assertIsNone(passwords.password_problem("no digits or symbols needed"))


class ClassroomCodeTests(unittest.TestCase):
    def test_generated_codes_use_the_unambiguous_alphabet(self) -> None:
        for _ in range(200):
            code = codes.generate_code()
            self.assertEqual(len(code), codes.CODE_LENGTH)
            self.assertTrue(set(code) <= set(codes.CODE_ALPHABET))
            self.assertIsNone(re.search(r"[01IOL]", code))

    def test_alphabet_has_no_look_alikes(self) -> None:
        for confusable in "0O1IL":
            self.assertNotIn(confusable, codes.CODE_ALPHABET)
        self.assertEqual(len(set(codes.CODE_ALPHABET)), len(codes.CODE_ALPHABET))

    def test_normalisation_is_forgiving_about_typing_and_pasting(self) -> None:
        for raw in (
            "7K4P9X",
            "7k4p9x",
            " 7k4 p9x ",
            "7K4-P9X",
            "7K4_P9X",
            "https://aiswe.uwb.edu/join/7k4p9x",
            "https://aiswe.uwb.edu/join/7K4P9X?utm=1",
            "/join/7K4P9X/",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(codes.normalize_code(raw), "7K4P9X")

    def test_normalisation_is_strict_about_the_result(self) -> None:
        for raw in ("7K4P9", "7K4P9XY", "7K4P0X", "7K4PIX", "7K4PLX", "", None, 123, "7K4P9%"):
            with self.subTest(raw=raw):
                self.assertIsNone(codes.normalize_code(raw))

    def test_is_code(self) -> None:
        self.assertTrue(codes.is_code("7K4P9X"))
        self.assertFalse(codes.is_code("7k4p9x"))


def professor(*course_ids: str) -> Principal:
    return Principal(
        user=StaffUser(
            user_id="u-prof",
            email="prof@uw.edu",
            display_name="Prof",
            role="professor",
            course_ids=frozenset(course_ids),
        )
    )


ADMIN = Principal(
    user=StaffUser(user_id="u-admin", email="admin@uw.edu", display_name="Admin", role="admin")
)


def participant(course_id: str) -> Principal:
    return Principal(participant=Participant(participant_id="p-1", course_id=course_id))


class PrincipalTests(unittest.TestCase):
    def test_anonymous_has_nothing(self) -> None:
        self.assertTrue(ANONYMOUS.is_anonymous)
        self.assertFalse(ANONYMOUS.is_staff)
        self.assertFalse(ANONYMOUS.can_staff_course(COURSE_A))
        self.assertFalse(ANONYMOUS.can_access_course(COURSE_A))
        self.assertEqual(ANONYMOUS.staff_course_ids(), frozenset())

    def test_admin_may_staff_every_course(self) -> None:
        self.assertTrue(ADMIN.is_admin)
        self.assertTrue(ADMIN.can_staff_course(COURSE_A))
        self.assertTrue(ADMIN.can_staff_course(COURSE_B))
        self.assertTrue(ADMIN.can_access_course(COURSE_B))
        self.assertIsNone(ADMIN.staff_course_ids())

    def test_professor_is_limited_to_memberships(self) -> None:
        prof = professor(COURSE_A)
        self.assertTrue(prof.is_professor)
        self.assertFalse(prof.is_admin)
        self.assertTrue(prof.can_staff_course(COURSE_A))
        self.assertFalse(prof.can_staff_course(COURSE_B))
        self.assertFalse(prof.can_access_course(COURSE_B))
        self.assertEqual(prof.staff_course_ids(), frozenset({COURSE_A}))
        self.assertFalse(professor().can_staff_course(COURSE_A))

    def test_participant_reaches_only_its_own_course(self) -> None:
        student = participant(COURSE_A)
        self.assertFalse(student.is_staff)
        self.assertTrue(student.can_access_course(COURSE_A))
        self.assertFalse(student.can_access_course(COURSE_B))
        self.assertFalse(student.can_staff_course(COURSE_A))
        self.assertIsNotNone(student.participant_for(COURSE_A))
        self.assertIsNone(student.participant_for(COURSE_B))

    def test_a_professor_who_also_joined_as_a_student_keeps_both(self) -> None:
        both = Principal(
            user=professor(COURSE_A).user,
            participant=Participant(participant_id="p-9", course_id=COURSE_B),
        )
        self.assertTrue(both.can_staff_course(COURSE_A))
        self.assertTrue(both.can_access_course(COURSE_B))
        self.assertFalse(both.can_staff_course(COURSE_B))
        self.assertIsNone(both.participant_for(COURSE_A))


if __name__ == "__main__":
    unittest.main()
