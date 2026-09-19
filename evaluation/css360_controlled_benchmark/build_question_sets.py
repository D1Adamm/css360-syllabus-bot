#!/usr/bin/env python3
"""Build the two question files for the controlled benchmark.

    python3 evaluation/css360_controlled_benchmark/build_question_sets.py

`questions_repeat22.json`: the 22 questions of the 2026-09-11 benchmark,
text unchanged, with a `kind` (factual / unanswerable / false_premise) and the
SHA-256 of the file they came from. `questions_heldout.json`: the new set
authored here, each question with a reference answer, key facts, the syllabus
section, and one or more `supportingPassages` that this script verifies are
verbatim excerpts of the syllabus text. Both files get an `overlapJulySplit`
block per question: the nearest question in the July 2026 laptop export (the
candidate for the v2/v3 training data, whose bytes are not retained) and which
key facts appear in its responses. The VM-side check against the actual v4
export happens in the runner.

Refuses to write if a passage is not in the syllabus or a held-out question is
too similar to a repeat-22 question.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_benchmark import EXPORT_DIR, SYLLABUS_FILE, overlap_for_question, similarity, training_rows, verdict_for  # noqa: E402

ORIGINAL = ROOT / "evaluation" / "model_version_benchmark" / "questions.json"
OLD_BANK = ROOT / "evaluation" / "held_out_questions.json"
COURSE_ID = "css-360-winter-2026-a7rp"

REPEAT_KINDS = {"q14": "false_premise", "q19": "unanswerable", "q20": "unanswerable", "q21": "unanswerable", "q22": "unanswerable"}
REPEAT_PREMISES = {"q14": "assumes the course has midterm exams; the syllabus says there will be no exams"}

#: Held-out questions the July-split check flags at REVIEW, each resolved by
#: hand with the reason. An unresolved REVIEW or any REJECT refuses the build.
JULY_RESOLUTIONS = {
    "h03": "shares only the tokens 'bot' and 'due' with 'When is the Bot Feedback due?'; a different task. The Task 4 date words appear in a training response about the Task 3 deadline, so the date is labelled a partial training fact.",
    "h09": "same token overlap with the Bot Feedback question; different task. The Task 7 due date appears in a training response about deploying version 2.1, so the date is a partial training fact; the reason it follows the demo is not in training.",
    "u06": "difflib character ratio only, against an unrelated deadline question; no shared content words.",
    "p03": "shares 'bot feedback' with 'When is the Bot Feedback due?'; that example teaches the due date, not who submits, and the premise being tested is not in training.",
}

HELD_OUT = [
    # ---- factual ----------------------------------------------------------------
    {"id": "h01", "kind": "factual", "category": "course_logistics",
     "question": "How many hours per week does the syllabus say to plan for outside class, in class, and in lab or office hours?",
     "referenceAnswer": "Plan on 10 hours per week of class work outside class, 4 hours per week in class, and as much as 1 hour per week in lab or office hours; it is a 5-credit class.",
     "keyFacts": ["10 hours per week on class work", "4 hours per week in class", "up to 1 hour per week in lab or office hours"],
     "sourceSection": "Class format and structure",
     "supportingPassages": ["This is a 5-credit class; you should plan to spend 10 hours per week working on class work as well as 4 hours per week in class and as much as 1 hour per week in lab or office hours."]},
    {"id": "h02", "kind": "factual", "category": "project_requirements",
     "question": "What three questions does each person answer at standup, and who is responsible for sending the post-standup notes to the instructor?",
     "referenceAnswer": "What you did since the last standup, what you will do before the next standup, and whether you have any blockers. The Scrum Master, a rotating role, pursues blockers and sends the post-standup notes so everyone gets credit.",
     "keyFacts": ["what did you do since the last standup", "what will you do before the next standup", "do you have any blockers", "the Scrum Master sends the post-standup notes"],
     "sourceSection": "Standup Meetings",
     "supportingPassages": ["In standup, you answer three questions: what did you do since the last standup, what will you do before the next standup, and do you have any blockers? One member will take on the role of Scrum Master, a duty that will rotate among group members. All members will update the Kanban board and the Scrum Master will pursue removing any blockers and send me post-standup notes so that everyone gets credit for the standup."]},
    {"id": "h03", "kind": "factual", "category": "deadlines",
     "question": "When is Bot Project Task 4 due, and what does the group have to produce for it?",
     "referenceAnswer": "Friday, October 24, 11:59 p.m. Pacific time. The group writes an architectural guide and adds it to the main repository on GitHub; every member must make at least one substantive commit to it, and there is no separate turn-in because it is graded from GitHub.",
     "keyFacts": ["Friday, October 24 at 11:59 p.m. Pacific", "an architectural guide added to the main GitHub repository", "each member makes at least one substantive commit"],
     "sourceSection": "Bot Project Task #4",
     "supportingPassages": ["Friday, October 24, 11:59 p.m. Pacific time.", "You are done with Task 4 if your group has written an architectural guide and added it to your main repository on GitHub. All group members should have made at least one substantive commit to the guide to receive credit."]},
    {"id": "h04", "kind": "factual", "category": "course_logistics",
     "question": "If a reading is marked \"[Tentative]\" and the instructor has not changed it by the deadline, do I still have to do it?",
     "referenceAnswer": "Yes. Something marked [Tentative] that is not changed before the deadline is assigned. By contrast, a reading marked [To Be Decided] that is not filled in six days before it is due is dropped, and the instructor's rule is not to change readings within six days of when they are due.",
     "keyFacts": ["yes, an unchanged [Tentative] item is assigned", "the six-day rule for changing readings"],
     "sourceSection": "Note about this Syllabus",
     "supportingPassages": ["If I don't change something marked \"[Tentative]\" before the deadline, then it is assigned.", "This means that if I don't fill in a reading marked \"[To Be Decided]\" six days before it's due, it is dropped."]},
    {"id": "h05", "kind": "factual", "category": "project_requirements",
     "question": "How long is each group's live demo slot on Demo Day, and how will presenters get their screen onto the projector?",
     "referenceAnswer": "Each group has a 10-minute slot, in class on Tuesday, November 25. The podium computer drives the projector; presenters present from their own laptop by joining a Zoom room and screensharing.",
     "keyFacts": ["10 minute slot", "join a Zoom room and screenshare from your own laptop", "the podium computer drives the projector"],
     "sourceSection": "Demonstration of Bot 2.0 - Live Demos",
     "supportingPassages": ["During your 10 minute slot, show off what your bot (and web page) can do.", "To avoid awkwardness around switching laptops, the podium computer will drive the projector. However, you will present from your own computer. To accomplish this, all presenting laptops will join a Zoom room and screenshare to present."]},
    {"id": "h06", "kind": "factual", "category": "project_requirements",
     "question": "For the Task 5 bug and technical-debt analysis, what two kinds of code analysis does the syllabus suggest dividing among group members, and what modeling technique does it recommend for uncovering problems?",
     "referenceAnswer": "Running automated tools on the code, and examining the code for evidence of problems from the readings; it recommends building a threat model (and thinking like a bad actor). Findings go into a shared code analysis report in the repository and become user stories.",
     "keyFacts": ["running automated tools on your code", "examining your code for evidence of problems from the readings", "consider building a threat model"],
     "sourceSection": "Bot Project Task #5",
     "supportingPassages": ["Running automated tools on your code.", "Examining your code for evidence of problems from our readings.", "Consider building a threat model as a way to uncover potential problems."]},
    {"id": "h07", "kind": "factual", "category": "course_logistics",
     "question": "How long is each class session scheduled to run, and is there a break?",
     "referenceAnswer": "Each session is scheduled to run for a maximum of 120 minutes, with a break halfway through; the instructor's goal is always to end a bit early.",
     "keyFacts": ["a maximum of 120 minutes", "a break halfway through"],
     "sourceSection": "Class format and structure",
     "supportingPassages": ["Each session is scheduled to run for a maximum of 120 minutes; we'll take a break half way through -- it's always my goal to end a bit early."]},
    {"id": "h08", "kind": "factual", "category": "course_logistics",
     "question": "Who is the guest speaker on October 28, and what is that session's topic?",
     "referenceAnswer": "Tiffany Chen, Software Engineer at ExtraHop Networks; the session is Security and Threat Modeling, with Reading Note 9, the Threat Modeling Manifesto, and a threat modeling activity.",
     "keyFacts": ["Tiffany Chen", "ExtraHop Networks", "Security and Threat Modeling"],
     "sourceSection": "Schedule, October 28",
     "supportingPassages": ["October 28 (Tuesday) -- Security and Threat Modeling", "Guest Speaker: Tiffany Chen, Software Engineer at ExtraHop Networks"]},
    {"id": "h09", "kind": "factual", "category": "deadlines",
     "question": "When is Bot Project Task 7 due, and why is it scheduled after the demo?",
     "referenceAnswer": "Thursday, December 11, 11:59 p.m. Pacific Time, after the demo because version 2.1 incorporates feedback from the demo. It falls in finals week, so the turn-in is screenshots of each new feature rather than a live demo.",
     "keyFacts": ["Thursday, December 11 at 11:59 p.m. Pacific", "it comes after the demo so you can incorporate feedback from the demo"],
     "sourceSection": "Bot Project Task #7",
     "supportingPassages": ["Thursday, December 11, 11:59 p.m. Pacific Time (Note that this is AFTER the demo because you will be incorporating feedback from the demo.)"]},
    {"id": "h10", "kind": "factual", "category": "course_logistics",
     "question": "Which chapter of The Mythical Man-Month is assigned before the September 30 class session?",
     "referenceAnswer": "Chapter 1 of The Mythical Man-Month, alongside Reading Note 1, the lecture video on organizing patterns, and the Agile Manifesto with its 12 Principles.",
     "keyFacts": ["Chapter 1 of The Mythical Man-Month"],
     "sourceSection": "Schedule, September 30",
     "supportingPassages": ["Read Chapter 1 from The Mythical Man Month", "September 30 (Tuesday): Organizing Patterns: Agile, Waterfall, and All That"]},
    # ---- unanswerable -------------------------------------------------------------
    {"id": "u01", "kind": "unanswerable", "category": "project_requirements",
     "question": "What is the maximum number of students allowed in one bot project group?",
     "referenceAnswer": "The syllabus does not state a group size. It says groups are assigned and announced on September 30, that the project is a group project to build a Discord bot, and that the instructor may rearrange group membership; no size limit is given.",
     "keyFacts": ["says the syllabus gives no group size", "does not invent a number"],
     "sourceSection": "Projects / Notes on Group Work (not specified)",
     "supportingPassages": ["Groups assigned and announced.", "I will be collecting confidential feedback on how groups are functioning, and I reserve the right to rearrange group membership to support everyone's learning."]},
    {"id": "u02", "kind": "unanswerable", "category": "grading",
     "question": "Does the syllabus give extra credit for finishing the final bot release early?",
     "referenceAnswer": "No extra credit is mentioned anywhere in the syllabus. For Task 7, the final release (version 2.1), it says only that a group might want to submit early to boost its finals-week flexibility.",
     "keyFacts": ["says no extra credit is specified", "does not invent points or a bonus"],
     "sourceSection": "Bot Project Task #7 (not specified)",
     "supportingPassages": ["Your group might want to submit this early to boost your finals week flexibility."]},
    {"id": "u03", "kind": "unanswerable", "category": "project_requirements",
     "question": "Is there a word limit for the Task 5 code analysis report?",
     "referenceAnswer": "The syllabus sets no length for the Task 5 report. It says the analysis is published as a single report in the main group repository that everyone contributes to, with findings turned into user stories; no word count or page limit is given.",
     "keyFacts": ["says no length limit is specified", "does not invent a word count"],
     "sourceSection": "Bot Project Task #5 (not specified)",
     "supportingPassages": ["publishing a report to the main group repository (everyone from the group should contribute to a single report)"]},
    {"id": "u04", "kind": "unanswerable", "category": "grading",
     "question": "What minimum score on the Reflection essay do I need in order to pass the course?",
     "referenceAnswer": "No such minimum is stated. The Reflection is 5% of the grade; the course grade maps an overall 97.0% to a 4.0 and 60.0% to a 0.7 with even spacing (a 2.0 needs 74.6%). Nothing sets a minimum on any single assignment.",
     "keyFacts": ["says no minimum Reflection score is specified", "does not invent a threshold"],
     "sourceSection": "Grading (not specified)",
     "supportingPassages": ["Reflection: 5%", "For this quarter of this class, I will map an overall 97.0% to a 4.0 and a 60.0% to a 0.7, with even spacing in between."]},
    {"id": "u05", "kind": "unanswerable", "category": "course_policies",
     "question": "What is the penalty if I accidentally upload the wrong file to Canvas for an assignment?",
     "referenceAnswer": "The syllabus states no penalty. It says only that it is the student's responsibility to verify that the correct file is submitted, and to contact UW-IT for Canvas difficulties.",
     "keyFacts": ["says no penalty is specified", "it is the student's responsibility to verify the correct file is submitted"],
     "sourceSection": "Websites and Technology Expectations (not specified)",
     "supportingPassages": ["It is the student's responsibility to verify that the correct file is submitted."]},
    {"id": "u06", "kind": "unanswerable", "category": "course_resources",
     "question": "What is the invite link for the class Discord server?",
     "referenceAnswer": "The syllabus does not contain the invite link. It says the invitation link is in the Class Setup Checklist on Canvas, where the joining instructions are.",
     "keyFacts": ["says the link is not in the syllabus", "the invitation link is in the Class Setup Checklist on Canvas"],
     "sourceSection": "Course Websites (not specified)",
     "supportingPassages": ["We will use Discord for chat, including to ask questions and share information around the course material. (Invitation link is in the Class Setup Checklist on Canvas.)"]},
    {"id": "u07", "kind": "unanswerable", "category": "office_hours",
     "question": "What time of day are office hours held each week?",
     "referenceAnswer": "The syllabus does not list office-hour times. Available hours are shown in the instructor's calendar scheduling site; appointments are booked there at least 24 hours in advance, and if the planned availability does not work you contact her on Discord or by email to arrange another time.",
     "keyFacts": ["says no times are listed in the syllabus", "hours are visible in the calendar scheduling site"],
     "sourceSection": "Office Hours (not specified)",
     "supportingPassages": ["My available hours are visible in this calendar scheduling site. If my planned availability does not work for you, please contact me in the Discord server or over email to arrange a meeting at another time."]},
    # ---- false premise -------------------------------------------------------------
    {"id": "p01", "kind": "false_premise", "category": "course_resources",
     "premise": "assumes a required textbook exists",
     "question": "Which edition of the required textbook should I buy, and is an older edition acceptable?",
     "referenceAnswer": "There is no required textbook. Material comes from a wide range of sources that should all be free through Canvas or the UW Library Proxy; report access problems via a public Discord message.",
     "keyFacts": ["there is no required textbook", "materials are free via Canvas or the UW Library proxy"],
     "sourceSection": "Textbook",
     "supportingPassages": ["We do not have a required textbook. You will instead engage with material from a wide range of sources. All of these sources should be free of charge providing you use the version on Canvas or the UW Library Proxy."]},
    {"id": "p02", "kind": "false_premise", "category": "ai_policy",
     "premise": "assumes AI use is allowed on the Reflection if cited",
     "question": "Can I have ChatGPT draft my Reflection on the bot project if I cite it as a source?",
     "referenceAnswer": "No. The Reflection must be written without AI tools; the syllabus says use of AI tools for this assignment is forbidden and warns that AI fabricates citations. The general policy allows AI on some assignments when the instructions permit it and it is cited, but not this one; grammar and spellchecking is not considered a prohibited use.",
     "keyFacts": ["no; AI tools are forbidden for the Reflection", "citing it does not make it allowed"],
     "sourceSection": "Reflection on Software Engineering and the Bot Project",
     "supportingPassages": ["Note that AI tools are notorious for fabricating these and use of AI tools for this assignment is forbidden.", "Do not use AI tools."]},
    {"id": "p03", "kind": "false_premise", "category": "project_requirements",
     "premise": "assumes Bot Feedback is one submission per group",
     "question": "Since Bot Feedback is submitted once per group, which team member should post our group's feedback paragraph?",
     "referenceAnswer": "Nobody submits for the group: Bot Feedback is individual work. Each student writes a paragraph of highly specific feedback for the group they are assigned and submits it as a text submission on Canvas by Monday, December 1, 11:59 p.m.; each group receives feedback from multiple peers and from the instructor, and AI tools may not be used for it.",
     "keyFacts": ["it is individual, not per group", "each student submits their own paragraph on Canvas"],
     "sourceSection": "Bot Feedback",
     "supportingPassages": ["You will work individually; each group will get feedback from multiple peers as well as from me.", "a text submission on Canvas"]},
    {"id": "p04", "kind": "false_premise", "category": "course_policies",
     "premise": "assumes written Reading Note answers are submitted and excuse cold calling",
     "question": "Where do I upload my written answers to the Reading Note questions so I can be excused from cold calling?",
     "referenceAnswer": "Nowhere: the instructor does not collect written answers to the Reading Note questions, and cold calling is not excused. Writing answers in advance is recommended and brainstorming with other students is welcome, but you will still be cold called on prepared questions from a random, self-balancing list.",
     "keyFacts": ["reading-note answers are not collected", "cold calling still happens"],
     "sourceSection": "Cold Calling",
     "supportingPassages": ["Although it is a very good idea to write out answers to these questions in advance, I will not be collecting these answers.", "I will be doing this in each class."]},
    {"id": "p05", "kind": "false_premise", "category": "attendance",
     "premise": "assumes standup has no effect on the grade",
     "question": "Since standup attendance doesn't affect our grade, can my group skip standup on weeks when we're busy?",
     "referenceAnswer": "The premise is wrong: standup participation is equivalent to answering one question in class and is assessed under in-class activities, which are 50% of the grade; serving as scrum master counts as one more question. There is no way to make up a missed standup, and missing one makes you more likely to be called on in later classes.",
     "keyFacts": ["standup participation counts, equivalent to answering one question in class", "no makeup for a missed standup"],
     "sourceSection": "Impact of Missing Class",
     "supportingPassages": ["Standup participation is equivalent to answering one question in class.", "There is no way to directly make up missing a standup meeting with your group"]},
    {"id": "p06", "kind": "false_premise", "category": "project_requirements",
     "premise": "assumes the instructor installs and runs the bot herself",
     "question": "How do I package our bot so Dr. Champion can install and run it on her own machine for grading?",
     "referenceAnswer": "You don't. She does not want to install your bot; she only wants to be a member of your group's deployment server, invited with the rest of the group. Because of the architecture, someone on the team must be running the bot locally when it is graded, so coordinate that.",
     "keyFacts": ["she does not want to install your bot", "she only needs to be a member of your server", "someone on the team runs the bot locally for grading"],
     "sourceSection": "Bot Project Task #1 / Task #3",
     "supportingPassages": ["She does not want to install your bot, she only wants to be a member of your server!", "Our architecture means someone will need to be running the bot locally for it to be graded, so expect to coordinate this."]},
    {"id": "p07", "kind": "false_premise", "category": "course_policies",
     "premise": "assumes the syllabus is a binding contract",
     "question": "Since the syllabus is a binding contract, can I insist on the reading list exactly as it was published on the first day?",
     "referenceAnswer": "No. The syllabus is explicitly a dynamic document and not a contract: readings and assignments will shift, with the six-day rule limiting late changes, and every change is trackable through the wiki page history and summarized in the weekly Canvas announcement.",
     "keyFacts": ["the syllabus is a dynamic document, not a contract", "readings can change, subject to the six-day rule"],
     "sourceSection": "Note about this Syllabus",
     "supportingPassages": ["You should expect this syllabus to be a dynamic document (not a 'contract')."]},
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    syllabus = normalize(SYLLABUS_FILE.read_text(encoding="utf-8"))
    problems: list[str] = []

    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    repeat_questions = []
    for q in original["questions"]:
        item = dict(q)
        item["kind"] = REPEAT_KINDS.get(q["id"], "factual")
        if q["id"] in REPEAT_PREMISES:
            item["premise"] = REPEAT_PREMISES[q["id"]]
        repeat_questions.append(item)

    seen_ids = set()
    for q in HELD_OUT:
        if q["id"] in seen_ids:
            problems.append(f"duplicate id {q['id']}")
        seen_ids.add(q["id"])
        for passage in q["supportingPassages"]:
            if normalize(passage) not in syllabus:
                problems.append(f"{q['id']}: passage not found verbatim in the syllabus: {passage[:80]!r}")
        for other in repeat_questions:
            s = similarity(q["question"], other["question"])
            if verdict_for(s) != "clearly held out":
                problems.append(f"{q['id']} vs repeat22 {other['id']}: {verdict_for(s)} {s}")

    old_bank_rows = []
    bank = json.loads(OLD_BANK.read_text(encoding="utf-8"))
    for course in bank.get("courses", []):
        if course.get("courseId") == COURSE_ID:
            old_bank_rows = [(x["id"], x["question"]) for x in course["questions"]]

    rows, files = training_rows(EXPORT_DIR)
    if not rows:
        problems.append("no July export under data/exports; cannot compute overlapJulySplit")

    if problems:
        for p in problems:
            print("PROBLEM", p)
        return 1

    def annotate(items: list[dict]) -> None:
        for q in items:
            q["overlapJulySplit"] = overlap_for_question(q["question"], q.get("keyFacts") or [], rows)
            nearest_bank = None
            for bid, text in old_bank_rows:
                s = similarity(q["question"], text)
                if nearest_bank is None or max(s.values()) > max(nearest_bank["measures"].values()):
                    nearest_bank = {"id": bid, "question": text, "measures": s, "verdict": verdict_for(s)}
            q["nearestInOldHeldOutBank"] = nearest_bank

    annotate(repeat_questions)
    annotate(HELD_OUT)
    for q in HELD_OUT:
        verdict = q["overlapJulySplit"]["questionVerdict"]
        if verdict.startswith("REJECT"):
            problems.append(f"{q['id']}: July-split verdict {verdict}; replace the question")
        elif verdict.startswith("REVIEW") and q["id"] not in JULY_RESOLUTIONS:
            problems.append(f"{q['id']}: July-split verdict {verdict} with no hand resolution")
        if q["id"] in JULY_RESOLUTIONS:
            q["overlapJulySplit"]["resolvedByHand"] = JULY_RESOLUTIONS[q["id"]]
    if problems:
        for p in problems:
            print("PROBLEM", p)
        return 1

    july = {"exportDir": "data/exports/css-360-winter-2026-a7rp", "files": files,
            "meaning": "July 2026 laptop export, 54 approved / 48 train / 6 validation: the candidate for the v2 and v3 training data, whose bytes are not retained (evolution record, sections 3 and 4). Not the v4 data; the runner checks the VM's export."}

    repeat = {
        "schemaVersion": 1, "set": "repeat22", "courseId": COURSE_ID, "createdAt": "2026-09-19",
        "purpose": "The 22 questions of the 2026-09-11 benchmark, text unchanged, asked again under the controlled settings of the research route. Results are reported separately from the 2026-09-11 scores and from the held-out set.",
        "sourceFile": str(ORIGINAL.relative_to(ROOT)), "sourceFileSha256": sha256_file(ORIGINAL),
        "kinds": {"factual": "answerable from the syllabus", "unanswerable": "the syllabus does not specify the asked fact", "false_premise": "the question assumes something the syllabus contradicts"},
        "overlapJulySplit": july, "questions": repeat_questions,
    }
    heldout = {
        "schemaVersion": 1, "set": "heldout", "courseId": COURSE_ID, "createdAt": "2026-09-19",
        "purpose": "New questions written for the controlled benchmark from the syllabus text, never asked of any model before this run: syllabus-supported factual, unanswerable, and false-premise questions. Each carries verbatim supporting passages verified against the syllabus. referenceAnswer, keyFacts and supportingPassages are for the scorer only and are never sent to a model.",
        "syllabusFile": str(SYLLABUS_FILE.relative_to(ROOT)), "syllabusSha256": sha256_file(SYLLABUS_FILE),
        "kinds": repeat["kinds"], "overlapJulySplit": july, "questions": HELD_OUT,
    }
    (HERE / "questions_repeat22.json").write_text(json.dumps(repeat, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "questions_heldout.json").write_text(json.dumps(heldout, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def report(label: str, items: list[dict]) -> None:
        print(f"== {label}: {len(items)} questions")
        for q in items:
            o = q["overlapJulySplit"]
            flag = "" if o["questionVerdict"] == "clearly held out" else "  <-- " + o["questionVerdict"]
            print(f"  {q['id']} {q['kind']:14s} july: facts {o['keyFactCoverage']}, nearest j={o['nearest']['measures']['jaccard']:.2f} c={o['nearest']['measures']['containment']:.2f} r={o['nearest']['measures']['ratio']:.2f}{flag}")
    report("repeat22", repeat_questions)
    report("heldout", HELD_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
