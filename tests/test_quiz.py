import pytest

from src.content import POST_TEST, PRE_TEST
from src.quiz import Question, score_quiz

QS = [Question("a", ("x", "y"), 1, "e"), Question("b", ("x", "y", "z"), 0, "e"), Question("c", ("x", "y"), 1, "e")]


def test_scoring():
    r = score_quiz(QS, [1, 0, 0])
    assert r["score"] == 2 and r["total"] == 3 and r["correct"] == [True, True, False]


def test_unanswered_counts_wrong():
    r = score_quiz(QS, [None, 0, None])
    assert r["score"] == 1 and r["unanswered"] == 2


def test_length_mismatch():
    with pytest.raises(ValueError):
        score_quiz(QS, [1])


@pytest.mark.parametrize("quiz", [PRE_TEST, POST_TEST])
def test_content_quizzes_are_well_formed(quiz):
    assert len(quiz) == 10
    for q in quiz:
        assert 0 <= q.answer < len(q.options) and len(set(q.options)) == len(q.options)
        assert q.explanation
    assert score_quiz(quiz, [q.answer for q in quiz])["score"] == 10
