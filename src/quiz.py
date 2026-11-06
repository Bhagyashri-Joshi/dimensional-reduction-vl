"""Quiz data model, scoring and Streamlit rendering helper (PRD 5.4 / 5.10)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    prompt: str
    options: tuple[str, ...]
    answer: int  # index into options
    explanation: str


def score_quiz(questions: list[Question], answers: list[int | None]) -> dict:
    """Score a quiz. ``answers[i]`` is the chosen option index or None (unanswered = wrong)."""
    if len(answers) != len(questions):
        raise ValueError("answers and questions must have the same length")
    correct = [a is not None and a == q.answer for q, a in zip(questions, answers)]
    return {
        "score": int(sum(correct)),
        "total": len(questions),
        "correct": correct,
        "unanswered": int(sum(a is None for a in answers)),
    }


def render_quiz(st, key: str, questions: list[Question]) -> None:
    """Render a quiz form; results persist in session state until the student retries."""
    result_key = f"{key}_result"
    result = st.session_state.get(result_key)

    if result is None:
        with st.form(f"{key}_form"):
            for i, q in enumerate(questions):
                st.radio(f"**Q{i + 1}.** {q.prompt}", q.options, index=None, key=f"{key}_q{i}")
            submitted = st.form_submit_button("Submit answers", type="primary")
        if submitted:
            answers = []
            for i, q in enumerate(questions):
                choice = st.session_state.get(f"{key}_q{i}")
                answers.append(q.options.index(choice) if choice in q.options else None)
            st.session_state[result_key] = {"answers": answers, **score_quiz(questions, answers)}
            st.rerun()
        return

    st.success(f"Score: **{result['score']} / {result['total']}**")
    if result["unanswered"]:
        st.warning(f"{result['unanswered']} question(s) were left unanswered and count as incorrect.")
    for i, q in enumerate(questions):
        chosen = result["answers"][i]
        ok = result["correct"][i]
        with st.expander(f"Q{i + 1}. {q.prompt}", expanded=not ok):
            st.markdown(f"**Your answer:** {q.options[chosen] if chosen is not None else '_not answered_'}")
            st.markdown(f"**Correct answer:** {q.options[q.answer]}")
            st.markdown(q.explanation)

    def _retry():
        st.session_state.pop(result_key, None)
        for i in range(len(questions)):
            st.session_state.pop(f"{key}_q{i}", None)

    st.button("Retake quiz", on_click=_retry, key=f"{key}_retry")
