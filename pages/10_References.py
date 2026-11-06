import streamlit as st

from src.content import REFERENCES

st.title("References")
st.caption("Please verify page numbers and license terms against the original sources before citing.")
for section, items in REFERENCES.items():
    st.subheader(section)
    for item in items:
        st.markdown(f"- {item}")
