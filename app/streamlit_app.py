from typing import Any

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
TIMEOUT_S = 180  # local 8B model on CPU can be slow


def fetch_answer(question: str) -> dict[str, Any]:
    """POST to /query. Return the response JSON, or {"error": message}."""
    try:
        resp = requests.post(
            f"{API_URL}/query", json={"question": question}, timeout=TIMEOUT_S
        )
    except requests.RequestException as exc:
        return {"error": f"Cannot reach backend at {API_URL}: {exc}"}
    if not resp.ok:
        try:
            detail = resp.json().get("detail", "")
        except ValueError:
            detail = ""
        return {"error": f"Backend error {resp.status_code}: {detail}"}
    return resp.json()


def render_result(result: dict[str, Any]) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(result["answer"])
    for src in result["sources"]:
        with st.expander(f"{src['source_file']}, p.{src['page_number']}"):
            st.markdown(src["excerpt"])


def main() -> None:
    st.title("Offshore Drilling Manuals")
    question = st.text_input("Ask a question about the manuals")
    if st.button("Ask") and question.strip():
        with st.spinner("Searching manuals..."):
            st.session_state["result"] = fetch_answer(question.strip())
    if "result" in st.session_state:
        render_result(st.session_state["result"])


main()
