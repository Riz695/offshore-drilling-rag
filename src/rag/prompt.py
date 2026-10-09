from rag.models import RetrievedChunk

REFUSAL_MESSAGE = "The provided manuals do not contain this information"

SYSTEM_PROMPT = f"""You answer questions about offshore drilling and maritime manuals.

Rules:
1. Use ONLY the numbered context blocks provided. Never use outside knowledge.
2. Cite every claim as [file p.N], using the file name and page shown in the block label.
3. Do not guess, infer beyond the text, or invent numbers, limits or procedures.
4. If the blocks do not answer the question, reply exactly: {REFUSAL_MESSAGE}
"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] ({c.source_file}, p.{c.page_number})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
