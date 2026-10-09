from rag.models import RetrievedChunk

REFUSAL_MESSAGE = "The provided manuals do not contain this information"

SYSTEM_PROMPT = f"""You answer questions about offshore drilling and maritime manuals.

Rules:
1. Use ONLY the context blocks provided. Never use outside knowledge.
2. Cite every claim by copying the label above its block, in the form [file p.N].
   Example: "Test pressure is 5000 psi [example.pdf p.7]."
   Never cite with numbers such as [1], and never write the citation any other way.
3. Do not guess, infer beyond the text, or invent numbers, limits or procedures.
4. If the blocks do not answer the question, reply exactly: {REFUSAL_MESSAGE}
"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{c.source_file} p.{c.page_number}]\n{c.text}" for c in chunks
    ]
    context = "\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
