import streamlit as st
from test import hybrid_search
from answer import generate_answer

st.set_page_config(page_title="Stardew Bot", page_icon="🐔", layout="centered")
st.title("🪴Stardew Valley🪴")
st.caption("Basado en la wiki oficial (Nota: la wiki estaba en inglés, por favor hacer las preguntas en inglés)")
#html y csss
st.html("""
<style>
    .stSpinner > div > div {
        border-top-color: #6B8E23 !important;
    }
    h1 {
        text-align: center;
        color: #4A3B2A;
        text-shadow: 1px 1px 0 #F5DEB3;
        letter-spacing: 1px;
    }
    [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #8B4513;
        font-style: italic;
    }
    .stTextInput > div > div > input {
        border: 1px solid #C4A77D;
        border-radius: 8px;
        background-color: #FFFDF5;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6B8E23 !important;
        box-shadow: 0 0 0 2px rgba(107, 142, 35, 0.25) !important;
    }
    .st-key-respuesta_junimo {
        background-color: #FAFAD2;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #6B8E23;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08);
        margin-top: 10px;
    }
    .st-key-respuesta_junimo h3 {
        color: #4A3B2A;
        margin-top: 0;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #F5DEB3;
        border-radius: 8px 8px 0 0;
        color: #4A3B2A;
        font-weight: bold;
        padding: 8px 12px;
    }
    div[data-testid="stExpander"] details[open] summary {
        border-bottom: 2px solid #6B8E23;
    }
    div[data-testid="stExpander"] details > div {
        background-color: #FFF8DC;
        border-left: 4px solid #6B8E23;
        padding: 10px 14px;
    }
</style>
""")

ALPHA = 0.3
TOP_K = 3

query = st.text_input(
    "Tu pregunta:",
    placeholder="Ex. What gifts does Haley love? Where does Alex live?...",
)

if query:
    with st.spinner("...Los junimos están buscando la respuesta..."):
        results = hybrid_search(
            query, page_filter=None, n_results=TOP_K, alpha=ALPHA
        )
        chunk_texts = [doc for _, _, doc, _ in results]
        answer = generate_answer(query, chunk_texts)

    with st.container(key="respuesta_junimo"):
        st.markdown("### Respuesta")
        st.write(answer)

    with st.expander("-- Fuentes"):
        for i, (score, section, doc, page) in enumerate(results):
            st.markdown(f"**{i+1}. [{page} / {section}]** (score={score:.3f})")
            st.caption(doc[:400] + ("..." if len(doc) > 400 else ""))