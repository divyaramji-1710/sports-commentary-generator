"""Streamlit interface for the LoRA commentary model."""
import sys
from pathlib import Path

import streamlit as st

# Make project imports work with `streamlit run app/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import CommentaryGenerator
from app.merging import merge_model


st.set_page_config(
    page_title="Sports Commentary Generator",
    page_icon="⚽",
)

st.title("⚽ Sports Commentary Generator")
st.write(
    "Turn a simulated football event into commentary "
    "using a locally trained LoRA adapter."
)
st.caption(
    "Trained on synthetic goals, saves, and full-time events "
    "with fictional teams. Other formats may be unreliable."
)


@st.cache_resource
def load_generator():
    checkpoint = PROJECT_ROOT / "artifacts/commentary/adapter.pt"
    generator = CommentaryGenerator(checkpoint)
    merge_model(generator.model)
    return generator


examples = {
    "Goal": (
        "Football. Minute 72. Goal for City. "
        "Score after goal: City 2-1 Rovers."
    ),
    "Save": (
        "Football. Minute 35. Save by Eagles. "
        "Score unchanged: Eagles 0-0 United."
    ),
    "Full-time": (
        "Football. Full-time. "
        "Final score: Wolves 1-2 Falcons."
    ),
}

selection = st.selectbox(
    "Choose an example",
    list(examples),
)

with st.form("commentary_form"):
    event = st.text_area(
        "Event description",
        value=examples[selection],
        height=120,
        key=f"event_{selection}",
    )
    max_tokens = st.slider(
        "Maximum new tokens",
        min_value=16,
        max_value=80,
        value=48,
        step=8,
    )
    submitted = st.form_submit_button("Generate commentary")

if submitted:
    if not event.strip():
        st.warning("Enter an event description.")
    else:
        try:
            with st.spinner("Generating commentary..."):
                generator = load_generator()
                commentary = generator.generate(
                    event,
                    max_new_tokens=max_tokens,
                )

            if commentary:
                st.subheader("Commentary")
                st.write(commentary)
            else:
                st.warning(
                    "The model returned no commentary. "
                    "Try a supported event format."
                )

        except FileNotFoundError:
            st.error(
                "The adapter checkpoint is missing. "
                "Train the model or restore "
                "artifacts/commentary/adapter.pt."
            )
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))