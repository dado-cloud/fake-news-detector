import os
import sys
from pathlib import Path

import streamlit as st

# --- import from src/ ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_text  # noqa: E402
# ------------------ UI CONFIG ------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Fake News Detector")
st.caption("Paste a news article or upload a text file to get an instant credibility prediction.")

# ------------------ SIDEBAR (USER-FRIENDLY) ------------------
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This tool analyzes the text of a news article and predicts whether it is **more likely REAL or FAKE** "
        "based on language patterns."
    )

    st.subheader("How to use")
    st.markdown(
        """
1. Paste an article (English) **or** upload a `.txt` file  
2. Click **Predict**  
3. Review the result and the notes below  
"""
    )

    st.subheader("Important note")
    st.info(
        "This is an AI-based prediction, not a fact-check. "
        "For important news, always verify using trusted and official sources."
    )

    st.divider()
    show_details = st.checkbox("Show technical details", value=False)


# ------------------ OPTIONAL: confidence-like score ------------------
def score_to_confidence(decision_score: float | None) -> float | None:
    """
    Converts a model decision score into a 0..1 number (confidence-like).
    This is NOT a true probability; it's just a helpful indicator for users.
    """
    if decision_score is None:
        return None
    import math
    return 1.0 / (1.0 + math.exp(-decision_score))


# ------------------ INPUT TABS ------------------
tab1, tab2 = st.tabs(["Paste text", "Upload .txt"])
text_input = ""

with tab1:
    text_input = st.text_area(
        "Paste your news article (English):",
        height=220,
        placeholder="Paste the full article text  streamlit run app/streamlit_app.pyhere...",
    )

   

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        try:
            text_input = uploaded.read().decode("utf-8", errors="ignore")
            st.success("File loaded. Go to the bottom and click Predict.")
        except Exception:
            st.error("Could not read this file. Please upload a plain .txt file.")


st.divider()

# ------------------ ACTION BUTTONS ------------------
col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()


# ------------------ PREDICTION ------------------
if predict_clicked:
    if not text_input or not text_input.strip():
        st.warning("Please paste an article or upload a .txt file first.")
    else:
        try:
            result = predict_text(text_input)

            label = result.get("label")  # expected: "REAL" or "FAKE"
            decision_score = result.get("decision_score", None)
            conf = score_to_confidence(decision_score)

            # Main result (simple + user-friendly)
            if label == "FAKE":
                st.error("Result: **Likely FAKE**")
                st.caption(
                    "This text contains patterns often seen in misleading or unreliable news. "
                    "Consider checking trusted sources before sharing."
                )
            else:
                st.success("Result: **Likely REAL**")
                st.caption(
                    "This text contains patterns often seen in reliable news writing. "
                    "Still, verify with trusted sources for important topics."
                )

            # Confidence-like indicator (optional but helpful)
            if conf is not None:
                conf_clamped = min(max(conf, 0.0), 1.0)
                conf_percent = round(conf_clamped * 100, 1)

                st.write(f"**Confidence in this prediction: {conf_percent}%**")
                st.progress(conf_clamped)

                # User-friendly explanation
                if conf_percent < 50:
                    confidence_text = (
                    "Low confidence: the text is very short, unclear, or does not look like a real news article."
                        )
                elif conf_percent < 75:
                    confidence_text = (
                    "Medium confidence: the text has some news-like patterns, but the signal is not strong."
                    )
                else:
                    confidence_text = (
                    "High confidence: the text strongly matches patterns seen in real or fake news articles."
                    )

                st.caption(
                    confidence_text
                    + " This score shows how confident the AI is — it does NOT mean the article is "
                    + f"{conf_percent}% fake or real."
                )

            

            # Extra guidance for users
            with st.expander("Tips to verify news (recommended)"):
                st.markdown(
                    """
- Check the same story on trusted outlets (official agencies, major news sites)  
- Look for author name, date, and credible references  
- Watch for emotional headlines, ALL CAPS, or “too good to be true” claims  
- If it’s about health or safety, rely on official sources only  
"""
                )

            # Technical details only if user wants them
            if show_details:
                with st.expander("Technical details"):
                    st.json(
                        {
                            "label": label,
                            "label_id": result.get("label_id"),
                            "decision_score": decision_score,
                            "confidence_like": conf,
                        }
                    )

        except FileNotFoundError:
            # Public-friendly message (no internal commands)
            st.error("The prediction model is not available right now.")
            st.info(
                "This app is still being set up. Please try again later, or contact the app owner."
            )

        except Exception:
            st.error("Something went wrong while analyzing the text.")
            st.info("Please try again with a different article or a simpler .txt file.")


st.divider()






