import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import load_model, predict

LABEL_MAP_PATH = Path("data/processed/label_map.json")
OVERRIDES_CSV = Path("data/overrides.csv")


@st.cache_resource
def get_labels() -> list[str]:
    with open(LABEL_MAP_PATH) as f:
        return sorted(json.load(f).keys())


@st.cache_resource
def init_model() -> None:
    load_model()


def log_override(entry: dict, override_tag: str) -> None:
    row = {
        "timestamp": entry["timestamp"],
        "original_text": entry["text"],
        "predicted_tag": entry["predicted_tag"],
        "confidence": entry["confidence"],
        "override_tag": override_tag,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_header = not OVERRIDES_CSV.exists()
    pd.DataFrame([row]).to_csv(OVERRIDES_CSV, mode="a", header=write_header, index=False)


def main() -> None:
    st.title("CFPB Complaint Review Tool")

    init_model()
    labels = get_labels()

    if "history" not in st.session_state:
        st.session_state.history = []

    tab_pred, tab_low = st.tabs(["Predictions", "Low Confidence Queue"])

    with tab_pred:
        narrative = st.text_area(
            "Complaint narrative",
            height=150,
            placeholder="Paste complaint text here…",
        )

        if st.button("Predict"):
            if narrative.strip():
                result = predict(narrative)
                st.session_state.history.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "text": narrative[:120],
                        "predicted_tag": result["tag"],
                        "confidence": result["confidence"],
                        "low_confidence": result["low_confidence"],
                        "override": None,
                    }
                )
            else:
                st.error("Please enter a complaint narrative before predicting.")

        if st.session_state.history:
            latest = st.session_state.history[-1]

            st.markdown("### Latest prediction")
            st.success(
                f"**{latest['predicted_tag']}** — {latest['confidence']:.1%} confidence"
            )
            if latest["low_confidence"]:
                st.warning("Low confidence — consider reviewing manually.")

            st.markdown("### Prediction history")
            st.dataframe(
                pd.DataFrame(st.session_state.history[-20:]),
                use_container_width=True,
            )

            st.markdown("### Override most recent prediction")
            override_tag = st.selectbox("Corrected tag", labels)
            if st.button("Submit Override"):
                st.session_state.history[-1]["override"] = override_tag
                log_override(latest, override_tag)
                st.success(f"Override saved: {override_tag}")

    with tab_low:
        st.info("Low confidence queue — coming Thursday")


if __name__ == "__main__":
    main()
