import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import load_model, predict

LABEL_MAP_PATH = Path("data/processed/label_map.json")
OVERRIDES_CSV = Path("data/overrides.csv")
PREDICTIONS_CSV = Path("data/predictions.csv")


@st.cache_resource
def get_labels() -> list[str]:
    with open(LABEL_MAP_PATH) as f:
        return sorted(json.load(f).keys())


@st.cache_resource
def init_model() -> None:
    load_model()


def log_prediction(narrative: str, predicted_label: str, confidence: float) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "narrative": narrative,
        "predicted_label": predicted_label,
        "confidence": confidence,
    }
    write_header = not PREDICTIONS_CSV.exists()
    pd.DataFrame([row]).to_csv(PREDICTIONS_CSV, mode="a", header=write_header, index=False)


def load_stats() -> tuple[int, float, int]:
    if not PREDICTIONS_CSV.exists():
        return 0, 0.0, 0
    preds = pd.read_csv(PREDICTIONS_CSV)
    total = len(preds)
    low_conf = int((preds["confidence"] < 0.65).sum())
    override_rate = 0.0
    if total > 0 and OVERRIDES_CSV.exists():
        overrides = pd.read_csv(OVERRIDES_CSV)
        override_rate = len(overrides) / total
    return total, override_rate, low_conf


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
                log_prediction(narrative, result["tag"], result["confidence"])
            else:
                st.error("Please enter a complaint narrative before predicting.")

        total_preds, override_rate, low_conf_count = load_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", total_preds)
        c2.metric("Override rate", f"{override_rate:.1%}")
        c3.metric("Low-confidence items", low_conf_count)

        if st.session_state.history:
            latest = st.session_state.history[-1]

            st.markdown("### Latest prediction")
            st.success(
                f"**{latest['predicted_tag']}** — {latest['confidence']:.1%} confidence"
            )
            if latest["low_confidence"]:
                st.warning("Low confidence — consider reviewing manually.")

            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, step=0.05)
            st.markdown("### Prediction history")
            filtered = [h for h in st.session_state.history[-20:] if h["confidence"] >= min_conf]
            st.dataframe(
                pd.DataFrame(filtered),
                use_container_width=True,
            )

            st.markdown("### Override most recent prediction")
            override_tag = st.selectbox("Corrected tag", labels)
            if st.button("Submit Override"):
                st.session_state.history[-1]["override"] = override_tag
                log_override(latest, override_tag)
                st.success(f"Override saved: {override_tag}")

    with tab_low:
        if not PREDICTIONS_CSV.exists():
            st.info("No low-confidence predictions in queue.")
        else:
            preds = pd.read_csv(PREDICTIONS_CSV)
            low = preds[preds["confidence"] < 0.65].sort_values("confidence").copy()
            if low.empty:
                st.info("No low-confidence predictions in queue.")
            else:
                overridden_texts: set[str] = set()
                if OVERRIDES_CSV.exists():
                    overrides = pd.read_csv(OVERRIDES_CSV)
                    overridden_texts = set(overrides["original_text"].dropna())
                low["narrative_display"] = low["narrative"].str[:120]
                low["overridden"] = low["narrative"].str[:120].apply(
                    lambda t: "✓" if t in overridden_texts else ""
                )
                st.dataframe(
                    low[["timestamp", "narrative_display", "predicted_label", "confidence", "overridden"]].rename(
                        columns={"narrative_display": "narrative"}
                    ),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
