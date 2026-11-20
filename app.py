"""Gradio County Risk Explorer — socioeconomic inputs → predicted death rate."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.predict import load_model, predict
from src.train import COMPARISON_PATH, IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parent

_ARTIFACT = None

SLIDER_SPECS: dict[str, dict] = {
    "incidencerate": {
        "label": "Cancer incidence rate (per 100k)",
        "minimum": 50,
        "maximum": 1200,
        "step": 1,
    },
    "medincome": {
        "label": "Median household income ($)",
        "minimum": 15000,
        "maximum": 120000,
        "step": 100,
    },
    "povertypercent": {
        "label": "Poverty percent",
        "minimum": 1,
        "maximum": 50,
        "step": 0.1,
    },
    "pctprivatecoverage": {
        "label": "Private insurance coverage %",
        "minimum": 10,
        "maximum": 95,
        "step": 0.1,
    },
    "medianage": {
        "label": "Median age",
        "minimum": 20,
        "maximum": 65,
        "step": 0.1,
    },
    "pctbachdeg25_over": {
        "label": "Bachelor's degree % (age 25+)",
        "minimum": 1,
        "maximum": 60,
        "step": 0.1,
    },
    "pctpubliccoveragealone": {
        "label": "Public coverage alone %",
        "minimum": 1,
        "maximum": 55,
        "step": 0.1,
    },
    "pctemployed16_over": {
        "label": "Employment % (age 16+)",
        "minimum": 20,
        "maximum": 85,
        "step": 0.1,
    },
    "pctunemployed16_over": {
        "label": "Unemployment % (age 16+)",
        "minimum": 0.5,
        "maximum": 30,
        "step": 0.1,
    },
    "pcths25_over": {
        "label": "High-school only % (age 25+)",
        "minimum": 5,
        "maximum": 55,
        "step": 0.1,
    },
}


def _get_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "artifacts/cancer_model.joblib missing — run `python -m src.train` first."
            )
        _ARTIFACT = load_model(MODEL_PATH)
    return _ARTIFACT


def _preset_choices() -> list[str]:
    art = _get_artifact()
    presets = art.get("presets") or []
    if presets:
        return [p["label"] for p in presets]
    return ["Default medians"]


def _metrics_blurb() -> str:
    if not METRICS_PATH.is_file():
        return "_Train with `python -m src.train` to populate metrics._"
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = m.get("best", {})
    rows = "\n".join(
        f"| `{r['name']}` | {r['cv_rmse']:.2f} | {r['test_rmse']:.2f} | "
        f"{r['mae']:.2f} | {r['r2']:.3f} |"
        for r in m.get("models", [])
    )
    return (
        f"**Best model:** `{best.get('name', '—')}` "
        f"(R²={best.get('r2', float('nan')):.3f}, "
        f"Test RMSE={best.get('test_rmse', float('nan')):.2f})\n\n"
        "| Model | CV RMSE | Test RMSE | MAE | R² |\n"
        "|-------|---------|-----------|-----|----|\n"
        f"{rows}"
    )


def _card_html(result: dict) -> str:
    drivers = result.get("key_risk_drivers") or []
    driver_rows = "".join(
        f"<li><code>{d['feature']}</code> = {d['value']} "
        f"(importance {d['importance']:.3f})</li>"
        for d in drivers
    )
    return f"""
<div style="border-radius:12px;padding:1.1rem 1.25rem;background:#e8f1f8;
            border:2px solid #2a6f97;font-family:system-ui,sans-serif;">
  <div style="font-size:0.85rem;letter-spacing:0.04em;text-transform:uppercase;
              opacity:0.75;">Predicted cancer death rate</div>
  <div style="font-size:2.4rem;font-weight:700;line-height:1.1;color:#111;">
    {result['predicted_deathrate']:.1f}
    <span style="font-size:1rem;font-weight:500;">per 100,000</span>
  </div>
  <div style="margin-top:0.55rem;color:#222;">
    <strong>{result['risk_band']}</strong> ·
    US median {result['national_median_deathrate']:.1f} ·
    Δ {result['vs_national_median']:+.1f} ·
    percentile {result['percentile_vs_us_counties']:.0f}th
  </div>
  <div style="margin-top:0.9rem;font-size:0.9rem;color:#333;">
    <strong>Key risk drivers</strong>
    <ul style="margin:0.35rem 0 0;padding-left:1.2rem;">{driver_rows}</ul>
  </div>
</div>
"""


def _demo_feature_names() -> list[str]:
    art = _get_artifact()
    return list(art.get("demo_features") or SLIDER_SPECS.keys())


def estimate(*values):
    art = _get_artifact()
    names = _demo_feature_names()
    features = {name: float(val) for name, val in zip(names, values)}
    result = predict(features, artifact=art)
    summary = (
        f"### Predicted death rate **{result['predicted_deathrate']:.1f}** "
        f"per 100k — {result['risk_band']}\n\n"
        f"US national median: **{result['national_median_deathrate']:.1f}** "
        f"(Δ {result['vs_national_median']:+.1f}). "
        f"Approx. **{result['percentile_vs_us_counties']:.0f}th** percentile "
        f"among US counties in the training data."
    )
    plot = str(IMPORTANCE_PATH) if IMPORTANCE_PATH.is_file() else None
    return summary, _card_html(result), plot


def apply_preset(name: str):
    art = _get_artifact()
    names = _demo_feature_names()
    defaults = art.get("feature_defaults") or {}
    presets = art.get("presets") or []
    match = next((p for p in presets if p.get("label") == name), None)
    if match is None:
        return [float(defaults.get(c, 0.0)) for c in names]
    values = match.get("values") or {}
    return [float(values.get(c, defaults.get(c, 0.0))) for c in names]


def build_ui() -> gr.Blocks:
    art = _get_artifact()
    names = _demo_feature_names()
    defaults = art.get("feature_defaults") or {}
    choices = _preset_choices()

    with gr.Blocks(title="County Cancer Risk Explorer") as demo:
        gr.Markdown(
            "# County Cancer Risk Explorer\n\n"
            "Adjust the top socioeconomic / health indicators for a US county "
            "to estimate **cancer death rate per 100,000**, compare against the "
            "national median, and inspect key risk drivers.\n\n"
            + _metrics_blurb()
        )
        with gr.Row():
            with gr.Column(scale=3):
                preset = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="County presets (from held-out split)",
                )
                sliders = []
                for name in names:
                    spec = SLIDER_SPECS.get(
                        name,
                        {
                            "label": name,
                            "minimum": 0,
                            "maximum": 100,
                            "step": 0.1,
                        },
                    )
                    value = float(defaults.get(name, (spec["minimum"] + spec["maximum"]) / 2))
                    value = min(max(value, spec["minimum"]), spec["maximum"])
                    sliders.append(
                        gr.Slider(
                            minimum=spec["minimum"],
                            maximum=spec["maximum"],
                            value=value,
                            step=spec["step"],
                            label=spec["label"],
                        )
                    )
                run = gr.Button("Estimate risk", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                card = gr.HTML()
                importance = gr.Image(label="Feature importance", type="filepath")
                if COMPARISON_PATH.is_file():
                    gr.Image(
                        value=str(COMPARISON_PATH),
                        label="Model comparison",
                        type="filepath",
                    )

        outputs = [summary, card, importance]
        preset.change(fn=apply_preset, inputs=preset, outputs=sliders)
        run.click(fn=estimate, inputs=sliders, outputs=outputs)
        demo.load(fn=estimate, inputs=sliders, outputs=outputs)
        demo.load(fn=apply_preset, inputs=preset, outputs=sliders)
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
