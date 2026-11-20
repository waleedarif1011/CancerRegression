# Hugging Face Space deploy

The Gradio County Risk Explorer is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create cancerregression --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/cancerregression
git push space main
```

3. Confirm the Space builds and the three county presets behave as expected:
   - Lower-risk county → below / near US median death rate
   - Median-risk county → near US median
   - Higher-risk county → above US median
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/cancerregression`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`cancer_model.joblib`, `metrics.json`)
- `assets/` (plots used in the UI / README)
- `dataset/` (only required if re-training on the Space; inference uses `artifacts/`)
