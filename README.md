# MedVQA — Cross-Attention & Gated Cross-Attention Fusion for Pathology VQA

Fine-tuning a multimodal GPT-2 (frozen ViT encoder + LoRA-adapted GPT-2 decoder)
to answer visual questions on pathology images, comparing standard cross-attention
fusion against a gated variant.

## Overview

**Visual Question Answering (VQA)** in pathology asks a model to answer a natural-
language question about a microscopy image — e.g. *"What is present?"* → *"Cardiovascular"*.
This project builds a **MedVQA** model that fuses a frozen vision encoder with a
language model, then compares two ways of merging the visual and textual signals:
plain cross-attention, and a gated version that learns how much visual information
to let through at each token.

## Dataset

**[PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa)** — a
large-scale visual question answering benchmark for digital pathology.

| | |
|---|---|
| Total QA pairs | ~32,000 |
| Pathology images | ~5,004 |
| Validation split used | 6,259 samples |

### Sample data

Three representative validation examples — minimum, average, and maximum
question+answer token length (measured with the GPT-2 tokenizer):

`figures/qa_samples_visualisation.png`

## Model Architecture

- **Visual encoder**: frozen ViT-Base, extracts patch-level image features
- **Fusion module**: merges image and text representations — either:
  - **Cross-Attention Fusion** — text tokens attend to image tokens as queries,
    with image tokens as keys/values (multi-head attention → residual → LayerNorm
    → feed-forward → residual → LayerNorm)
  - **Gated Cross-Attention Fusion** — extends the above with a learned sigmoid
    gate that controls how much attended visual context is blended into each
    text token before it's passed on
- **Language decoder**: GPT-2, fine-tuned with **LoRA** adapters (visual encoder
  stays frozen throughout)

## Training Setup

| | Baseline | Tuned |
|---|---|---|
| Learning rate | 2e-4 | 5e-5 |
| Epochs | 5 | 8 |
| Batch size | 24 | 24 |
| Max sequence length | 68 | 68 |
| Optimizer | Adam | Adam |
| Decoding (inference) | Greedy search | Greedy search |

Both Cross-Attention and Gated Cross-Attention fusion were trained under
identical settings for a fair comparison, then re-trained with the tuned
hyperparameters above.

## Results

### Cross-Attention vs. Gated Cross-Attention (baseline settings)

See `results/table1_metrics.csv` for the full numbers (BLEU-1, BLEU-2, ROUGE-L,
METEOR).

**Prediction examples:**
- Cross-Attention: `figures/fig1_ca_predictions.png`
- Gated Cross-Attention: `figures/fig1_gca_predictions.png`

The Gated model traded a small amount of BLEU (n-gram overlap) for improved
ROUGE-L and METEOR — i.e. its answers diverged slightly more in exact wording
but captured the meaning of the ground-truth answer more consistently.

### Hyperparameter tuning

Reducing the learning rate (2e-4 → 5e-5) and increasing training length
(5 → 8 epochs) improved optimisation stability overall, and benefited the Gated
model most consistently.

**Training curves:**
- Cross-Attention (baseline vs. tuned): `figures/training_curves_ca.png`
- Gated Cross-Attention (baseline vs. tuned): `figures/training_curves.png`
- All four experiments side-by-side: `figures/training_curves_comparison.png`

**Tuned prediction examples:**
- Cross-Attention (tuned): `figures/fig1_ca_tuned_predictions.png`
- Gated Cross-Attention (tuned): `figures/fig1_tuned_predictions.png`

### Final comparison — all four models

See `results/table1_full.csv` for BLEU-1, BLEU-2, ROUGE-L, METEOR, learning
rate, and epoch count across:
1. Cross-Attention Fusion (baseline)
2. Gated Cross-Attention Fusion (baseline)
3. Cross-Attention Fusion (tuned)
4. Gated Cross-Attention Fusion (tuned)

## Discussion

Cross-Attention Fusion was stable at baseline settings but didn't improve much
with tuning. Gated Cross-Attention introduces extra learnable capacity (the
gate), making it more sensitive to training dynamics — it benefited more
consistently from the lower learning rate and longer training, but needs
careful tuning to balance exact wording (BLEU) against answer completeness
(ROUGE-L / METEOR).

## Tech Stack

- **Python** — `torch`, `torchvision`, `transformers`, `peft` (LoRA), `datasets`
- **Evaluation** — `evaluate`, `bert_score`, `rouge_score`, `nltk` (BLEU/METEOR)
- **Visualization** — `matplotlib`, `pandas`

## How to Reproduce

The notebook is designed to run end-to-end in Google Colab with a GPU runtime
(see the "Open in Colab" badge at the top of the notebook). It will:
1. Install dependencies and download NLTK resources
2. Load PathVQA directly from Hugging Face
3. Train both fusion variants (baseline + tuned)
4. Automatically save all figures and result tables listed above
