![MangARTI: translation between Formal Indonesian and Manado Malay](assets/banner.png)

# MangARTI

MangARTI is a LoRA adapter for Llama 3 8B Instruct that translates single sentences between **Formal Indonesian** and **Manado Malay** (Bahasa Manado), the everyday language of Manado and much of North Sulawesi.

The name comes from *mangarti*, the Manado Malay word for *mengerti* ("to understand").

This repository holds the code and data used to build it. The trained adapter is on Hugging Face:

- **Model:** [Jmnlalu/MangARTI](https://huggingface.co/Jmnlalu/MangARTI) (full model card, usage examples and limitations)
- **Dataset:** [Jmnlalu/Bahasa-Manado-Alpaca-Translations](https://huggingface.co/datasets/Jmnlalu/Bahasa-Manado-Alpaca-Translations)
- **Base model:** [unsloth/llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)

## Repository layout

```
.
├── app.py                     # Web translator: FastAPI server + JSON API
├── train.py                   # LoRA fine-tuning with Unsloth + TRL
├── prompt.py                  # Alpaca prompt template and instructions (shared)
├── requirements.txt
├── adapter/
│   └── adapter_config.json    # LoRA config of the published adapter (weights are on Hugging Face)
├── assets/                    # Brand banner and logo
├── web/                       # Translator page (HTML, CSS, JS, icons)
├── data/
│   ├── raw/                   # Original sentence pairs, one "<sentence> /<label>" per line
│   ├── labeled.jsonl          # Parsed and hand-corrected pairs
│   └── translations.jsonl     # Final training set (6,908 Alpaca examples, both directions)
└── tools/
    ├── parse_raw.py           # raw/*.txt  -> labelled JSONL
    ├── dataset_editor.py      # Streamlit app to review and fix spellings
    └── build_translations.py  # labelled pairs -> bidirectional translation examples
```

## Setup

An NVIDIA GPU with CUDA is required, since the base model is 4-bit bitsandbytes.

```bash
git clone https://github.com/JomnLalu/MangARTI.git
cd MangARTI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Run the translator

```bash
python app.py                                  # downloads the adapter from Hugging Face
python app.py --model lora_translation_model   # or use an adapter you trained locally
python app.py --mock                           # work on the UI without a GPU
python app.py --host 0.0.0.0                   # make it reachable from other devices on your network
```

Open http://127.0.0.1:7860. The page opens straight away and shows the model's loading status until it is ready.

The page calls a small JSON API you can also use directly (interactive docs at `/api/docs`):

```bash
curl -X POST http://127.0.0.1:7860/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya mau makan nasi goreng.", "direction": "id-mdo"}'
# {"translation": "...", "seconds": 0.84}
```

`direction` is `id-mdo` (Indonesian → Manado) or `mdo-id` (Manado → Indonesian). Optional fields: `max_new_tokens` (default 128), `do_sample` (default false), `temperature`, `top_p`.

## Train

```bash
python train.py                                   # uses the dataset from Hugging Face
python train.py --data data/translations.jsonl    # or the local copy
python train.py --resume outputs/checkpoint-1678  # resume from a checkpoint
```

Checkpoints go to `outputs/` and the final adapter to `lora_translation_model/`. Both are git-ignored.

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 16 / 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 2 |
| Batch size | 1 × 8 gradient accumulation |
| Learning rate | 1e-4, cosine schedule, 50 warmup steps |
| Max sequence length | 512 |

## Dataset pipeline

```bash
# 1. Parse the raw text files (writes data/labeled.raw.jsonl; won't overwrite the edited file)
python tools/parse_raw.py

# 2. Review and standardise Manado spellings
streamlit run tools/dataset_editor.py

# 3. Build bidirectional translation examples
python tools/build_translations.py --labeled data/labeled.jsonl --out data/translations.new.jsonl
```

`data/translations.jsonl` is the published training set. It combines the pairs in `data/labeled.jsonl` with extra pairs added from CSV files, which `build_translations.py` can also read with `--csv`.

## Prompt format

The adapter was trained on the Alpaca template only (see [`prompt.py`](prompt.py)), with these exact instructions:

- `Translate the following sentence from Formal Indonesian to Manado dialect.`
- `Translate the following sentence from Manado dialect to Formal Indonesian.`

Don't use the tokenizer's chat template, and always set the direction explicitly. The model doesn't detect the input language.

## Limitations

- Built for short, conversational sentences. Not for paragraphs or for medical, legal or official text.
- Manado Malay has no standard spelling, so unfamiliar spellings and slang can be misread.
- Some words change meaning between varieties (*kita* = "I" in Manado, "we" in Indonesian). Using the wrong direction can flip the meaning silently.
- No quantitative evaluation has been published yet. Have a fluent speaker review anything important.

See the [model card](https://huggingface.co/Jmnlalu/MangARTI) for details.

## License

The adapter is released under the [Meta Llama 3 Community License](https://www.llama.com/llama3/license/), inherited from the base model, and its use must follow the [Llama 3 Acceptable Use Policy](https://www.llama.com/llama3/use-policy/). The dataset is released under the MIT license.

Built with Meta Llama 3.

## Author

Jonathan Immanuel Montolalu
