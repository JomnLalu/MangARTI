---
base_model: unsloth/llama-3-8b-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
thumbnail: "https://share.jomontolalu.com/mangarti/logo.png"
license: llama3
datasets:
- Jmnlalu/Bahasa-Manado-Alpaca-Translations
language:
- id
- xmm
tags:
- base_model:adapter:unsloth/llama-3-8b-Instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- translation
- bahasa-indonesia
- manadonese
---

![MangARTI: translation between Formal Indonesian and Manado Malay](https://share.jomontolalu.com/mangarti/banner.png)

# MangARTI

MangARTI is a LoRA adapter for Llama 3 8B Instruct that translates single sentences between Formal Indonesian and Manado Malay (Bahasa Manado), the everyday language of Manado and much of North Sulawesi.

The name comes from *mangarti*, the Manado Malay word for *mengerti* ("to understand").

## Model details

| Property | Value |
|---|---|
| Developer | Jonathan Immanuel Montolalu |
| Model type | LoRA adapter for a causal language model |
| Base model | [`unsloth/llama-3-8b-Instruct-bnb-4bit`](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit), a 4-bit build of Meta Llama 3 8B Instruct |
| Languages | Formal Indonesian (`id`), Manado Malay (`xmm`) |
| Directions | Formal Indonesian → Manado Malay, Manado Malay → Formal Indonesian |
| Prompt format | Alpaca (instruction, input, response) |
| Adapter size | 168 MB |
| License | Meta Llama 3 Community License |

## Intended use

MangARTI is built for short, conversational sentences: the kind of thing people say at home, at the market or in a chat. Typical uses:

- Translation features in apps and chatbots that serve users in North Sulawesi
- Learning tools for people studying Manado Malay or Formal Indonesian
- Research on low-resource Indonesian regional varieties

### Out of scope

- Medical, legal, financial or official documents
- Paragraphs, articles or any multi-sentence input
- Other languages of North Sulawesi, such as Tombulu, Tontemboan or Sangirese, which are separate languages, not Manado Malay
- Open-ended chat or tasks other than translation

## Quick start

### Requirements

```bash
pip install -U transformers peft accelerate bitsandbytes
```

The base model is stored in 4-bit bitsandbytes format, so plan on an NVIDIA GPU with CUDA. A free Google Colab T4 is enough.

### Translate a sentence

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "Jmnlalu/MangARTI"

# Loads the 4-bit base model and applies the MangARTI adapter on top.
model = AutoPeftModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# These must match the training data exactly.
INSTRUCTIONS = {
    "to_manado": "Translate the following sentence from Formal Indonesian to Manado dialect.",
    "to_indonesian": "Translate the following sentence from Manado dialect to Formal Indonesian.",
}


def translate(sentence: str, direction: str = "to_manado") -> str:
    prompt = ALPACA_PROMPT.format(INSTRUCTIONS[direction], sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode only the newly generated tokens, not the prompt.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print(translate("Saya tidak tahu mau pergi ke mana hari ini."))
print(translate("Ngana mo pigi ke mana?", direction="to_indonesian"))
```

### Faster inference with Unsloth

If Unsloth is installed, load the model this way instead. The `translate` function above works unchanged.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Jmnlalu/MangARTI",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

### Prompt format

The adapter was trained on the Alpaca template only. Do not use `tokenizer.apply_chat_template`: the chat template bundled with the tokenizer is the standard Llama 3 format, so prompts built with it will not match what the adapter learned.

The model does not detect the input language. Always choose the direction that matches the sentence you pass in.

## Training

### Data

The adapter was trained on [Jmnlalu/Bahasa-Manado-Alpaca-Translations](https://huggingface.co/datasets/Jmnlalu/Bahasa-Manado-Alpaca-Translations) (MIT license): 6,908 instruction examples of paired sentences, covering both translation directions. Inputs range from 3 to 407 characters.

Examples from the training data:

| Formal Indonesian | Manado Malay |
|---|---|
| Saya mau makan nasi goreng. | Kita mo makan nasi goreng. |
| Kamu mau pergi ke mana? | Ngana mo pigi ke mana? |
| Mereka belum tidur dari tadi. | Dorang belum tidor dari tadi. |
| Kami sedang menonton televisi. | Torang ada bauni tv. |
| Saya tidak mengerti maksudmu. | Kita nda mangarti ngape maksud. |

### Procedure

Supervised fine-tuning with Unsloth and TRL on the 4-bit base model, using these LoRA settings:

| Setting | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| PEFT version | 0.18.1 |

## Evaluation

No quantitative evaluation (such as BLEU or chrF on a held-out test set) has been published for this release. The dataset has a single training split, so there is no official test set yet. Treat output quality as unmeasured and review translations before relying on them.

## Limitations

### Words that change meaning between the two varieties

Some common words mean different things in each variety. In Manado Malay, *kita* means "I", while in Formal Indonesian it means "we". Manado Malay uses *torang* for "we". If a sentence is sent with the wrong direction, its meaning can flip without any sign of an error.

### Informal spelling

Manado Malay has no standard written form, and people type it many ways: *ngana* or *nga* for "you", *skali* or *sekali* for "very". Chat abbreviations and new slang are not in the training data. The model handles spellings close to those in the dataset best and can misread or invent meaning for unfamiliar forms.

### Sentence length

The training data contains single sentences only. With paragraphs or longer passages, quality drops: the model can lose context, skip parts of the input or add content that is not in the source.

### General model risks

Like any language model, MangARTI can produce fluent output that is wrong, and it can reflect biases in its training data and its base model.

## Recommendations

- Send one sentence per call. Split paragraphs before translating.
- Set the translation direction explicitly for every request.
- Mark translations as AI-generated in any product that shows them to users.
- Have a fluent speaker review anything that will be published or acted on.

## License

The adapter is released under the [Meta Llama 3 Community License](https://dev.meta.ai/llama/llama3/license), inherited from the base model. Use must also follow the [Llama 3 Acceptable Use Policy](https://dev.meta.ai/llama/llama3/use-policy). The training dataset is released separately under the MIT license.

Built with Meta Llama 3.
