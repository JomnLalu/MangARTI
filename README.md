---
base_model: unsloth/llama-3-8b-Instruct-bnb-4bit
library_name: peft
pipeline_tag: translation
thumbnail: "https://mangarti.jomontolalu.com/assets/logo.png"
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
license: llama3
datasets:
- Jmnlalu/Bahasa-Manado-Alpaca-Translations
language:
- id
---

<p align="center">
  <img src="https://share.jomontolalu.com/mangarti/banner.png" alt="MangARTI Banner" width="100%">
</p>

# MangARTI

This is a LoRA fine-tuned model based on `unsloth/llama-3-8b-Instruct-bnb-4bit`. It is specifically trained to translate text between Formal Bahasa Indonesia and the Manado dialect (Dialek Manado, North Sulawesi).

### Model Description

MangARTI leverages the Llama-3-8B architecture and has been fine-tuned using Unsloth and PEFT (LoRA). It understands specific instructions formatted in the Alpaca template to seamlessly translate sentences from Formal Indonesian to Manadonese, and vice versa. 

- **Developed by:** Jonathan Immanuel Montolalu
- **Model type:** Causal Language Model (LoRA Fine-tune)
- **Language(s) (NLP):** Indonesian (ind), Manado Malay (xmm)
- **License:** llama3
- **Finetuned from model:** unsloth/llama-3-8b-Instruct-bnb-4bit

## Uses

### Direct Use

The primary use case for this model is text translation. It can be used by developers, researchers, or locals looking to build applications that bridge the communication gap between standard formal Indonesian and the regional dialect of Manado.

### Out-of-Scope Use

This model is not intended for high-stakes, professional medical, or legal translations. Because it is an 8B parameter model trained on a specific regional dialect, it may hallucinate or struggle with highly technical jargon outside of standard everyday conversational contexts.

### Training Data

The model was trained on a custom dataset containing paired sentences in Formal Bahasa Indonesia and Dialek Manado. The dataset was formatted using an instruction-based Alpaca template. 

You can find the open-source dataset used to train this model here: [Jmnlalu/Bahasa-Manado-Alpaca-Translations](https://huggingface.co/datasets/Jmnlalu/Bahasa-Manado-Alpaca-Translations)

## Bias, Risks, and Limitations

Large language models can occasionally generate inaccurate translations or adopt biases present in their training data. Dialects often rely heavily on cultural context, slang, and tone, which may not always map 1:1 with formal Bahasa Indonesia. Users should verify critical translations. 

**Dynamic Language & Organic Typing**
The nature of local dialects like Bahasa Manado is highly dynamic and constantly evolving. Daily text messaging and casual chatting by locals often involve highly organic, non-standard spelling, and rapidly changing slang. Because of the sheer unpredictability and variance of this organic typing in the wild, the model can easily hallucinate, misinterpret context, or fail to recognize highly informal abbreviations.

**Input Length Limitations**
MangARTI is **not** designed or trained to translate long paragraphs, articles, or extensive documents. The underlying dataset focused exclusively on single-sentence translations. Feeding the model multi-sentence paragraphs or large blocks of text will likely degrade the output quality, cause the model to lose context, or trigger hallucinations. 

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. When using the model in user-facing applications, it is recommended to include a disclaimer that the translations are AI-generated. To get the best results, users should input text one sentence at a time and attempt to use relatively standardized spelling even when typing in the local dialect.

## How to Get Started with the Model

Use the code below to get started with the model. It requires the Alpaca prompt format to work correctly.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Jomnlalu/MangARTI"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Choose your direction
direction = "Bahasa Indonesia Formal -> Dialek Manado"
text_to_translate = "Saya tidak tahu mau pergi ke mana hari ini."

if direction == "Bahasa Indonesia Formal -> Dialek Manado":
    instruction = "Translate the following sentence from Formal Indonesian to Manado dialect."
else:
    instruction = "Translate the following sentence from Manado dialect to Formal Indonesian."

inputs = tokenizer(
[
    alpaca_prompt.format(instruction, text_to_translate, "")
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs, skip_special_tokens = True)[0])
