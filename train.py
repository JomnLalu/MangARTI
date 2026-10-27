"""Fine-tune Llama 3 8B Instruct with a LoRA adapter for Indonesian <-> Manado translation.

Usage:
    python train.py                               # train on the Hugging Face dataset
    python train.py --data data/translations.jsonl  # train on the local copy
    python train.py --resume outputs/checkpoint-1678
"""

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from prompt import ALPACA_PROMPT

BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
HF_DATASET = "Jmnlalu/Bahasa-Manado-Alpaca-Translations"
MAX_SEQ_LENGTH = 512
SEED = 3407


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="Local JSONL file to train on (default: the Hugging Face dataset)")
    parser.add_argument("--output-dir", default="outputs", help="Where checkpoints are written")
    parser.add_argument("--save-dir", default="lora_translation_model", help="Where the final adapter is saved")
    parser.add_argument("--resume", help="Checkpoint directory to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    # 4-bit loading keeps the 8B model within a laptop GPU's VRAM (trained on an RTX 5050).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    # The EOS token teaches the model where a translation ends.
    eos = tokenizer.eos_token

    def format_examples(batch):
        texts = [
            ALPACA_PROMPT.format(instruction, source) + target + eos
            for instruction, source, target in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        return {"text": texts}

    if args.data:
        dataset = load_dataset("json", data_files=args.data, split="train")
    else:
        dataset = load_dataset(HF_DATASET, split="train")
    dataset = dataset.shuffle(seed=SEED).map(format_examples, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            num_train_epochs=2,
            save_strategy="steps",
            save_steps=350,
            save_total_limit=5,
            learning_rate=1e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            output_dir=args.output_dir,
        ),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"Saving LoRA adapter to {args.save_dir}/")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Done!")


if __name__ == "__main__":
    main()
