import gradio as gr
from unsloth import FastLanguageModel
import torch
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 1. Load Model

print("Loading model and adapter to VRAM... Please wait.")

max_seq_length = 512
dtype = None 
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_translation_model", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"
print("Model loaded successfully!")

# 2. Prompt Template

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 3. Translation Function

def translate(text_to_translate, direction, max_new_tokens, do_sample, temperature, top_p):
    if not text_to_translate.strip():
        return "⚠️ Please enter text to translate."

    if direction == "Formal Indonesian -> Manado Dialect":
        instruction = "Translate the following sentence from Formal Indonesian to Manado dialect."
    else:
        instruction = "Translate the following sentence from Manado dialect to Formal Indonesian."

    formatted_prompt = alpaca_prompt.format(instruction, text_to_translate)
    
    inputs = tokenizer(
        [formatted_prompt], 
        return_tensors="pt",
        padding=True,
    ).to(model.device) 
    
    gen_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "use_cache": False,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample
    }
    
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        
    outputs = model.generate(**gen_kwargs)
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    final_translation = decoded_output.split("### Response:\n")[-1].strip()
    
    return final_translation

# 4. Ultra-Gradient Custom Theme & CSS

custom_theme = gr.themes.Soft().set(
    body_text_color="#FFFFFF",                     # Ensure base text is white
    block_background_fill="transparent",           
    block_border_color="rgba(91, 192, 190, 0.4)",  
    block_label_text_color="#6FFFE9",              
    block_title_text_color="#6FFFE9",              
    input_background_fill="transparent",           
    slider_color="#5BC0BE",
)

custom_css = """
/* 1. Main Animated Background */
body, .gradio-container {
    background: linear-gradient(270deg, #0B132B, #1C2541, #0a1128) !important;
    background-size: 400% 400% !important;
    animation: gradientBG 10s ease infinite !important;
    position: relative !important;
    overflow-x: hidden !important; /* Prevent horizontal scrollbar from appearing */
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 2. "DVD SCREENSAVER" EFFECT - Bouncing Glowing Orb */
/* We use 2 different animations (X and Y) to create random bounces */
.gradio-container::before {
    content: "";
    position: fixed;
    width: 1000px;
    height: 1000px;
    border-radius: 50%;
    /* Transparent radial gradient fading at the edges */
    background: radial-gradient(circle, rgba(111, 255, 233, 0.15) 0%, rgba(91, 192, 190, 0) 70%);
    top: 0;
    left: 0;
    pointer-events: none; /* Important! So the orb doesn't block your mouse clicks */
    z-index: 0; /* Place at the very back */
    animation: 
        orbMoveX 13s linear infinite alternate, 
        orbMoveY 19s linear infinite alternate !important;
}

@keyframes orbMoveX {
    0% { transform: translateX(-50px); }
    100% { transform: translateX(calc(100vw - 300px)); }
}
@keyframes orbMoveY {
    0% { transform: translateY(-50px); }
    100% { transform: translateY(calc(100vh - 300px)); }
}

/* Elevate UI elements so they are not covered by the orb */
.gradio-container .main {
    z-index: 1;
    position: relative;
}

/* 3. Glowing & Moving Header Text (FIXED) */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    background: linear-gradient(270deg, #5BC0BE, #6FFFE9, #9B9ECE, #5BC0BE) !important;
    background-size: 200% 100% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important; /* Force Gradio to remove default color */
    font-weight: 900 !important;
    text-align: center;
    animation: textShine 3s linear infinite !important;
}

@keyframes textShine {
    0% { background-position: 200% 50%; }
    100% { background-position: 0% 50%; }
}

#header-container { text-align: center; margin-bottom: 20px; z-index: 2; position: relative; }

/* 4. Input & Output Text Becomes Pure White (FIXED) */
textarea, input[type="text"] {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important; /* Force browser to use pure white */
    background: linear-gradient(180deg, rgba(11, 19, 43, 0.9) 0%, rgba(28, 37, 65, 0.9) 100%) !important;
    border: 1px solid rgba(91, 192, 190, 0.4) !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #6FFFE9 !important;
    box-shadow: 0 0 12px rgba(111, 255, 233, 0.6) !important;
}

/* Gradient Box Backgrounds (Back Panel) */
.gradio-container .block, .svelte-1b6s6s {
    background: linear-gradient(135deg, rgba(28, 37, 65, 0.6) 0%, rgba(11, 19, 43, 0.8) 100%) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(91, 192, 190, 0.2) !important;
    border-radius: 12px !important;
}

/* Animated Primary Button */
button.primary {
    background: linear-gradient(90deg, #1C2541, #3A506B, #5BC0BE, #1C2541) !important;
    background-size: 300% 100% !important;
    color: #FFFFFF !important; /* Button text changed to white */
    border: 1px solid #5BC0BE !important;
    font-weight: bold !important;
    font-size: 16px !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 4px 15px rgba(91, 192, 190, 0.3) !important;
    margin-top: 10px !important;
}

button.primary:hover {
    background-position: 100% 0 !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(111, 255, 233, 0.5) !important;
}

.wrap.gap-4 { justify-content: center !important; }
"""

# 5. Gradio User Interface

with gr.Blocks(theme=custom_theme, css=custom_css, title="Dialect Translator") as demo:
    
    with gr.Column(elem_id="header-container"):
        gr.Markdown("# 🌌 MangARTI Project")
        gr.Markdown("Created by Jonathan Montolalu")
    
    with gr.Row():
        direction = gr.Radio(
            choices=["Formal Indonesian -> Manado Dialect", "Manado Dialect -> Formal Indonesian"],
            value="Formal Indonesian -> Manado Dialect",
            label="Select Translation Direction",
            show_label=False,
            container=False
        )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=6, 
                label="Original Text", 
                placeholder="Type your sentence here..."
            )
            translate_btn = gr.Button("🚀 Translate Now", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(
                lines=6, 
                label="Translation Result", 
                interactive=False,
                placeholder="Translation result will appear here..."
            )

    with gr.Accordion("⚙️ Model Generation Parameters", open=False):
        with gr.Row():
            with gr.Column():
                max_tokens = gr.Slider(minimum=0, maximum=8192, value=256, step=256, label="Max New Tokens")
                do_sample = gr.Checkbox(value=True, label="Enable Sampling (Required for Top-p & Temp)")
            with gr.Column():
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)")

    translate_btn.click(
        fn=translate,
        inputs=[input_text, direction, max_tokens, do_sample, temperature, top_p],
        outputs=output_text
    )

# 6. Run Application

if __name__ == "__main__":
    demo.queue().launch()
