"""Prompt template and instructions shared by training, the demo app and the dataset tools.

The instructions must match the training data exactly, or the adapter will not recognise them.
"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

TO_MANADO = "Translate the following sentence from Formal Indonesian to Manado dialect."
TO_INDONESIAN = "Translate the following sentence from Manado dialect to Formal Indonesian."
