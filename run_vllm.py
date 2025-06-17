import os
# Set environment variables to disable PyTorch compilation
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE"] = "0" 
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["VLLM_USE_COMPILATION"] = "0"

from vllm import LLM, SamplingParams
def format_prompt(text):
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

raw_prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is a",
]

prompts = [format_prompt(p) for p in raw_prompts]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
    dtype="auto",
    max_model_len=8192,
    tensor_parallel_size=1,
    enforce_eager=True
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    full_prompt = output.prompt
    generation = output.outputs[0].text.strip()

    # Extract user message from chat-formatted prompt
    if "<|im_start|>user" in full_prompt:
        user_msg = full_prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
    else:
        user_msg = full_prompt.strip()

    print(f"Prompt: {user_msg}")
    print(f"LLM: {generation}")
    print("-------------")

