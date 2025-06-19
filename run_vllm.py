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

# Configure KV cache quantization
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tokenizer="meta-llama/Llama-2-7b-hf",
    # model_type="llama",
    dtype="auto",
    max_model_len=8192,
    tensor_parallel_size=1,
    enforce_eager=True,
    # KV cache quantization settings
    kv_cache_dtype="int8",  # Enable int8 quantization
    kv_quant_group=64,      # Group size for quantization
    #kv_quant_params_path="examples/int8/work_dir/ceval/kv_cache_scales_layer_level.json"

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

