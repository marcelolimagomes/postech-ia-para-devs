from unsloth import FastLanguageModel
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B",               # Default
    "unsloth/tinyllama",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Based on title of a product, get the real description for the follow product.

### Input:
{}

### Response:
{}"""


def get_model_by_id(id, max_seq_length, dtype, load_in_4bit):
  """
    ID - Model Name
    0. unsloth/Meta-Llama-3.1-8B,               # Default
    1. unsloth/Meta-Llama-3.1-8B-bnb-4bit,      # Llama-3.1 15 trillion tokens model 2x faster!
    2. unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit,
    3. unsloth/Meta-Llama-3.1-70B-bnb-4bit,
    4. unsloth/Meta-Llama-3.1-405B-bnb-4bit,    # We also uploaded 4bit for 405b!
    5. unsloth/Mistral-Nemo-Base-2407-bnb-4bit, # New Mistral 12b 2x faster!
    6. unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit,
    7. unsloth/mistral-7b-v0.3-bnb-4bit,        # Mistral v3 2x faster!
    8. unsloth/mistral-7b-instruct-v0.3-bnb-4bit,
    9. unsloth/Phi-3.5-mini-instruct,           # Phi-3.5 2x faster!
    10. unsloth/Phi-3-medium-4k-instruct,
    11. unsloth/gemma-2-9b-bnb-4bit,
    12. unsloth/gemma-2-27b-bnb-4bit,            # Gemma 2x faster!  

  Args:
      id (int): Id listed above 

  Returns:
      model, tokenizer : model, tokenizer 
  """

  print(f'Id Model: {id} - Model Name: {fourbit_models[id]}')
  raw_model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=fourbit_models[id],
      max_seq_length=max_seq_length,
      dtype=dtype,
      load_in_4bit=load_in_4bit,
      revision='main'
      # token = "hf_..., # use one if using gated models like meta-llama/Llama-2-7b-hf
  )
  return fourbit_models[id], raw_model, tokenizer


def predict(model, tokenizer, title):
  # alpaca_prompt = Copied from above
  FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
  inputs = tokenizer(
      [
          alpaca_prompt.format(
              title,  # Title
              "",    # Description - leave this blank for generation!
              # "",    # UID Code
          )
      ], return_tensors="pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
  res = tokenizer.batch_decode(outputs)
  return res[0].split('\n')[-1]


def predict_text_streamer(model, tokenizer, title):
  # alpaca_prompt = Copied from above
  FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
  inputs = tokenizer(
      [
          alpaca_prompt.format(
              title,  # Title
              "",    # Description - leave this blank for generation!
              # "",    # UID Code
          )
      ], return_tensors="pt").to("cuda")

  from transformers import TextStreamer
  text_streamer = TextStreamer(tokenizer)
  _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

  return None


def print_start_memory_usage():
  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
  print(f"{start_gpu_memory} GB of memory reserved.")

  return start_gpu_memory, max_memory


def print_final_memory_usage(start_gpu_memory, max_memory, trainer_stats):
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory / max_memory * 100, 3)
  lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
  print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
  print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
  print(f"Peak reserved memory = {used_memory} GB.")
  print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
  print(f"Peak reserved memory % of max memory = {used_percentage} %.")
  print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def get_fast_language_model(raw_model):
  return FastLanguageModel.get_peft_model(
      raw_model,
      r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
      lora_alpha=32,
      lora_dropout=0,  # Supports any, but = 0 is optimized
      bias="none",    # Supports any, but = "none" is optimized
      # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
      use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
      random_state=3407,
      use_rslora=False,  # We support rank stabilized LoRA
      loftq_config=None,  # And LoftQ
  )
