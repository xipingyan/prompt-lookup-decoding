from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import time

model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
model_name_or_path = "/mnt/data_nvme1n1p1/xiping_workpath/models/mistralai/Mistral-7B-Instruct-v0.1"
 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
# )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

# Manually set the pad_token_id to the eos_token_id
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

article = '''
What is the capital of China? 
'''
 
chat = [
    {
        "role": "system",
        "content": "You are Hermes 2, an unbiased AI assistant, and you always answer to the best of your ability."
    },
    {
        "role": "user",
        "content": (
            f"{article} Please answer like: the capital of English is london."
        )
    },
]
processed_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(processed_chat, return_tensors='pt').to(model.device)
 
streamer = TextStreamer(tokenizer)
 
# Run and measure generate
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
torch.cuda.reset_peak_memory_stats(model.device)
torch.cuda.empty_cache()
torch.cuda.synchronize()
start_event.record()

for i in range(6):
    # streamer=streamer,
    t1 = time.time()
    # generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512)
    generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, prompt_lookup_num_tokens=4, max_matching_ngram_size=2)
    t2 = time.time()

    decoded_outputs = tokenizer.decode(generation_output[0][input_ids.input_ids.shape[1]:generation_output.shape[1]], skip_special_tokens=True)
    print(f"== loop: {i}, tm : {t2-t1:.3f} s, decoded_outputs =", decoded_outputs)

end_event.record()
torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated(model.device)
print("Max memory (MB): ", max_memory * 1e-6)
new_tokens = generation_output.shape[1] - input_ids.input_ids.shape[1]
print("Throughput (tokens/sec): ", new_tokens / (start_event.elapsed_time(end_event) * 1.0e-3))


