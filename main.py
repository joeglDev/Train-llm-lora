import json
import os
import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login

# environment variables
load_dotenv()

CUDA_DEVICE = os.getenv("CUDA_DEVICE")  # todo: move this to class field
DATA_PATH = os.getenv("DATA_PATH")
USER_NAME = os.getenv("USER_NAME")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)

# This is device specific
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# load quantised model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
repo_id = "pankajmathur/orca_mini_3b" # "mistralai/Mistral-7B-Instruct-v0.3" # todo: move this to class field
model = AutoModelForCausalLM.from_pretrained(
    repo_id, device_map="cuda:0", quantization_config=bnb_config
)

print(f"Model memory footprint: {model.get_memory_footprint() / 1e6}")

# set up lora adaptor
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    # the rank of the adapter, the lower the fewer parameters you'll need to train
    r=8,
    lora_alpha=16,  # multiplier, usually 2*r
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # Newer models, such as Phi-3 at time of writing, may require
    # manually setting target modules
    target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
)

model = get_peft_model(model, config)

print(f"Model memory footprint 2: {model.get_memory_footprint() / 1e6}")
train_p, tot_p = model.get_nb_trainable_parameters()
print(f"Trainable parameters:      {train_p / 1e6:.2f}M")
print(f"Total parameters:          {tot_p / 1e6:.2f}M")
print(f"% of trainable parameters: {100 * train_p / tot_p:.2f}%")

# load training dataset
with open(os.path.join(os.path.dirname(__file__), DATA_PATH), "r") as f:
    data = json.load(f)

# Transform the nested structure into flat samples
samples = []
conversation_id = 0

for conversation in data:
    for message in conversation:
        sample = {
            "conversation_id": conversation_id,
            "author": message["author"],
            "text": message["text"],
        }
        samples.append(sample)

    # Increment conversation ID after processing each conversation
    conversation_id += 1

# Create dataset from the transformed samples
dataset = Dataset.from_dict(
    {
        "text": [sample["text"] for sample in samples],
        "author": [sample["author"] for sample in samples],
    }
)

user_messages = dataset.filter(lambda x: x["author"] == USER_NAME)
assistant_messages = dataset.filter(lambda x: x["author"] == ASSISTANT_NAME)

joined_user_messages = "\n".join(user_messages["text"])
joined_assistant_messages = "\n".join(assistant_messages["text"])

training_dataset = []
index = 0
for item in enumerate(user_messages):
    conversation = [
        {"role": "user", "content": user_messages[index]["text"]},
        {"role": "assistant", "content": assistant_messages[index]["text"]},
    ]
    training_dataset.append(conversation)
    index += 1

print("The first three items in the training dataset are:")
print(training_dataset[0:2])

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

# print(tokenizer.apply_chat_template(training_dataset, tokenize=False))

# training
sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,  # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=16,
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,
    ## GROUP 2: Dataset-related
    max_seq_length=64,
    # Dataset
    # packing a dataset means no padding is needed
    packing=True,
    ## GROUP 3: These are typical training parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim="paged_adamw_8bit",
    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir="./logs",
    output_dir="./output",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset,
)

dl = trainer.get_train_dataloader()
batch = next(iter(dl))

trainer.train()
