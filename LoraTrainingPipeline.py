import json
import os
import torch
from datasets import Dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from huggingface_hub import login


class LoraTrainingPipeline:
        def __init__(self, device: str, hf_model: str):
            self.device = device
            self.hf_model = hf_model
            self.model: AutoModelForCausalLM.from_pretraine | None = None
            self.dataset: Dataset | None = None
            self.user_name = ""
            self.assistant_name = ""
            self.data_path = ""
            self.tokenizer: bool | PreTrainedTokenizerFast = False

        def _set_environment_variables(self):
            load_dotenv()

            self.data_path = os.getenv("DATA_PATH")
            self.user_name = os.getenv("USER_NAME")
            self.assistant_name = os.getenv("ASSISTANT_NAME")
            HF_TOKEN = os.getenv("HF_TOKEN")

            login(token=HF_TOKEN)

            # This is device specific
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device

        def _set_quantised_model(self):
            # load quantised model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32,
            )
            repo_id = self.hf_model
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
            self.model = model

            print(f"Model memory footprint 2: {model.get_memory_footprint() / 1e6}")

            train_p, tot_p = model.get_nb_trainable_parameters()
            print(f"Trainable parameters:      {train_p / 1e6:.2f}M")
            print(f"Total parameters:          {tot_p / 1e6:.2f}M")
            print(f"% of trainable parameters: {100 * train_p / tot_p:.2f}%")

        def _create_training_dataset(self):
            # load training dataset
            data_path = os.path.join(os.path.dirname(__file__), self.data_path)
            with open(data_path, "r") as f:
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

            user_messages = dataset.filter(lambda x: x["author"] == self.user_name)
            assistant_messages = dataset.filter(lambda x: x["author"] == self.assistant_name)

            training_dataset = []
            index = 0
            for _ in user_messages:
                conversation = [
                    {"role": "user", "content": user_messages[index]["text"]},
                    {"role": "assistant", "content": assistant_messages[index]["text"]},
                ]
                training_dataset.append(conversation)
                index += 1

            print("The first three items in the training dataset are:")
            print(training_dataset[0:2])

            processed_dataset = Dataset(training_dataset)
            self.dataset = processed_dataset

        def _set_tokenizer(self):
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(tokenizer))

            self.tokenizer = tokenizer
            # print(tokenizer.apply_chat_template(training_dataset, tokenize=False))

        def _run_training(self):
            # training
            logging_path = os.path.join(os.path.dirname(__file__), "logs")
            output_path = os.path.join(os.path.dirname(__file__), "output")
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
                logging_dir=logging_path,
                output_dir=output_path,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=sft_config,
                train_dataset=self.dataset,
            )

            # dl = trainer.get_train_dataloader()
            # batch = next(iter(dl))
            # print(f"Batch: {batch}")

            trainer.train()

        def run(self):
            try:
                self._set_environment_variables()
                self._set_quantised_model()
                self._create_training_dataset()
                self._set_tokenizer()
                self._run_training()

            except Exception as ex:
                print(f"An exception occurred: {ex}")

