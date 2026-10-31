# =====================================================================
# QLoRA Fine-Tuning (English + मराठी explanation)
# =====================================================================
# QLoRA = Quantized LoRA.
# Idea: LoRA प्रमाणेच adapters train करायचे, पण BASE MODEL ला 4-bit मध्ये compress (quantize) करून
#       GPU memory मध्ये load करायचे. Frozen base model 4-bit मध्ये असतो, आणि LoRA adapters 16-bit मध्ये train होतात.
# Result: 7B model सुद्धा ~6-8 GB VRAM मध्ये fine-tune होतो (free Colab T4 वर शक्य!).
#
# LoRA vs QLoRA फरक:
#   LoRA  -> base model 16-bit मध्ये. जास्त memory, थोडी जास्त accuracy, थोडा fast.
#   QLoRA -> base model 4-bit मध्ये. खूप कमी memory, थोडा slow, accuracy जवळजवळ सारखीच.
#
# Install (Colab / Linux + NVIDIA GPU लागते, bitsandbytes साठी):
#   pip install -q transformers peft trl datasets accelerate bitsandbytes
# =====================================================================

# ---------------------------- 1. IMPORTS ------------------------------
import torch
# PyTorch: tensors, dtype आणि GPU साठी.

from datasets import load_dataset
# HuggingFace dataset load करण्यासाठी.

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# BitsAndBytesConfig = NEW! Quantization (4-bit compression) चे settings ठेवणारा class.

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# prepare_model_for_kbit_training = NEW! 4-bit model ला training साठी तयार करतो
# (layer norms ला float32 करतो, gradient checkpointing चालू करतो, input grads enable करतो).

from trl import SFTTrainer, SFTConfig
# Supervised fine-tuning trainer आणि त्याचे config.

# ---------------------------- 2. CONFIG -------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Base model. मोठा हवा असेल तर "mistralai/Mistral-7B-v0.1" सारखा 7B model बदलून बघा; QLoRA मुळे तो T4 वर बसेल.

DATASET_NAME = "mlabonne/guanaco-llama2-1k"
# 1k examples चा instruction dataset ("text" column).

OUTPUT_DIR = "./tinyllama-qlora"
# Output folder.

# ---------------------------- 3. TOKENIZER ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Tokenizer load.

tokenizer.pad_token = tokenizer.eos_token
# Pad token नसेल तर EOS token ला pad म्हणून वापर.

tokenizer.padding_side = "right"
# Padding उजवीकडे.

# ---------------------------- 4. QUANTIZATION CONFIG (QLoRA चा गाभा) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # Model weights 4-bit मध्ये load कर. 16-bit ला 4x कमी memory लागते.

    bnb_4bit_quant_type="nf4",
    # Quantization प्रकार: NF4 = "NormalFloat 4-bit". Neural network weights साधारण normal distribution
    # follow करतात, म्हणून NF4 त्यासाठी optimize केलेला आहे -> साध्या fp4 पेक्षा चांगली quality.

    bnb_4bit_compute_dtype=torch.float16,
    # Storage 4-bit आहे, पण actual calculation (matrix multiply) float16 मध्ये होते.
    # (Ampere GPU / A100 असेल तर torch.bfloat16 वापरा.)

    bnb_4bit_use_double_quant=True,
    # DOUBLE QUANTIZATION: quantization constants ना सुद्धा quantize करतो -> आणखी ~0.4 bits/param memory वाचते.
)

# ---------------------------- 5. LOAD 4-BIT MODEL ---------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # Model name.
    quantization_config=bnb_config,
    # वरचे 4-bit settings apply कर -> model load होताना लगेच compress होतो.
    device_map={"": 0},
    # पूर्ण model GPU 0 वर ठेव. (QLoRA साठी GPU आवश्यक आहे.)
)

model.config.use_cache = False
# Training मध्ये KV cache बंद (gradient checkpointing सोबत conflict टाळण्यासाठी).

model = prepare_model_for_kbit_training(model)
# 4-bit model ला stable training साठी तयार करतो:
#   - काही layers float32 मध्ये ठेवतो, - gradient checkpointing चालू करतो (memory वाचवतो).

# ---------------------------- 6. LoRA CONFIG --------------------------
lora_config = LoraConfig(
    r=16,
    # Rank: adapter चा आकार. 8/16/32/64.

    lora_alpha=32,
    # Scaling factor (update ला alpha/r ने scale). Thumb rule: 2 x r.

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # QLoRA paper प्रमाणे सगळ्या linear layers ला adapter लावा:
    # attention (q,k,v,o) + MLP (gate, up, down). यामुळे quality LoRA-attention-only पेक्षा चांगली येते.

    lora_dropout=0.05,
    # 5% dropout -> overfitting कमी.

    bias="none",
    # Bias train करू नकोस.

    task_type="CAUSAL_LM",
    # Next-token prediction task.
)

model = get_peft_model(model, lora_config)
# 4-bit frozen model वर 16-bit LoRA adapters add होतात. फक्त तेच train होतील.

model.print_trainable_parameters()
# Trainable parameters % print कर. (~1% पेक्षा कमी.)

# ---------------------------- 7. DATASET ------------------------------
dataset = load_dataset(DATASET_NAME, split="train")
# Train split load.

# ---------------------------- 8. TRAINING ARGS ------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # Output folder.

    num_train_epochs=1,
    # 1 epoch.

    per_device_train_batch_size=2,
    # QLoRA मध्ये memory आधीच वाचलेली असते, तरी T4 साठी 2 सुरक्षित. OOM आले तर 1 कर.

    gradient_accumulation_steps=8,
    # Effective batch = 2 x 8 = 16.

    learning_rate=2e-4,
    # QLoRA paper मध्ये 2e-4 वापरला आहे.

    optim="paged_adamw_8bit",
    # NEW! PAGED 8-bit AdamW optimizer: optimizer states 8-bit मध्ये आणि memory spike आला तर
    # CPU RAM मध्ये page-out होतात -> out-of-memory errors टळतात.

    lr_scheduler_type="cosine",
    # Cosine LR decay.

    warmup_ratio=0.03,
    # पहिले 3% steps warmup.

    max_grad_norm=0.3,
    # Gradient clipping: gradients 0.3 पेक्षा मोठे असतील तर कापतो -> training stable राहते.

    gradient_checkpointing=True,
    # Forward pass चे सगळे activations साठवण्याऐवजी backward ला पुन्हा calculate करतो
    # -> memory खूप वाचते, थोडा वेळ जास्त लागतो.

    logging_steps=10,
    # प्रत्येक 10 steps ला loss print.

    save_strategy="epoch",
    # प्रत्येक epoch ला checkpoint.

    fp16=True,
    # Mixed precision (bf16 support असेल तर bf16=True).

    max_length=512,
    # Max tokens per example. (जुन्या trl मध्ये max_seq_length.)

    dataset_text_field="text",
    # Training text चा column.

    report_to="none",
    # External logging बंद.
)

# ---------------------------- 9. TRAINER ------------------------------
trainer = SFTTrainer(
    model=model,
    # QLoRA model (4-bit base + LoRA).
    train_dataset=dataset,
    # Data.
    args=training_args,
    # Settings.
    processing_class=tokenizer,
    # Tokenizer. (जुन्या trl मध्ये `tokenizer=tokenizer`.)
)

# ---------------------------- 10. TRAIN -------------------------------
trainer.train()
# Training सुरू. Backward pass 4-bit weights मधून जातो (dequantize on the fly), पण update फक्त LoRA weights चे होतात.

# ---------------------------- 11. SAVE --------------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
# फक्त QLoRA adapter save (काही MB).

tokenizer.save_pretrained(OUTPUT_DIR)
# Tokenizer save.

# ---------------------------- 12. INFERENCE TEST ----------------------
prompt = "<s>[INST] Explain what QLoRA is in two lines. [/INST]"
# Test prompt (Llama-2 [INST] format).

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Tokenize करून GPU वर पाठव.

model.config.use_cache = True
# Generation साठी cache चालू.

with torch.no_grad():
    # Inference मध्ये gradients नको.
    output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    # 100 new tokens generate, थोडे sampling सह.

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
# Token IDs -> readable text.

# ---------------------------- 13. (OPTIONAL) MERGE --------------------
# IMPORTANT (महत्त्वाचे): 4-bit model वर थेट merge करू नका (quality जाते).
# Merge करायचा असेल तर base model पुन्हा 16-bit मध्ये load करा आणि adapter merge करा:
#
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
# # Original model पुन्हा 16-bit (non-quantized) मध्ये load.
# merged = PeftModel.from_pretrained(base, OUTPUT_DIR).merge_and_unload()
# # Saved adapter load कर आणि base मध्ये merge कर.
# merged.save_pretrained("./tinyllama-qlora-merged")
# # Final merged model save.
