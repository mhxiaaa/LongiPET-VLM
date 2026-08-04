import os
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/project/huggingface/hub"
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.llm_qwen import VLMQwenForCausalLM, VLMQwenConfig
from train.step2_trainVLM import vision_words, trainable_words, find_all_linear_names
import json
"""
module load miniconda
conda activate Med3DVLM
module load CUDA/12.6.0

python3 /home/mx79/project_pi_cl598/mx79/LongiPET-VLM-code/train/step4_mergeWeights.py
"""
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='Qwen/Qwen3-8B')
    model_with_lora: Optional[str] = field(default="/home/project/LongiPET-VLM-model_results/step3_trainVLM_vision/final_checkpoint/pytorch_model/mp_rank_00_model_states.pt")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default='/home/project/LongiPET-VLM-model_results/step1_pretrain_projector/mm_projector.bin')
    pretrain_vision_modules: str = field(default='/home/project/LongiPET-VLM-model_results/step0_vision/mp_rank_00_model_states.pt')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated."},)
    output_dir: str = "/home/project/LongiPET-VLM-model_results/model"
    saveModel_dir: str = "/home/project/LongiPET-VLM-model_results/model"


def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path, 
                                              cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length,
                                              padding_side="right", use_fast=False,)
    print("=" * 20 + " Tokenizer initlized from: " + model_args.model_name_or_path + "=" * 20)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<imPre_patch>"]})
    assert tokenizer.pad_token is not None
    assert tokenizer.bos_token is None
    
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.imgPre_token_id = tokenizer.convert_tokens_to_ids("<imPre_patch>")
    model_args.vocab_size = len(tokenizer)
    print(f"vocab_size: {model_args.vocab_size} | img_token_id: {model_args.img_token_id} | imgPre_token_id: {model_args.imgPre_token_id}")

    config = VLMQwenConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,)
    model, loading_info = VLMQwenForCausalLM.from_pretrained(model_args.model_name_or_path, config=config,
                                                            cache_dir=training_args.cache_dir, output_loading_info=True,)
    bad_missing = [k for k in loading_info["missing_keys"] if not any(s in k for s in vision_words)]
    assert not bad_missing, f"Unexpected missing keys: {bad_missing}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys found: {loading_info['unexpected_keys']}"

    model_args.num_new_tokens = 2
    model.initialize_vision_tokenizer(model_args, tokenizer)

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
                            target_modules=find_all_linear_names(model), lora_dropout=training_args.lora_dropout,
                            bias=training_args.lora_bias, task_type="CAUSAL_LM",)
    print("Adding LoRA adapters only on LLM.")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    checkpoint = torch.load(model_args.model_with_lora, map_location="cpu")
    if "module" in checkpoint and isinstance(checkpoint["module"], dict):  # unwrap DeepSpeed
        checkpoint = checkpoint["module"]
    model.load_state_dict(checkpoint, strict=True)
    print("=" * 20 + " model initilized from: " + model_args.model_with_lora + "=" * 20)

    print("Merge weights with LoRA")
    model = model.merge_and_unload()

    print("=" * 20 + " Save model " + "=" * 20)
    os.makedirs(training_args.saveModel_dir, exist_ok=True)
    
    model.config.architectures = [model.__class__.__name__]
    model._name_or_path = training_args.saveModel_dir
    
    print("Save trained")
    model.config.save_pretrained(training_args.saveModel_dir)
    model.save_pretrained(training_args.saveModel_dir)
    tokenizer.save_pretrained(training_args.saveModel_dir)
    print("Finish")

if __name__ == "__main__":
    main()
