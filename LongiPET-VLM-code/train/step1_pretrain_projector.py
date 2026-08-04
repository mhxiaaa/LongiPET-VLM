import logging
import os, json
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_dir = '/home/project/'

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.multiscan_dataset import CapDataset, TextDatasets
from model.llm_qwen import VLMQwenForCausalLM, VLMQwenConfig, MLLMTrainer
import matplotlib.pyplot as plt

local_rank = None
def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='Qwen/Qwen3-8B')
    tune_mm_mlp_adapter: bool = field(default=True, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},)
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."},)
    pretrain_vision_modules: str = field(default='/home/project/LongiPET-VLM-model_results/step0_vision/mp_rank_00_model_states.pt') # load weights from step0
    resume_ckpt: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    # caption
    REPORT_train: List[str] = field(default_factory=list)
    REPORT_val:   List[str] = field(default_factory=list)
    REPORT_test:  List[str] = field(default_factory=list)
    # VQA
    VQA_cls_train: List[str] = field(default_factory=list)
    VQA_cls_val:   List[str] = field(default_factory=list)
    VQA_cls_test:  List[str] = field(default_factory=list)

    VQA_hpv_train: List[str] = field(default_factory=list)
    VQA_hpv_val:   List[str] = field(default_factory=list)
    VQA_hpv_test:  List[str] = field(default_factory=list)

    VQA_relapse_train: List[str] = field(default_factory=list)
    VQA_relapse_val:   List[str] = field(default_factory=list)
    VQA_relapse_test:  List[str] = field(default_factory=list)

    max_length: int = 4096
    proj_out_num: int = 512

    def __post_init__(self):
        with open(data_dir + "LongiPET-VLM-dataPath/REPORT.json", "r", encoding="utf-8") as f: 
            data = json.load(f)
        self.REPORT_train, self.REPORT_val, self.REPORT_test = data["train"], data["val"], data["test"]
        rank0_print("for train/val/test REPORT task:", len(self.REPORT_train), len(self.REPORT_val), len(self.REPORT_test))

        with open(data_dir + "LongiPET-VLM-dataPath/VQA_DiseaseDiagnosis.json", "r", encoding="utf-8") as f: 
            data = json.load(f)
        self.VQA_cls_train, self.VQA_cls_val, self.VQA_cls_test = data["train"], data["val"], data["test"]
        rank0_print("for train/val/test VQA_cls task:", len(self.VQA_cls_train), len(self.VQA_cls_val), len(self.VQA_cls_test))
        
        with open(data_dir + "LongiPET-VLM-dataPath/VQA_HPVstatus.json", "r", encoding="utf-8") as f: 
            data = json.load(f)
        self.VQA_hpv_train, self.VQA_hpv_val, self.VQA_hpv_test = data["train"], data["val"], data["test"]
        rank0_print("for train/val/test VQA_hpv task:", len(self.VQA_hpv_train), len(self.VQA_hpv_val), len(self.VQA_hpv_test))

        with open(data_dir + "LongiPET-VLM-dataPath/VQA_RFSstatus.json", "r", encoding="utf-8") as f: 
            data = json.load(f)
        self.VQA_relapse_train, self.VQA_relapse_val, self.VQA_relapse_test = data["train"], data["val"], data["test"]
        rank0_print("for train/val/test VQA_relapse task:", len(self.VQA_relapse_train), len(self.VQA_relapse_val), len(self.VQA_relapse_test))

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated."},)
    seed: int = 42
    optim: str = field(default="adamw_torch")

    bf16: bool = True
    output_dir: str = data_dir + "LongiPET-VLM-model_results/step1_pretrain_projector"

    num_train_epochs: float = 7
    per_device_train_batch_size: int = 9
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    eval_strategy: str = "no"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    
    save_strategy: str = "steps"
    save_steps: int = 300
    save_total_limit: int = 1
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict: 
        images, dose, input_ids, labels, attention_mask, images_gt, images_pre = tuple([b[key] for b in batch] 
                                for key in ("images", "dose", "input_id", "label", "attention_mask", "images_gt", "images_pre"))
        task_types = [b.get("task_type") for b in batch]

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        has_pre = [x is not None for x in images_pre] # list of bool, length b
        if any(has_pre):
            ref = next(x for x in images_pre if x is not None) # find the first non-None image_pre to get the shape, used as reference to fill None values with zeros
            images_pre = torch.stack([x if x is not None else torch.zeros_like(ref) for x in images_pre])
            images_pre_mask = torch.tensor(has_pre, dtype=torch.bool)
        else:
            images_pre, images_pre_mask = None, None
        images_gt = None if images_gt[0] is None else torch.cat([_.unsqueeze(0) for _ in images_gt], dim=0)
        dose = torch.tensor(dose, dtype=torch.float32, device=images.device)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0) # [b,4096]
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0) # [b,4096]
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0) # [b,4096]

        # Trim to longest real sequence in the batch to save attention memory.
        max_valid_len = int(attention_mask.sum(dim=1).max().item())
        input_ids = input_ids[:, :max_valid_len] # [b, max_valid_len]
        labels = labels[:, :max_valid_len]
        attention_mask = attention_mask[:, :max_valid_len]
        # plt.imsave('y1.png',images[0,0,:,:,64].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        return dict(images=images, dose=dose, input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                    task_type=task_types, images_gt=images_gt, images_pre=images_pre, images_pre_mask=images_pre_mask)

def print_trainable_parameters(model):
    rank0_print("=" * 20 + " Trainable parameters " + "=" * 20)
    total_numel = 0
    trainable_numel = 0
    for n, p in model.named_parameters():
        n_params = p.numel()
        total_numel += n_params
        if p.requires_grad:
            trainable_numel += n_params
            rank0_print(f"[trainable] {n} {tuple(p.shape)}")
    rank0_print(f"trainable params: {trainable_numel:,}")
    rank0_print(f"total params:     {total_numel:,}")
    rank0_print(f"trainable ratio:  {100.0 * trainable_numel / total_numel:.4f}%")

def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    if local_rank in [0, -1]:
        os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,
                                              model_max_length=training_args.model_max_length, padding_side="right", use_fast=True,)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<imPre_patch>"]})
    assert tokenizer.pad_token is not None
    assert tokenizer.bos_token is None
    
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.imgPre_token_id = tokenizer.convert_tokens_to_ids("<imPre_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print(f"vocab_size: {model_args.vocab_size} | img_token_id: {model_args.img_token_id} | imgPre_token_id: {model_args.imgPre_token_id}")

    rank0_print("="*20 + " Model preparation " + "="*20)
    config = VLMQwenConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    model, loading_info = VLMQwenForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, 
                                                             cache_dir=training_args.cache_dir, output_loading_info=True)
    bad_missing = [k for k in loading_info["missing_keys"] if not any(s in k for s in 
                                                ("vision_encoder", "mm_projector", "seg_", "den_",
                                                "bce_pos_weight", "dice_loss", "ssim_loss"))]
    assert not bad_missing, f"Unexpected missing keys: {bad_missing}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys found: {loading_info['unexpected_keys']}"
    model.config.use_cache = False

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing: model.gradient_checkpointing_enable()
     
    model.get_model().initialize_vision_modules(model_args=model_args)
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    assert model_args.tune_mm_mlp_adapter == True
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    model_args.num_new_tokens = 2
    model.initialize_vision_tokenizer(model_args, tokenizer)
    # Re-freeze embed_tokens: its 623M-param gradient (~2.5 GB fp32) causes an
    # OOM-induced NCCL hang during all-reduce. Projector outputs overwrite the
    # <im_patch> embeddings anyway, so training embed_tokens here is useless.
    model.get_input_embeddings().weight.requires_grad_(False)

    print_trainable_parameters(model)

    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    
    train_dataset = TextDatasets(data_args, tokenizer, mode="train")
    eval_dataset = CapDataset(data_args, tokenizer, mode="val")
    data_collator = DataCollator()

    rank0_print("=" * 20 + " Training " + "=" * 20)
    trainer = MLLMTrainer(model=model, args=training_args, data_collator=data_collator,
                          train_dataset=train_dataset, eval_dataset=eval_dataset, 
                          compute_metrics=None, preprocess_logits_for_metrics=None)
    
    trainer.train(resume_from_checkpoint=model_args.resume_ckpt)
    # trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
