import os
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_dir = '/home/project/'
import json
import random
import shutil
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.multiscan_dataset import CapDataset, VQADataset
from model.llm_qwen import VLMQwenForCausalLM, VLMQwenConfig
import matplotlib.pyplot as pl
local_rank = None

def rank0_print(*args):
    if local_rank in [0, -1, None]:
        print(*args)

def cycle(loader):
    while True:
        for batch in loader:
            yield batch

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='Qwen/Qwen3-8B')
    resume_ckpt: Optional[str] = field(default=None)
    pre_ckpt: Optional[str] = field(default=None)
    pretrain_vision_modules: str = field(default='/home/project/LongiPET-VLM-model_results/step0_vision/mp_rank_00_model_states.pt')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default='/home/project/LongiPET-VLM-model_results/step1_pretrain_projector/mm_projector.bin')

@dataclass
class DataArguments:
    REPORT_train: List[str] = field(default_factory=list)
    REPORT_val:   List[str] = field(default_factory=list)
    REPORT_test:  List[str] = field(default_factory=list)
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
        _is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0

        with open(data_dir + "LongiPET-VLM-dataPath/REPORT.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.REPORT_train, self.REPORT_val, self.REPORT_test = data["train"], data["val"], data["test"]
        self.REPORT_train = self.REPORT_train + self.REPORT_val + self.REPORT_train + self.REPORT_val # augment report training
        random.shuffle(self.REPORT_train)
        if _is_rank0: print("for train/val/test REPORT task:", len(self.REPORT_train), len(self.REPORT_val), len(self.REPORT_test))

        with open(data_dir + "LongiPET-VLM-dataPath/VQA_DiseaseDiagnosis.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.VQA_cls_train, self.VQA_cls_val, self.VQA_cls_test = data["train"], data["val"], data["test"]
        self.VQA_cls_train = self.VQA_cls_train + self.VQA_cls_val + self.VQA_cls_train + self.VQA_cls_val
        if _is_rank0: print("for train/val/test VQA_cls task:", len(self.VQA_cls_train), len(self.VQA_cls_val), len(self.VQA_cls_test))

        with open(data_dir + "LongiPET-VLM-dataPath/VQA_HPVstatus.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.VQA_hpv_train, self.VQA_hpv_val, self.VQA_hpv_test = data["train"], data["val"], data["test"]
        self.VQA_hpv_train = self.VQA_hpv_train + self.VQA_hpv_val + self.VQA_hpv_train + self.VQA_hpv_val
        if _is_rank0: print("for train/val/test VQA_hpv task:", len(self.VQA_hpv_train), len(self.VQA_hpv_val), len(self.VQA_hpv_test))

        with open(data_dir + "LongiPET-VLM-dataPath/VQA_RFSstatus.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.VQA_relapse_train, self.VQA_relapse_val, self.VQA_relapse_test = data["train"], data["val"], data["test"]
        self.VQA_relapse_train = self.VQA_relapse_train + self.VQA_relapse_val + self.VQA_relapse_train + self.VQA_relapse_val
        if _is_rank0: print("for train/val/test VQA_relapse task:", len(self.VQA_relapse_train), len(self.VQA_relapse_val), len(self.VQA_relapse_test))

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded/truncated."},)
    seed: int = 37

    bf16: bool = True
    output_dir: str = data_dir + "LongiPET-VLM-model_results/step2_trainVLM"

    train_epochs_report: float = 10
    train_epochs_VQA: float = 5

    train_batch_size_report: int = 8
    train_batch_size_VQA: int = 18

    dataloader_num_workers: int = 8
    gradient_accumulation_steps: int = 1

    log_every: int = field(default=20)
    save_every: int = field(default=200)
    save_total_limit: Optional[int] = field(default=1)

    eval_strategy: str = field(default="no")
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    learning_rate: float = field(default=5e-5)

    warmup_ratio_custom: float = field(default=0.03)
    logging_steps: float = field(default=0.001)
    gradient_checkpointing: bool = field(default=True)

vision_words = ["mm_projector", "embed_tokens", "lm_head", "vision_encoder",
                'seg_decoder', 'den_decoder',
                "bce_pos_weight", "dice_loss", "ssim_loss"]
trainable_words = ["lora_A", "lora_B"]

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in vision_words):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

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
        assert (labels != -100).any(), "Batch has no valid label tokens (all -100); would produce NaN loss"
        # plt.imsave('y1.png',images[0,0,:,:,64].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        return dict(images=images, dose=dose, input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                    task_type=task_types, images_gt=images_gt, images_pre=images_pre, images_pre_mask=images_pre_mask)

def print_trainable_parameters(model):
    total_numel = 0
    trainable_numel = 0
    rank0_print("=" * 20 + " Trainable parameters " + "=" * 20)
    for n, p in model.named_parameters():
        total_numel += p.numel()
        if p.requires_grad:
            trainable_numel += p.numel()
            assert any(s in n for s in trainable_words)
            rank0_print(f"[trainable] {n} {tuple(p.shape)}")
    rank0_print(f"trainable params: {trainable_numel:,}")
    rank0_print(f"total params:     {total_numel:,}")
    rank0_print(f"trainable ratio:  {100.0 * trainable_numel / total_numel:.4f}%")

def choose_task_balanced(global_step: int,
                            report_seen_steps: int, report_target_steps: int,
                            VQA_seen_steps: int, VQA_target_steps: int) -> str:
    total_target_steps = report_target_steps + VQA_target_steps

    report_done = report_seen_steps >= report_target_steps
    VQA_done = VQA_seen_steps >= VQA_target_steps

    if report_done and VQA_done: return None
    if report_done: return "VQA"
    if VQA_done: return "REPORT"

    ideal_report = (global_step + 1) * report_target_steps / total_target_steps
    ideal_VQA = (global_step + 1) * VQA_target_steps / total_target_steps

    report_deficit = ideal_report - report_seen_steps
    VQA_deficit = ideal_VQA - VQA_seen_steps

    if report_deficit >= VQA_deficit: return "REPORT"
    return "VQA"

def save_resume_metadata(ckpt_dir, global_step, report_seen_steps, VQA_seen_steps):
    state = {"global_step": int(global_step), "report_seen_steps": int(report_seen_steps), "VQA_seen_steps": int(VQA_seen_steps)}
    with open(os.path.join(ckpt_dir, "resume_meta.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_resume_metadata(ckpt_dir):
    meta_path = os.path.join(ckpt_dir, "resume_meta.json")
    if not os.path.exists(meta_path):
        return 0, 0, 0
    with open(meta_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return (int(state.get("global_step", 0)), int(state.get("report_seen_steps", 0)), int(state.get("VQA_seen_steps", 0)))

def cleanup_old_checkpoints(output_dir: str, save_total_limit: Optional[int]):
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoint_dirs = []
    for entry in os.listdir(output_dir):
        path = os.path.join(output_dir, entry)
        if not os.path.isdir(path) or not entry.startswith("checkpoint-"):
            continue
        try:
            step = int(entry.split("checkpoint-")[1])
        except (IndexError, ValueError):
            continue
        checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)
    for _, ckpt_path in checkpoint_dirs[save_total_limit:]:
        shutil.rmtree(ckpt_path)


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                              mixed_precision="bf16" if training_args.bf16 else "fp16",
                              log_with="tensorboard", project_dir=os.path.join(training_args.output_dir, "logs"),)
    local_rank = accelerator.process_index
    if accelerator.is_main_process: os.makedirs(training_args.output_dir, exist_ok=True)
    tracker_config = {k: v for k, v in asdict(model_args).items() if isinstance(v, (int, float, str, bool))}
    accelerator.init_trackers(project_name="vlm_multitask_train", config=tracker_config)

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
    bad_missing = [k for k in loading_info["missing_keys"] if not any(s in k for s in vision_words)]
    assert not bad_missing, f"Unexpected missing keys: {bad_missing}"
    assert not loading_info["unexpected_keys"], f"Unexpected keys found: {loading_info['unexpected_keys']}"
    model.config.use_cache = False

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.get_model().initialize_vision_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model_args.num_new_tokens = 2
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # Read proj_out_num before LoRA wrapping — PeftModel attribute forwarding is fragile.
    proj_out_num = model.get_model().mm_projector.proj_out_num

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=training_args.lora_r, lora_alpha=training_args.lora_alpha, target_modules=find_all_linear_names(model),
                                 lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type="CAUSAL_LM",)
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(x in n for x in trainable_words):
                p.requires_grad = True

    print_trainable_parameters(model)
    # ============================================dataset loader====================================
    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = proj_out_num
    rank0_print("vision tokens output from projector:", data_args.proj_out_num)

    report_loader = DataLoader(CapDataset(data_args, tokenizer=tokenizer, mode="train"), batch_size=training_args.train_batch_size_report,
                             shuffle=True, num_workers=training_args.dataloader_num_workers, pin_memory=True, collate_fn=DataCollator(), drop_last=True,)
    VQA_loader = DataLoader(VQADataset(data_args, tokenizer=tokenizer, mode="train"), batch_size=training_args.train_batch_size_VQA,
                             shuffle=True, num_workers=training_args.dataloader_num_workers, pin_memory=True, collate_fn=DataCollator(), drop_last=True,)

    report_target_steps = int(training_args.train_epochs_report * len(report_loader)) // accelerator.num_processes
    VQA_target_steps    = int(training_args.train_epochs_VQA    * len(VQA_loader))    // accelerator.num_processes
    total_target_steps = report_target_steps + VQA_target_steps
    rank0_print(f"total_target_steps={total_target_steps}, warmup_steps={int(total_target_steps * training_args.warmup_ratio_custom)}")
    # ============================================Optimizer====================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_args.learning_rate, weight_decay=0.01,)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_target_steps * training_args.warmup_ratio_custom),
                                                num_training_steps=total_target_steps)

    model, optimizer, scheduler, report_loader, VQA_loader = accelerator.prepare(
                                                        model, optimizer, scheduler, report_loader, VQA_loader)

    report_iter = cycle(report_loader)
    VQA_iter = cycle(VQA_loader)

    global_step, report_seen_steps, VQA_seen_steps = 0, 0, 0
    last_loss_report, last_loss_VQA = None, None

    if model_args.pre_ckpt:
        rank0_print(f"Initialized from checkpoint: {model_args.pre_ckpt}")
        accelerator.load_state(model_args.pre_ckpt)

    resumed_from_step = 0
    if model_args.resume_ckpt:
        rank0_print(f"Resuming from checkpoint: {model_args.resume_ckpt}")
        accelerator.load_state(model_args.resume_ckpt)
        global_step, report_seen_steps, VQA_seen_steps = load_resume_metadata(model_args.resume_ckpt)
        resumed_from_step = global_step
        rank0_print(f"Resumed counters: global_step={global_step}, report_seen_steps={report_seen_steps}, VQA_seen_steps={VQA_seen_steps}")
        # DeepSpeed may not restore the HF LR scheduler state — fast-forward to the correct step.
        # In non-DeepSpeed mode the scheduler is unwrapped (no .scheduler attribute); access last_epoch directly.
        _inner = getattr(scheduler, 'scheduler', scheduler)
        steps_already = _inner.last_epoch
        for _ in range(global_step - steps_already):
            scheduler.step()
        if global_step > steps_already:
            rank0_print(f"Scheduler fast-forwarded to step {global_step}, lr={scheduler.get_last_lr()[0]:.6e}")
        else:
            rank0_print(f"Scheduler state fully restored at step {global_step} (no fast-forward needed)")

    model.train()

    while report_seen_steps < report_target_steps or VQA_seen_steps < VQA_target_steps:
        task = choose_task_balanced(global_step=global_step,
                                     report_seen_steps=report_seen_steps, report_target_steps=report_target_steps,
                                     VQA_seen_steps=VQA_seen_steps, VQA_target_steps=VQA_target_steps)
        assert task is not None, "choose_task_balanced returned None inside the training loop"
        if task == "REPORT": batch = next(report_iter)
        else:                batch = next(VQA_iter)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss = model(**batch)["loss"]

            # Synchronise the finite-loss check across all ranks before branching.
            # If one rank has NaN/Inf and another doesn't, they would enter different
            # code paths: one calls backward (triggering NCCL allreduce) and the other
            # skips it, causing an NCCL collective timeout.
            is_finite_flag = torch.tensor(1 if torch.isfinite(loss) else 0,
                                          dtype=torch.int32, device=loss.device)
            if accelerator.num_processes > 1:
                dist.all_reduce(is_finite_flag, op=dist.ReduceOp.MIN)
            all_finite = is_finite_flag.item() > 0

            if not all_finite:
                rank0_print(f"[step={global_step}] WARNING: non-finite loss on at least one rank (task={task} local_loss={loss.item():.4f}), skipping update")
            else:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients: scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if task == "REPORT": report_seen_steps += 1
        else:                VQA_seen_steps    += 1
        if all_finite:
            loss_val = loss.detach().cpu().item()
            if task == "REPORT": last_loss_report = loss_val
            else:                last_loss_VQA    = loss_val

        if all_finite and global_step % training_args.log_every == 0 and global_step > 0:
            log_dict = {"train/loss": float(loss.detach().cpu().item()),
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/steps": float(global_step),
                        "train/report_pass": float(report_seen_steps / len(report_loader)),
                        "train/VQA_pass": float(VQA_seen_steps / len(VQA_loader)),}
            if last_loss_report is not None: log_dict["train/loss_report"] = last_loss_report
            if last_loss_VQA is not None: log_dict["train/loss_VQA"] = last_loss_VQA
            accelerator.log(log_dict, step=global_step)
            rank0_print(f"[step={global_step}] task={task} loss={loss.detach().cpu().item():.4f} lr={float(scheduler.get_last_lr()[0]):.6e} | "
                        f"report={report_seen_steps}/{report_target_steps} "
                        f"VQA={VQA_seen_steps}/{VQA_target_steps}")

        if global_step > 0 and global_step % training_args.save_every == 0 and global_step != resumed_from_step:
            ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
            accelerator.wait_for_everyone()
            accelerator.save_state(ckpt_dir)

            if accelerator.is_main_process:
                save_resume_metadata(ckpt_dir=ckpt_dir, global_step=global_step,
                                     report_seen_steps=report_seen_steps, VQA_seen_steps=VQA_seen_steps)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.config.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                cleanup_old_checkpoints(training_args.output_dir, training_args.save_total_limit)
            torch.cuda.empty_cache()

        global_step += 1

    accelerator.wait_for_everyone()
    final_ckpt_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    accelerator.save_state(final_ckpt_dir)

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if accelerator.is_main_process:
        save_resume_metadata(ckpt_dir=final_ckpt_dir, global_step=global_step,
                             report_seen_steps=report_seen_steps, VQA_seen_steps=VQA_seen_steps)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(training_args.output_dir)
        unwrapped_model.config.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    accelerator.end_training()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
