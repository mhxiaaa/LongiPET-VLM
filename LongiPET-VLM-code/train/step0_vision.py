import os
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_dir = '/home/project/'
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.multiscan_dataset import CLIPDataset, RefVAEDataset, RefSegDataset
from model.arch_vision import Vision, VisionConfig
## pretrain vision encoder & decoders

def rank0_print(accelerator: Accelerator, *args):
    if accelerator.is_main_process: print(*args)

def cycle(loader):
    while True:
        for batch in loader:
            yield batch

@dataclass
class ModelArguments:
    language_model_name_or_path: str = field(default="yikuan8/Clinical-Longformer") # text encoder for CLIP
    pretrained_vision_modules: str = field(default=None)
    preloaded_ckpt: Optional[str] = field(default=None) # just load model weights, no resume training
    resume_ckpt: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    REPORT_train: List[str] = field(default_factory=list)
    REPORT_val:   List[str] = field(default_factory=list)
    REPORT_test:  List[str] = field(default_factory=list)

    DEN_train: List[str] = field(default_factory=list)
    DEN_val:   List[str] = field(default_factory=list)
    DEN_test:  List[str] = field(default_factory=list)

    SEG_train: List[str] = field(default_factory=list)
    SEG_val:   List[str] = field(default_factory=list)
    SEG_test:  List[str] = field(default_factory=list)

    max_length: int = field(default=4096)
    proj_out_num: int = field(default=256)

    def __post_init__(self):
        with open(data_dir + "LongiPET-VLM-dataPath/REPORT.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.REPORT_train, self.REPORT_val, self.REPORT_test = data["train"], data["val"], data["test"]
        
        with open(data_dir + "LongiPET-VLM-dataPath/DEN.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.DEN_train, self.DEN_val, self.DEN_test = data["train"], data["val"], data["test"]
        
        minority_den = [item for item in self.DEN_train
                        if not any(k in (item[0] if isinstance(item, list) else item.get("low_dose", ""))
                                   for k in ("uExplorer", "Quadra", "Davis"))]
        self.DEN_train = self.DEN_train + minority_den  # minority appear 2×
        
        with open(data_dir + "LongiPET-VLM-dataPath/SEG.json", "r", encoding="utf-8") as f: 
            data = json.load(f)
        self.SEG_train, self.SEG_val, self.SEG_test = data["train"], data["val"], data["test"]

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    bf16: bool = field(default=True)
    output_dir: str = field(default=data_dir + "LongiPET-VLM-model_results/step0_vision")
    
    clip_target_epochs: int = field(default=100) # 100
    den_target_epochs: int = field(default=40) # modify according to how many cases for DEN and if loaded pre-weights
    seg_target_epochs: int = field(default=40) # modify according to how many cases for SEG and if loaded pre-weights

    batch_size_clip: int = 11
    batch_size_den: int = 4
    batch_size_seg: int = 4
    num_workers: int = 8
    
    log_every: int = field(default=200)
    save_every: int = field(default=100)

    eval_strategy: str = field(default="no")
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: float = field(default=0.001)
    gradient_checkpointing: bool = field(default=True)

@dataclass
class CLIPCollator:
    def __call__(self, batch):
        images = torch.stack([b["images"] for b in batch], dim=0)
        doses = torch.stack([b["dose"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        return {"images": images, "dose": doses, "input_ids": input_ids, "attention_mask": attention_mask, "task_type": "CLIP",}

@dataclass
class DENCollator:
    def __call__(self, batch):
        images = torch.stack([b["images"] for b in batch], dim=0)
        images_gt = torch.stack([b["images_gt"] for b in batch], dim=0)
        doses = torch.stack([b["dose"] for b in batch], dim=0)
        task_type = batch[0]["task_type"]
        return {"images": images, "images_gt": images_gt, "dose": doses, "task_type": task_type,}

@dataclass
class SEGCollator:
    def __call__(self, batch):
        images = torch.stack([b["images"] for b in batch], dim=0)
        images_gt = torch.stack([b["images_gt"] for b in batch], dim=0)
        doses = torch.stack([b["dose"] for b in batch], dim=0)
        task_type = batch[0]["task_type"]
        return {"images": images, "images_gt": images_gt, "dose": doses, "task_type": task_type,}

def choose_task_balanced(global_step: int, clip_seen_steps: int, clip_target_steps: int,
                                            den_seen_steps: int, den_target_steps: int,
                                            seg_seen_steps: int, seg_target_steps: int):
    total_target_steps = clip_target_steps + den_target_steps + seg_target_steps

    clip_done = clip_seen_steps >= clip_target_steps
    den_done = den_seen_steps >= den_target_steps
    seg_done = seg_seen_steps >= seg_target_steps

    if clip_done and den_done and seg_done: return None
    if clip_done and den_done: return "SEG"
    if clip_done and seg_done: return "DEN"
    if den_done and seg_done: return "CLIP"

    # ideal counts by this point in training
    ideal_clip = (global_step + 1) * clip_target_steps / total_target_steps
    ideal_den  = (global_step + 1) * den_target_steps  / total_target_steps
    ideal_seg  = (global_step + 1) * seg_target_steps  / total_target_steps

    clip_deficit = ideal_clip - clip_seen_steps if not clip_done else -float("inf")
    den_deficit  = ideal_den  - den_seen_steps  if not den_done  else -float("inf")
    seg_deficit  = ideal_seg  - seg_seen_steps  if not seg_done  else -float("inf")

    max_deficit = max(clip_deficit, den_deficit, seg_deficit)
    if clip_deficit == max_deficit: return "CLIP"
    if den_deficit  == max_deficit: return "DEN"
    return "SEG"

def save_resume_metadata(ckpt_dir, global_step, clip_seen_steps, den_seen_steps, seg_seen_steps):
    state = {"global_step": global_step, "clip_seen_steps": clip_seen_steps, "den_seen_steps": den_seen_steps, "seg_seen_steps": seg_seen_steps,}
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_resume_metadata(ckpt_dir):
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return 0, 0, 0, 0
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return (int(state.get("global_step", 0)), int(state.get("clip_seen_steps", 0)), int(state.get("den_seen_steps", 0)), int(state.get("seg_seen_steps", 0)),)

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16" if train_args.bf16 else "no",
                              log_with="tensorboard", project_dir=os.path.join(train_args.output_dir, "logs"),)
    if accelerator.is_main_process: os.makedirs(train_args.output_dir, exist_ok=True)

    tracker_config = {k: v for k, v in asdict(model_args).items() if isinstance(v, (int, float, str, bool))}
    tracker_config.update({"learning_rate": train_args.learning_rate, "weight_decay": train_args.weight_decay,
                           "warmup_ratio": train_args.warmup_ratio, "bf16": train_args.bf16,
                           "batch_size_clip": train_args.batch_size_clip, "batch_size_den": train_args.batch_size_den, "batch_size_seg": train_args.batch_size_seg,
                           "clip_target_epochs": train_args.clip_target_epochs, "den_target_epochs": train_args.den_target_epochs, "seg_target_epochs": train_args.seg_target_epochs,
                           "n_report_train": len(data_args.REPORT_train), "n_den_train": len(data_args.DEN_train), "n_seg_train": len(data_args.SEG_train),})
    accelerator.init_trackers(project_name="vision", config=tracker_config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path)
    assert tokenizer.pad_token is not None

    config = VisionConfig(language_model_name_or_path=model_args.language_model_name_or_path,
                          pretrained_vision_modules=model_args.pretrained_vision_modules)
    model = Vision(config)
    model.initialize_vision_module(model_args, strict=False)
    if train_args.gradient_checkpointing:
        model.language_encoder.gradient_checkpointing_enable()

    clip_loader = DataLoader(CLIPDataset(data_args, tokenizer=tokenizer, mode="train"), 
                             batch_size=train_args.batch_size_clip, shuffle=True, num_workers=train_args.num_workers,
                             pin_memory=True, collate_fn=CLIPCollator(), drop_last=True,)
    den_loader = DataLoader(RefVAEDataset(data_args, tokenizer=None, mode="train"), 
                            batch_size=train_args.batch_size_den, shuffle=True, num_workers=train_args.num_workers,
                            pin_memory=True, collate_fn=DENCollator(), drop_last=True,)
    seg_loader = DataLoader(RefSegDataset(data_args, tokenizer=None, mode="train"),
                            batch_size=train_args.batch_size_seg, shuffle=True, num_workers=train_args.num_workers,
                            pin_memory=True, collate_fn=SEGCollator(), drop_last=True,)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay,)

    model, optimizer, clip_loader, den_loader, seg_loader = accelerator.prepare(model, optimizer, clip_loader, den_loader, seg_loader)

    clip_target_steps = train_args.clip_target_epochs * len(clip_loader)
    den_target_steps  = train_args.den_target_epochs  * len(den_loader)
    seg_target_steps  = train_args.seg_target_epochs  * len(seg_loader)
    total_target_steps = clip_target_steps + den_target_steps + seg_target_steps

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_target_steps * train_args.warmup_ratio),
                                                num_training_steps=total_target_steps,)
    scheduler = accelerator.prepare(scheduler)

    rank0_print(accelerator, "REPORT train/val/test:", len(data_args.REPORT_train), len(data_args.REPORT_val), len(data_args.REPORT_test),)
    rank0_print(accelerator, "DEN train/val/test:", len(data_args.DEN_train), len(data_args.DEN_val), len(data_args.DEN_test),)
    rank0_print(accelerator, "SEG train/val/test:", len(data_args.SEG_train), len(data_args.SEG_val), len(data_args.SEG_test),)
    rank0_print(accelerator, f"len(clip_loader) = {len(clip_loader)}")
    rank0_print(accelerator, f"len(den_loader)  = {len(den_loader)}")
    rank0_print(accelerator, f"len(seg_loader)  = {len(seg_loader)}")
    rank0_print(accelerator, f"clip_target_steps = {clip_target_steps} ({train_args.clip_target_epochs} full passes)",)
    rank0_print(accelerator, f"den_target_steps  = {den_target_steps} ({train_args.den_target_epochs} full passes)",)
    rank0_print(accelerator, f"seg_target_steps  = {seg_target_steps} ({train_args.seg_target_epochs} full passes)",)
    rank0_print(accelerator, f"total_target_steps = {total_target_steps}")

    clip_iter = cycle(clip_loader)
    den_iter  = cycle(den_loader)
    seg_iter  = cycle(seg_loader)

    global_step = 0
    clip_seen_steps = 0
    den_seen_steps  = 0
    seg_seen_steps  = 0

    last_loss_clip = None
    last_loss_rec  = None
    last_loss_seg  = None
    last_loss_i2t  = None
    last_loss_t2i  = None

    if model_args.preloaded_ckpt is not None: # just load model weights
        rank0_print(accelerator, f"Initialized from checkpoint: {model_args.preloaded_ckpt}")
        accelerator.load_state(model_args.preloaded_ckpt)

    if model_args.resume_ckpt is not None:
        rank0_print(accelerator, f"Resuming from checkpoint: {model_args.resume_ckpt}")
        accelerator.load_state(model_args.resume_ckpt)
        global_step, clip_seen_steps, den_seen_steps, seg_seen_steps = load_resume_metadata(model_args.resume_ckpt)
        rank0_print(accelerator, f"Resumed counters: global_step={global_step}, "
                    f"clip_seen_steps={clip_seen_steps}, den_seen_steps={den_seen_steps}, seg_seen_steps={seg_seen_steps}",)
        # DeepSpeed may not restore the HF LR scheduler state — fast-forward to the correct step.
        # scheduler.step() is pure arithmetic so stepping thousands of times is negligible.
        steps_already = scheduler.last_epoch  # 0 or 1 if not restored; global_step if restored
        for _ in range(global_step - steps_already):
            scheduler.step()
        rank0_print(accelerator, f"Scheduler fast-forwarded to step {global_step}, lr={scheduler.get_last_lr()[0]:.6e}")

    model.train()

    while clip_seen_steps < clip_target_steps or den_seen_steps < den_target_steps or seg_seen_steps < seg_target_steps:
        task = choose_task_balanced(global_step=global_step,
                                    clip_seen_steps=clip_seen_steps, clip_target_steps=clip_target_steps,
                                    den_seen_steps=den_seen_steps,   den_target_steps=den_target_steps,
                                    seg_seen_steps=seg_seen_steps,   seg_target_steps=seg_target_steps,)

        assert task is not None, "choose_task_balanced returned None inside the training loop"
        if   task == "CLIP": batch = next(clip_iter)
        elif task == "DEN":  batch = next(den_iter)
        else:                batch = next(seg_iter)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                if task == "CLIP":
                    outputs = model.forward_clip(images=batch["images"], dose=batch["dose"],
                                                 input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],)
                elif task == "DEN":
                    outputs = model.forward_den(images=batch["images"], images_gt=batch["images_gt"], dose=batch["dose"],)
                else:
                    outputs = model.forward_seg(images=batch["images"], images_gt=batch["images_gt"], dose=batch["dose"],)

            loss = outputs["loss"]
            accelerator.backward(loss)
            optimizer.step()
            if accelerator.sync_gradients: scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if task == "CLIP":
            clip_seen_steps += 1
            if "loss_clip" in outputs: last_loss_clip = float(outputs["loss_clip"].item())
            if "loss_i2t"  in outputs: last_loss_i2t  = float(outputs["loss_i2t"].item())
            if "loss_t2i"  in outputs: last_loss_t2i  = float(outputs["loss_t2i"].item())
        elif task == "DEN":
            den_seen_steps += 1
            if "loss_rec" in outputs: last_loss_rec = float(outputs["loss_rec"].item())
        else:
            seg_seen_steps += 1
            if "loss_seg" in outputs: last_loss_seg = float(outputs["loss_seg"].item())

        if global_step % train_args.log_every == 0:
            lr = float(scheduler.get_last_lr()[0])

            log_dict = {"train/loss": float(loss.item()), "train/lr": lr,
                        "train/clip_seen_steps": float(clip_seen_steps), "train/den_seen_steps": float(den_seen_steps), "train/seg_seen_steps": float(seg_seen_steps),
                        "train/clip_pass": float(clip_seen_steps / len(clip_loader)), "train/den_pass": float(den_seen_steps / len(den_loader)), "train/seg_pass": float(seg_seen_steps / len(seg_loader)),
                        "train/task_clip": 1.0 if task == "CLIP" else 0.0, "train/task_den": 1.0 if task == "DEN" else 0.0, "train/task_seg": 1.0 if task == "SEG" else 0.0,}
            if last_loss_clip is not None: log_dict["train/loss_clip"] = last_loss_clip
            if last_loss_rec  is not None: log_dict["train/loss_rec"]  = last_loss_rec
            if last_loss_seg  is not None: log_dict["train/loss_seg"]  = last_loss_seg
            if last_loss_i2t  is not None: log_dict["train/loss_i2t"]  = last_loss_i2t
            if last_loss_t2i  is not None: log_dict["train/loss_t2i"]  = last_loss_t2i

            accelerator.log(log_dict, step=global_step)
            rank0_print(accelerator, f"[step={global_step}] task={task} loss={loss.item():.4f} lr={lr:.6e}")

        if global_step > 0 and global_step % train_args.save_every == 0:
            ckpt_dir = os.path.join(train_args.output_dir, f"checkpoint-{global_step}")
            accelerator.wait_for_everyone()
            accelerator.save_state(ckpt_dir)

            if accelerator.is_main_process:
                save_resume_metadata(ckpt_dir=ckpt_dir, global_step=global_step, clip_seen_steps=clip_seen_steps, den_seen_steps=den_seen_steps, seg_seen_steps=seg_seen_steps,)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.config.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        global_step += 1

    accelerator.wait_for_everyone()
    accelerator.save_state(train_args.output_dir)

    if accelerator.is_main_process:
        save_resume_metadata(ckpt_dir=train_args.output_dir, global_step=global_step, clip_seen_steps=clip_seen_steps, den_seen_steps=den_seen_steps, seg_seen_steps=seg_seen_steps,)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(train_args.output_dir)
        tokenizer.save_pretrained(train_args.output_dir)

    rank0_print(accelerator, "Training finished.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
