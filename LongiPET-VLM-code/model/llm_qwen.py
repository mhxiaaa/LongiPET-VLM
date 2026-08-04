from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3Model,)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from .vlm_arch import VLMMetaForCausalLM, VLMMetaModel
import torch.nn.functional as F
from monai.losses import DiceFocalLoss, SSIMLoss, HausdorffDTLoss
from transformers import Trainer
import matplotlib.pyplot as plt

class MLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if isinstance(outputs, tuple): loss = outputs[0]
        elif isinstance(outputs, dict): loss = outputs["loss"]
        else: raise TypeError(f"Unexpected model output type: {type(outputs)}")
        return (loss, outputs) if return_outputs else loss

def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    pred_dx, pred_dy, pred_dz = pred[:, :, 1:, :, :]-pred[:, :, :-1, :, :], pred[:, :, :, 1:, :]-pred[:, :, :, :-1, :], pred[:, :, :, :, 1:]-pred[:, :, :, :, :-1]
    target_dx, target_dy, target_dz = target[:, :, 1:, :, :]-target[:, :, :-1, :, :], target[:, :, :, 1:, :]-target[:, :, :, :-1, :], target[:, :, :, :, 1:]-target[:, :, :, :, :-1]
    loss = (F.l1_loss(pred_dx, target_dx, reduction=reduction) + F.l1_loss(pred_dy, target_dy, reduction=reduction) + F.l1_loss(pred_dz, target_dz, reduction=reduction))
    return loss / 3.0

class VLMQwenConfig(Qwen3Config):
    model_type = "vlm_qwen"
    loss_type = "ForCausalLMLoss"

class VLMQwenModel(VLMMetaModel, Qwen3Model):
    config_class = VLMQwenConfig
    def __init__(self, config: Qwen3Config):
        super(VLMQwenModel, self).__init__(config)


class VLMQwenForCausalLM(Qwen3ForCausalLM, VLMMetaForCausalLM):
    config_class = VLMQwenConfig
    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = VLMQwenModel(config)
        self.pretraining_tp = getattr(config, "pretraining_tp", None)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.dice_loss = DiceFocalLoss(softmax=True, squared_pred=True, reduction='mean')
        self.register_buffer('bce_pos_weight', torch.tensor([0.05, 0.95]))
        self.ssim_loss = SSIMLoss(spatial_dims=3, reduction="mean")

    def get_model(self):
        return self.model

    @staticmethod
    def _mean_pool_hidden(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        weights = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.LongTensor] = None,
                images: Optional[torch.FloatTensor] = None, dose: Optional[torch.FloatTensor] = None,
                images_gt: Optional[torch.FloatTensor] = None, task_type=None,
                images_pre: Optional[torch.FloatTensor] = None, images_pre_mask: Optional[torch.BoolTensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs: Any,) -> Union[Tuple, CausalLMOutputWithPast]:

        if task_type is None:
            task_type = getattr(self, "_task_type_cache", None)
        assert task_type is not None, f"task_type must be provided"
        if isinstance(task_type, str): task_type = [task_type]

        image_feature_list = None
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_feature, image_feature_list, dose_emb) = self.prepare_inputs_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, images_pre, dose, task_type=task_type, images_pre_mask=images_pre_mask)
        # labels.shape = attention_mask.shape=[B,512]; inputs_embeds.shape = [B,512,3584]

        if all((t == "report") or t.startswith("VQA") for t in task_type): # 'report' and 'VQA' =======================================================
            outputs = super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                      labels=labels, output_hidden_states=None,
                                      position_ids=position_ids, past_key_values=past_key_values,
                                      use_cache=use_cache, output_attentions=output_attentions, return_dict=return_dict,)
            if labels is None:
                return outputs  # CausalLMOutputWithPast — generate() needs .logits attribute
            return {"loss": outputs.loss}
        
        elif all(t == "SEG" for t in task_type): # 'SEG' task =================================================================================================
            outputs = self.get_model().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                            position_ids=position_ids, past_key_values=past_key_values,
                                            use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True,)
            last_hidden_state = outputs.last_hidden_state # last_hidden_state.shape=[B,?,4096]
            if not torch.isfinite(last_hidden_state).all():
                print(f"[SEG] NaN/Inf in LLM last_hidden_state: nan={last_hidden_state.isnan().sum()} inf={last_hidden_state.isinf().sum()}")
                return {"loss": torch.tensor(float("nan"), device=last_hidden_state.device, requires_grad=True)}
            prompt = self._mean_pool_hidden(last_hidden_state, attention_mask) # [B,4096]
            logits = self.get_model().seg_decoder(h=image_feature, hs=image_feature_list, emb=dose_emb, text_emb=prompt)
            if not torch.isfinite(logits).all():
                print(f"[SEG] NaN/Inf in decoder logits: nan={logits.isnan().sum()} inf={logits.isinf().sum()}")
                return {"loss": torch.tensor(float("nan"), device=logits.device, requires_grad=True)}
            if images_gt is not None:
                gt_onehot = torch.cat((1.0 - images_gt.float(), images_gt.float()), dim=1)
                loss_dice = self.dice_loss(logits, gt_onehot)
                loss_ce   = nn.CrossEntropyLoss(weight=self.bce_pos_weight)(logits, images_gt[:, 0].long())
                if not torch.isfinite(loss_dice) or not torch.isfinite(loss_ce):
                    print(f"[SEG] loss components: dice={loss_dice.item():.4f} ce={loss_ce.item():.4f}")
                loss_seg = loss_dice + loss_ce
                return {"loss": loss_seg}
            else:
                return logits
            
        elif all(t == "DEN" for t in task_type): # 'DEN' task =================================================================================================
            assert not torch.isnan(inputs_embeds).any(), "NaN in inputs_embeds (before LLM)"
            assert not torch.isinf(inputs_embeds).any(), "Inf in inputs_embeds (before LLM)"
            outputs = self.get_model().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                            position_ids=position_ids, past_key_values=past_key_values,
                                            use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True,)
            last_hidden_state = outputs.last_hidden_state # last_hidden_state.shape=[B,L,4096]
            if not torch.isfinite(last_hidden_state).all():
                print(f"[DEN] NaN/Inf in LLM last_hidden_state: nan={last_hidden_state.isnan().sum()} inf={last_hidden_state.isinf().sum()}")
                return {"loss": torch.tensor(float("nan"), device=inputs_embeds.device, requires_grad=True)}
            prompt = self._mean_pool_hidden(last_hidden_state, attention_mask) # prompt.shape = [B,4096]
            logits = self.get_model().den_decoder(h=image_feature, hs=image_feature_list, emb=dose_emb, text_emb=prompt)
            if not torch.isfinite(logits).all():
                print(f"[DEN] NaN/Inf in decoder logits: nan={logits.isnan().sum()} inf={logits.isinf().sum()}")
                return {"loss": torch.tensor(float("nan"), device=inputs_embeds.device, requires_grad=True)}
            if images_gt is not None:
                loss_l1   = 10.0 * F.l1_loss(logits, images_gt.float(), reduction="mean")
                loss_grad = gradient_loss_3d(logits, images_gt.float())
                loss_ssim = 0.1 * self.ssim_loss(logits.float(), images_gt.float())
                if not torch.isfinite(loss_l1) or not torch.isfinite(loss_grad) or not torch.isfinite(loss_ssim):
                    print(f"[DEN] loss components: l1={loss_l1.item():.4f} grad={loss_grad.item():.4f} ssim={loss_ssim.item():.4f}")
                loss_den = loss_l1 + loss_grad + loss_ssim
                # plt.imsave('y1.png',images_gt[0,0,:,:,64].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=0.3)
                # plt.imsave('y2.png',logits[0,0,:,:,64].float().detach().cpu().numpy(), cmap='jet', vmin=0, vmax=0.3)
                return {"loss": loss_den}
            else:
                return logits
        
        else: raise ValueError(f"Unsupported task_type batch: {task_type}")

    @torch.no_grad()
    def generate(self, images: Optional[torch.Tensor] = None, 
                    images_pre: Optional[torch.Tensor] = None, images_pre_mask = None,
                    input_ids: Optional[torch.Tensor] = None, dose: Optional[torch.FloatTensor] = None,
                 task_type=None, **kwargs,) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs: raise NotImplementedError("`inputs_embeds` is not supported")

        if isinstance(task_type, (list, tuple)):
            if not task_type:
                raise ValueError("task_type cannot be empty.")
            canonical_task = task_type[0]
            if any(t != canonical_task for t in task_type):
                raise ValueError(f"Mixed task_type in generate is not supported: {task_type}")
        else:
            canonical_task = task_type

        if canonical_task == "report" or (isinstance(canonical_task, str) and canonical_task.startswith("VQA")):
            (input_ids, position_ids, attention_mask, _, inputs_embeds, _, _, _, _) = self.prepare_inputs_for_multimodal(
                            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, 
                            past_key_values=None, labels=None, images=images, images_pre=images_pre, dose=dose, 
                            task_type=[canonical_task], images_pre_mask=images_pre_mask)

            self._task_type_cache = canonical_task
            output_ids = super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
            self._task_type_cache = None
            return output_ids
        
        elif canonical_task in {"DEN", "SEG"}:
            batch_size = images.shape[0] if images is not None else input_ids.shape[0]
            logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=None,
                                  images=images, images_pre=images_pre, images_pre_mask=images_pre_mask,
                                  dose=dose, images_gt=None,
                                  task_type=[canonical_task] * batch_size,
                                  return_dict=True,)
            return logits

        raise ValueError(f"Unsupported task_type for generate: {task_type}")


AutoConfig.register("vlm_qwen", VLMQwenConfig)
AutoModelForCausalLM.register(VLMQwenConfig, VLMQwenForCausalLM)
