from abc import ABC, abstractmethod
import logging
import torch
import numpy as np
from .projector_mlp import MixerLowHighHybridMLP
from .vision_modules import UNetModel_Encoder, UNetModel_Decoder
import torch.nn as nn

def _unwrap_state_dict(state):
    if "module" in state and isinstance(state["module"], dict): return state["module"]
    if "state_dict" in state and isinstance(state["state_dict"], dict): return state["state_dict"]
    return state

def _load_compatible(module, state_dict):
    """Load state_dict into module, skipping keys with shape mismatches."""
    current = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k not in current or current[k].shape == v.shape}
    skipped = [k for k in state_dict if k in current and current[k].shape != state_dict[k].shape]
    if skipped:
        print(f"  skipping {len(skipped)} key(s) with shape mismatch: {skipped}")
    module.load_state_dict(compatible, strict=False)

class VLMMetaModel:
    def __init__(self, config):
        super(VLMMetaModel, self).__init__(config)
        self.config = config
        image_size, in_channels, model_channels, num_res_blocks, channel_mult, attention_resolutions = \
            (288, 176, 128), 1, 64, 2, (1, 2, 2, 4, 6), tuple([8, 16])
        use_fp16 = False
        self.vision_encoder = UNetModel_Encoder(image_size=image_size, in_channels=in_channels,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        self.mm_projector = MixerLowHighHybridMLP(low_input_size=[12672, 256], low_output_size=[512, 256],
                                                  high_input_size=[1584, 384], high_output_size=[256, 256],
                                                  output_dim=4096, depth=2, mlp_depth=2, proj_out_num=512)

        self.den_decoder = UNetModel_Decoder(image_size=image_size, out_channels=1,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        
        self.seg_decoder = UNetModel_Decoder(image_size=image_size, out_channels=2,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        

    def initialize_vision_modules(self, model_args):
        if model_args.pretrain_vision_modules is not None:
            tt_weight = _unwrap_state_dict(torch.load(model_args.pretrain_vision_modules, map_location="cpu", weights_only=False))
            state_dict = tt_weight['module'] if 'module' in tt_weight else tt_weight

            self.vision_encoder.load_state_dict({k.replace("vision_encoder.", ""): v for k, v in state_dict.items() if "vision_encoder" in k}, strict=True)
            _load_compatible(self.den_decoder, {k.replace("den_decoder.", ""): v for k, v in state_dict.items() if "den_decoder" in k})
            _load_compatible(self.seg_decoder, {k.replace("seg_decoder.", ""): v for k, v in state_dict.items() if "seg_decoder" in k})
            print("load pretrained vision modules from:", model_args.pretrain_vision_modules)

class VLMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, images_pre, dose, 
                                      task_type=None, images_pre_mask=None):
        if input_ids is None:
            return (input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None)
        else:
            if images is not None:
                image_feature, image_feature_list, dose_emb, projector_feature_list = self.get_model().vision_encoder(images, timesteps=dose)
                # projector_feature_list: [[b,12672,256]; [b,1584,384]]
                assert not any(torch.isnan(f).any() or torch.isinf(f).any() for f in projector_feature_list), \
                    f"NaN/Inf from vision_encoder: " + ", ".join(f"f{i} nan={f.isnan().sum()} inf={f.isinf().sum()} max={f.abs()[f.isfinite()].max():.3g}" for i, f in enumerate(projector_feature_list))
                
                image_projector_feature = self.get_model().mm_projector(projector_feature_list) # [B,512,4096]
                if torch.isnan(image_projector_feature).any() or torch.isinf(image_projector_feature).any():
                    print(f"[WARNING] NaN/Inf in image_projector_feature: nan={image_projector_feature.isnan().sum()} inf={image_projector_feature.isinf().sum()} — returning NaN inputs_embeds to skip step")
                    nan_embeds = self.get_model().embed_tokens(input_ids) * float('nan')
                    return (None, position_ids, attention_mask, past_key_values, nan_embeds, labels, None, None, None)

            else:
                image_feature, image_feature_list, dose_emb = None, None, None

            inputs_embeds = self.get_model().embed_tokens(input_ids).clone() # [B,T,4096]

            if images is not None:
                inputs_embeds[input_ids == self.get_model().config.img_token_id] = image_projector_feature.reshape(-1, image_projector_feature.shape[-1]).to(inputs_embeds.dtype)

            if images_pre is not None:
                # images_pre_mask: [B] bool — True for samples that have a prior scan.
                valid_idx = images_pre_mask.nonzero(as_tuple=True)[0]  # [N_valid]

                if len(valid_idx) > 0:
                    _, _, _, projector_feature_list_pre = self.get_model().vision_encoder(images_pre[valid_idx],
                                                    timesteps=torch.tensor(100, dtype=torch.float32, device=images.device).expand(len(valid_idx)))
                    assert not any(torch.isnan(f).any() or torch.isinf(f).any() for f in projector_feature_list_pre), "NaN/Inf from vision_encoder (pre)"
                    
                    image_features_pre = self.get_model().mm_projector(projector_feature_list_pre)  # [N_valid,256,4096]
                    assert not torch.isnan(image_features_pre).any()
                    
                    for k, b in enumerate(valid_idx):
                        mask_b = (input_ids[b] == self.get_model().config.imgPre_token_id)  # [T]
                        inputs_embeds[b][mask_b] = image_features_pre[k].to(inputs_embeds.dtype)  # [256,4096]

            return (None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_feature, image_feature_list, dose_emb)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        self.get_model().config.img_token_id = model_args.img_token_id
        self.get_model().config.imgPre_token_id = model_args.imgPre_token_id

        input_embeddings = self.get_input_embeddings().weight.data
        output_embeddings = self.get_output_embeddings().weight.data

        if num_new_tokens > 0:
            print("resized token embeddings to length", len(tokenizer))

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:].copy_(input_embeddings_avg)
            output_embeddings[-num_new_tokens:].copy_(output_embeddings_avg)

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu", weights_only=False)
            if "module" in mm_projector_weights and isinstance(mm_projector_weights["module"], dict):
                mm_projector_weights = mm_projector_weights["module"]
            elif "state_dict" in mm_projector_weights and isinstance(mm_projector_weights["state_dict"], dict):
                mm_projector_weights = mm_projector_weights["state_dict"]

            mm_projector_state = {k.replace("model.mm_projector.", ""): v for k, v in mm_projector_weights.items()
                                  if k.startswith("model.mm_projector.")}
            if not mm_projector_state: raise ValueError("No mm_projector weights found in checkpoint.")

            self.get_model().mm_projector.load_state_dict(mm_projector_state, strict=True)
            print("load pretrained mm_mlp_adapter from:", model_args.pretrain_mm_mlp_adapter)

            if "model.embed_tokens.weight" in mm_projector_weights:
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]

                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings.copy_(embed_tokens_weight)
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:].copy_(embed_tokens_weight)
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. " f"Pretrained: {embed_tokens_weight.shape}. "
                                     f"Current: {input_embeddings.shape}. " f"Number of new tokens: {num_new_tokens}.")
            else:
                print("Warning: model.embed_tokens.weight not found in pretrained mm_mlp_adapter.")
