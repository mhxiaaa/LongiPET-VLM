import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.vision_modules import UNetModel_Encoder, UNetModel_Decoder
from monai.losses import DiceFocalLoss, SSIMLoss
import matplotlib.pyplot as plt
import numpy as np
try:
    import torch.distributed.nn
    from torch import distributed as dist
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

class VisionConfig(PretrainedConfig):
    model_type = "vision"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Vision(PreTrainedModel):
    config_class = VisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        image_size, in_channels, model_channels, num_res_blocks, channel_mult, attention_resolutions = \
            (288, 176, 128), 1, 64, 2, (1, 2, 2, 4, 6), tuple([8, 16])
        use_fp16 = False
        self.vision_encoder = UNetModel_Encoder(image_size=image_size, in_channels=in_channels,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        # CLIP
        self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path)
        self.mm_vision_proj = nn.Linear(model_channels*channel_mult[-1], self.language_encoder.config.hidden_size)
        self.mm_language_proj = nn.Linear(self.language_encoder.config.hidden_size, self.language_encoder.config.hidden_size) # 768
        self.t_prime = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        # DEN
        self.den_decoder = UNetModel_Decoder(image_size=image_size, out_channels=1,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        self.ssim_loss = SSIMLoss(spatial_dims=3)
        # SEG
        self.seg_decoder = UNetModel_Decoder(image_size=image_size, out_channels=2,  
                                                model_channels=model_channels, num_res_blocks=num_res_blocks, channel_mult=channel_mult,
                                                attention_resolutions=attention_resolutions, use_fp16=use_fp16, use_checkpoint=True)
        self.dice_loss = DiceFocalLoss(softmax=True, squared_pred=True, reduction='mean')
        self.register_buffer('bce_pos_weight', torch.tensor([0.05, 0.95]))
    
    def _load_state_dict_skip_mismatched(self, module, state_dict, strict):
        module_state = module.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in module_state and module_state[k].shape != v.shape:
                print(f"Skipping {k}: checkpoint shape {tuple(v.shape)} != model shape {tuple(module_state[k].shape)}")
                continue
            filtered[k] = v
        module.load_state_dict(filtered, strict=strict)

    def initialize_vision_module(self, model_args, strict=True):
        if model_args.pretrained_vision_modules is not None:
            tt_weight = torch.load(model_args.pretrained_vision_modules, map_location='cpu') # load_file(model_args.pretrained_vision_encoder, device="cpu")
            state_dict = tt_weight['module'] if 'module' in tt_weight else tt_weight

            self.vision_encoder.load_state_dict({k.replace("vision_encoder.", ""): v for k, v in state_dict.items() if "vision_encoder" in k}, strict=True)
            self._load_state_dict_skip_mismatched(self.den_decoder, {k.replace("den_decoder.", ""): v for k, v in state_dict.items() if "den_decoder" in k}, strict=strict)
            self._load_state_dict_skip_mismatched(self.seg_decoder, {k.replace("seg_decoder.", ""): v for k, v in state_dict.items() if "seg_decoder" in k}, strict=strict)
            print("=========================load pretrained vision modules from: ", model_args.pretrained_vision_modules)

    def encode_image(self, image, dose):
        image_feature, image_feature_list, dose_emb, projector_feature_list = self.vision_encoder(image, timesteps=dose)
        # image_feature_list[0,1,2]: [b,64,384,208,144]; [3]: [b, 64, 192, 104, 72]
        # image_feature_list[4,5]: [b,128,192,104,72]; 
        # image_feature_list[6,7,8]: [b,128,96,52,36]; [9]: [b,128,48,26,18]
        # image_feature_list[10,11]: [b,256,48,26,18]; [12]: [b,256,24,13,9]
        # image_feature_list[13,14]: [b,512,24,13,9];
        # projector_feature_list: [b,12672,256], [b,1584,384]
        projector_feats = projector_feature_list[-1] # [b,2808,384]
        projector_feats = projector_feats.mean(dim=1) # [b,384]
        projector_feats = self.mm_vision_proj(projector_feats) # [b,768]
        projector_feats = F.normalize(projector_feats, dim=-1)
        return image_feature, image_feature_list, dose_emb, projector_feats
    
    def encode_text(self, input_id, attention_mask):
        global_attention_mask = torch.zeros_like(attention_mask) # [b,4096]
        global_attention_mask[:, 0] = 1  # CLS token attends globally over all 4096 tokens
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask,
                                           global_attention_mask=global_attention_mask)["last_hidden_state"]
        text_feats = text_feats[:, 0]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats
    
    def forward_clip(self, images, dose, input_ids, attention_mask):
        _, _, _, projector_feats = self.encode_image(images, dose)
        text_features = self.encode_text(input_ids, attention_mask)

        logit_scale = self.t_prime.exp().clamp(max=100)

        if HAS_DISTRIBUTED and dist.is_available() and dist.is_initialized():
            all_image_features = torch.cat(torch.distributed.nn.all_gather(projector_feats), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)

            logits_per_image = logit_scale * (projector_feats @ all_text_features.T)
            logits_per_text = logit_scale * (text_features @ all_image_features.T)

            rank = dist.get_rank()
            local_bs = projector_feats.size(0)
            labels = torch.arange(local_bs, device=images.device) + rank * local_bs
        else:
            logits_per_image = logit_scale * (projector_feats @ text_features.T)
            logits_per_text = logit_scale * (text_features @ projector_feats.T)

            local_bs = projector_feats.size(0)
            labels = torch.arange(local_bs, device=images.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss_clip = 0.1 * (loss_i + loss_t)

        return {"loss": loss_clip, "loss_clip": loss_clip.detach(), "loss_i2t": loss_i.detach(), "loss_t2i": loss_t.detach(), 
                "logits": logits_per_image.detach(),}
    
    def forward_den(self, images, images_gt, dose, text_emb=None):
        # plt.imsave('y1.png',images[0, 0,:,128,:].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y2.png',images_gt[0, 0,:,128,:].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y3.png',images[0, 0,:,:,64].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y4.png',images_gt[0, 0,:,:,64].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
        image_feature, image_feature_list, dose_emb, _ = self.encode_image(images, dose)

        vision_rec = self.den_decoder(h=image_feature, hs=image_feature_list, emb=dose_emb, text_emb=text_emb)
        if images_gt is not None:
            loss_rec = 10 * F.l1_loss(vision_rec.float(), images_gt.float()) + self.ssim_loss(vision_rec.float(), images_gt.float())
            # plt.imsave('y1.png',vision_rec[0,0,:,128,:].detach().cpu().float().numpy(), cmap='jet', vmin=0, vmax=5)
            # plt.imsave('y2.png',images_gt[0,0,:,128,:].detach().cpu().float().numpy(), cmap='jet', vmin=0, vmax=5)
            return {"loss": loss_rec, "loss_rec": loss_rec.detach(), "recon": vision_rec.detach(),}
        else:
            return vision_rec.detach()
    
    def forward_seg(self, images, images_gt, dose, text_emb=None):
        image_feature, image_feature_list, dose_emb, _ = self.encode_image(images, dose)
        vision_seg = self.seg_decoder(h=image_feature, hs=image_feature_list, emb=dose_emb, text_emb=text_emb)
        
        if images_gt is not None:
            loss = self.dice_loss(vision_seg, torch.cat((1.0 - images_gt.float(), images_gt.float()), dim=1)) \
                + nn.CrossEntropyLoss(weight=self.bce_pos_weight)(vision_seg, images_gt[:, 0].long())
            # tt = torch.argwhere(images_gt[0, 0] == 1)
            # mid_idx = tt[len(tt) // 2][1].item() if len(tt) >= 1 else images_gt.shape[3] // 2
            # plt.imsave('y1.png', images[0,0,:,mid_idx,:].detach().float().cpu().numpy(), cmap='jet', vmin=0, vmax=5)
            # plt.imsave('y2.png', images_gt[0,0,:,mid_idx,:].detach().float().cpu().numpy(), cmap='gray', vmin=0, vmax=1.0)
            # plt.imsave('y3.png', vision_seg[0,0,:,mid_idx,:].detach().float().cpu().numpy(), cmap='gray', vmin=0, vmax=1.0)
            return {"loss": loss, "loss_seg": loss.detach(), "seg": vision_seg.detach(),}
        else:
            return vision_seg.detach()

AutoConfig.register("vision", VisionConfig)
AutoModel.register(VisionConfig, Vision)
