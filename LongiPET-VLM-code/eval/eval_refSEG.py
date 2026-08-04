import os
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
save_dir = '/home/dataset/'
dst_dir = '/home/project/'
import json
import random
import numpy as np
import argparse
import torch
from skimage.transform import resize
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.multiscan_preprocess import get_text_from_path
from dataset.multiscan_dataset import get_date_from_path
from dataset.assistFun import get_patchList_Test, reverseSeg, crop_image_zeroOut3D, uncrop_image_zeroOut3D, crop_like_with_meta
from model.llm_qwen import VLMQwenForCausalLM, VLMQwenConfig  # registers vlm_qwen with AutoModel
import SimpleITK as sitk
from pathlib import Path

src_dir = 'user/dataset/'

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='/home/project/LongiPET-VLM-model_results/model')
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--proj_out_num", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="/home/project/LongiPET-VLM-model_results/eval_refSEG/")
    # the vision encoder/decoder are trained on a fixed (288,176,128) patch (see multiscan_preprocess.target_shape); not adjustable here
    parser.add_argument('--input_size', default=[288, 176, 128])
    parser.add_argument('--stride_size', default=[36, 22, 16])
    parser.add_argument('--patchBatchSize', default=1)
    return parser.parse_args(args)

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    with open("/home/project/LongiPET-VLM-dataPath/SEG.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    _, _, args.seg_data_test_path = data["train"], data["val"], data["test"]
    print('num of pairs for testing seg (from SEG.json):', len(args.seg_data_test_path))

    # tokenizer=======================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=args.max_length,
                                              padding_side="right", use_fast=True,)
    assert tokenizer.convert_tokens_to_ids("<im_patch>") == 151669 and tokenizer.convert_tokens_to_ids("<imPre_patch>") == 151670
    assert len(tokenizer) == 151671

    # model============================
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for item in args.seg_data_test_path:
        image_path = item["image"].replace(src_dir, dst_dir)
        dose = item["dose_tag"]
        pre_pet_path = item.get("previous_high_doses") or []
        savePath = image_path.replace(dst_dir, args.output_dir)
        assert image_path != savePath

        if os.path.exists(savePath):
            continue

        imageOri = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(imageOri)

        text_path = os.path.join(os.path.dirname(image_path), 'report.txt')
        priorInfo, _ = get_text_from_path(text_path)
        imagePre = None
        if pre_pet_path:
            pre_pet_path0 = pre_pet_path[0].replace(src_dir, dst_dir)
            pre_text_path = os.path.join(os.path.dirname(pre_pet_path0), 'report.txt')
            if os.path.exists(pre_text_path):
                _, pre_diagnoInfo = get_text_from_path(pre_text_path)
                date_cur, date_pre = get_date_from_path(text_path), get_date_from_path(pre_text_path)
                timeDiff = (date_cur.year - date_pre.year) * 12 + (date_cur.month - date_pre.month)
                priorInfo = priorInfo + f' Report from {timeDiff} months ago: ' + pre_diagnoInfo + ' '

            # align the previous-dose scan onto the current scan's raw grid (mirrors multiscan_preprocess.get_img_from_path lines 162-167)
            imagePre = sitk.GetArrayFromImage(sitk.ReadImage(pre_pet_path0))
            imagePre = imagePre[-image.shape[0]::, :, :]
            imagePre = resize(imagePre, image.shape, anti_aliasing=False)

        question = "<im_patch>" * args.proj_out_num + ' Task: SEG. ' + priorInfo
        if imagePre is not None:
            question = question + "<imPre_patch>" * args.proj_out_num
        question_tensor = tokenizer(question, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt",)
        input_id = question_tensor["input_ids"].to(device=device)
        attention_mask = question_tensor["attention_mask"].to(device=device)

        # 1. tight-crop the background out, keeping reconstruction meta (mirrors multiscan_preprocess.get_img_from_path's threshold rule)
        suvThreshold = 0.1 if np.max(image) < 5.0 else 0.2
        image, meta_crop = crop_image_zeroOut3D(image, tol=suvThreshold, min_size=(1, 1, 1), pad_value=0.0)
        if imagePre is not None: imagePre = crop_like_with_meta(imagePre, meta_crop)

        # 2. move the smallest axis to last, matching dataset.multiscan_preprocess.move_axis_with_fewest_slices_to_last
        moved_axis = int(np.argmin(image.shape))
        perm = [ax for ax in range(3) if ax != moved_axis] + [moved_axis]
        reverse_perm = list(np.argsort(perm))
        image = np.transpose(image, axes=perm)
        if imagePre is not None: imagePre = np.transpose(imagePre, axes=perm)

        # 3. pad up to the model's fixed patch size (tol=-1 keeps this crop a no-op, padding only)
        image, meta_pad = crop_image_zeroOut3D(image, tol=-1.0, min_size=args.input_size, pad_value=0.0)
        if imagePre is not None: imagePre = crop_like_with_meta(imagePre, meta_pad)

        patchIdxAll = get_patchList_Test(img=image, patch_size=args.input_size, patch_moving_stride=args.stride_size, validValueThresh=suvThreshold)
        patchAll_result = []

        for ppIdx in range(0, len(patchIdxAll), args.patchBatchSize):
            if ppIdx <= (len(patchIdxAll)-args.patchBatchSize): BatchSizeCurrent = args.patchBatchSize
            else: BatchSizeCurrent = len(patchIdxAll) - (len(patchIdxAll)//args.patchBatchSize) * args.patchBatchSize

            patchesCurrent, patchesPreCurrent = [], []
            for bbIdx in range(BatchSizeCurrent):
                boxPP = patchIdxAll[ppIdx+bbIdx]
                patchesCurrent.append(image[boxPP[0]:boxPP[1], boxPP[2]:boxPP[3], boxPP[4]:boxPP[5]])
                if imagePre is not None:
                    patchesPreCurrent.append(imagePre[boxPP[0]:boxPP[1], boxPP[2]:boxPP[3], boxPP[4]:boxPP[5]])

            # raw SUV values, unnormalized — step3 training feeds images straight from get_img_from_path with no clip/scale
            lowS = torch.from_numpy(np.array(patchesCurrent)).to(device=device).float().unsqueeze(dim=1)
            if imagePre is not None:
                preS = torch.from_numpy(np.array(patchesPreCurrent)).to(device=device).float().unsqueeze(dim=1)
                preS_mask = torch.ones(lowS.shape[0], dtype=torch.bool, device=device)
            else:
                preS, preS_mask = None, None
            with torch.inference_mode():
                predS = model.generate(images=lowS, images_pre=preS, images_pre_mask=preS_mask,
                                       input_ids=input_id.repeat(lowS.shape[0], 1), attention_mask=attention_mask.repeat(lowS.shape[0], 1),
                                       dose=torch.tensor(dose, dtype=torch.float32, device=lowS.device).repeat(lowS.shape[0]),
                                       task_type='SEG',)
            # predS: [B,2,H,W,D] raw seg_decoder logits (channel 0=background, channel 1=tumor, see llm_qwen.py gt_onehot construction)
            predS = torch.softmax(predS, dim=1).detach().to(torch.float32).cpu().numpy()
            for bbIdx in range(BatchSizeCurrent):
                patchAll_result.append(predS[bbIdx])

        predIm_result = reverseSeg(img_size=image.shape, predictions=patchAll_result, patchIdx=patchIdxAll)  # [2,*image.shape] blended class probabilities
        predSeg = np.argmax(predIm_result, axis=0).astype(np.float32)  # [*image.shape] binary tumor mask

        predFinal = uncrop_image_zeroOut3D(predSeg, meta_pad)
        predFinal = np.transpose(predFinal, axes=reverse_perm)
        predFinal = uncrop_image_zeroOut3D(predFinal, meta_crop)

        predSave = sitk.GetImageFromArray(predFinal)
        predSave.CopyInformation(imageOri)
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        sitk.WriteImage(predSave, savePath)
        print('saving: ', savePath)

if __name__ == "__main__":
    main()
