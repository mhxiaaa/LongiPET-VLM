import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ["HF_HUB_CACHE"] = "/data28/user/mx79/huggingface/hub"
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.assistFun import get_patchList_Test, reverseImg, crop_image_zeroOut3D, uncrop_image_zeroOut3D
from model.arch_vision import Vision, VisionConfig
import SimpleITK as sitk
from pathlib import Path
import numpy as np
# /home2/mx79/Anaconda3/bin/python /home2/mx79/code_PET-VLM_6/eval/eval0_vision_den.py
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
    parser.add_argument('--language_model_name_or_path', type=str, default="yikuan8/Clinical-Longformer")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--output_dir', type=str, default="/data22/user/mx79/results_code-PET-VLM_6/eval0_vision_den/")
    parser.add_argument('--patchBatchSize', default=1)
    parser.add_argument('--pretrain_module', type=str, default="/data22/user/mx79/results_code-PET-VLM_6/step0_vision_run1.pt")
    return parser.parse_args(args)
          
def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    args.vae_data_test_path = []
    # ===================00.Downstream_mCT_FDG_from_dynamicData====================
    for file in Path('/data17/user/mx79/03.Downstream_mCT_FDG_from_dynamicData/').rglob('*.nii.gz'):
        if 'Seg.nii.gz' not in str(file) and '/CT_' not in str(file) and 'HC.nii.gz' not in str(file):
            low_path = str(file)
            high_path = low_path.replace(low_path.split('/')[-1], 'HC.nii.gz')
            dose = int(low_path.split('/')[-1].replace('.nii.gz', '').replace('_', ''))
            args.vae_data_test_path.append({"low_dose": low_path, "high_dose": high_path, "dose_tag": dose})

    with open("/data22/user/mx79/0.___allTaskSort_file_path/DEN.json", "r", encoding="utf-8") as f: 
        data = json.load(f)
    path_DENtask_train, path_DENtask_val, path_DENtask_test = data["train"], data["val"], data["test"]

    args.vae_data_test_path.extend(file for file in path_DENtask_test)
    print('num of pairs for testing den:', len(args.vae_data_test_path))

    config = VisionConfig(language_model_name_or_path=args.language_model_name_or_path, pretrained_vision_modules=None)
    model = Vision(config)
    if args.pretrain_module is not None:
        tt_weight = torch.load(args.pretrain_module, map_location='cpu')
        state_dict = tt_weight['module'] if 'module' in tt_weight else tt_weight
        model.load_state_dict(state_dict, strict=False)
        print("model load pretrained weights from: ", args.pretrain_module)
    model = model.to(device=device)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _test_indices = list(range(1, 3)) + list(range(99, 101)) + list(range(267, 269))
    # for tt in [args.vae_data_test_path[i] for i in _test_indices]: # 1-2; 99-100; 267-268
    for tt in args.vae_data_test_path:
        image_path, gt_path, dose = tt["low_dose"], tt["high_dose"], tt["dose_tag"]
        savePath = image_path.replace('/data17/user/mx79/', args.output_dir)
        if not os.path.exists(savePath):
            imageOri = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(imageOri)
            # 1. make sure in h-w-w format ===================================
            if image.shape[0] == image.shape[1]: image = np.transpose(image, axes=(2,0,1)); reverse1 = (1,2,0)
            elif image.shape[0] == image.shape[2]: image = np.transpose(image, axes=(1,0,2)); reverse1 = (1,0,2)
            else: reverse1 = (0,1,2)
            # 2. crop outer bound =========================================
            image, meta = crop_image_zeroOut3D(image, tol=0.2, min_size=(288, 176, 176), pad_value=0.0)
            # 3. change smaller w to the last axes =======================================
            if image.shape[1] < image.shape[2]: image = np.transpose(image, axes=(0,2,1)); reverse3 = (0,2,1)
            else: reverse3 = (0,1,2)

            patchIdxAll = get_patchList_Test(img=image, patch_size=(288, 176, 128), patch_moving_stride=(18, 11, 8), validValueThresh=0.2)
            patchAll_result = []
            
            for ppIdx in range(0, len(patchIdxAll), args.patchBatchSize):
                if ppIdx <= (len(patchIdxAll)-args.patchBatchSize): BatchSizeCurrent = args.patchBatchSize
                else: BatchSizeCurrent = len(patchIdxAll) - (len(patchIdxAll)//args.patchBatchSize) * args.patchBatchSize

                patchesCurrent = []
                for bbIdx in range(BatchSizeCurrent):
                    boxPP = patchIdxAll[ppIdx+bbIdx]
                    patchesCurrent.append(image[boxPP[0]:boxPP[1], boxPP[2]:boxPP[3], boxPP[4]:boxPP[5]])

                lowS = torch.from_numpy(np.array(patchesCurrent)).cuda().float().unsqueeze(dim=1)
                with torch.inference_mode():
                    vision_rec = model.forward_den(images=lowS, images_gt=None,
                                                   dose=torch.tensor(dose, dtype=torch.float32, device=lowS.device).repeat(lowS.shape[0]),
                                                   text_emb=None)
                predS = vision_rec.clone()
                predS[predS <= 0.0] = 0.0
                predS = predS[:,0,:,:,:].detach().cpu().numpy()

                for bbIdx in range(BatchSizeCurrent):
                    patchAll_result.append(predS[bbIdx])
            
            predIm_result = reverseImg(img_size=image.shape, predictions=patchAll_result, patchIdx=patchIdxAll)
            predIm_result = np.transpose(predIm_result, axes=reverse3)
            
            predFinal = uncrop_image_zeroOut3D(predIm_result, meta)
            predFinal = np.transpose(predFinal, axes=reverse1)
            predFinal[predFinal < 0.0] = 0.0
            # plt.imsave('/data22/user/mx79/c1Fig.png', sitk.GetArrayFromImage(imageOri)[:,128,:], cmap='jet', vmin=0, vmax=4)
            # plt.imsave('/data22/user/mx79/c3Fig.png', predFinal[:,128,:], cmap='jet', vmin=0, vmax=4)
            # plt.imsave('/data22/user/mx79/c2Fig.png', label[:,128,:], cmap='jet', vmin=0, vmax=4)
                
            predSave = sitk.GetImageFromArray(predFinal)
            predSave.CopyInformation(imageOri)
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            sitk.WriteImage(predSave, savePath)
            print('saving: ', savePath)

if __name__ == "__main__":
    main()