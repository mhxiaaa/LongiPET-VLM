import torch
import random
import re
from datetime import datetime
import monai.transforms as mtf
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.multiscan_preprocess import get_img_from_path, get_text_from_path
import matplotlib.pyplot as plt

src_dir, dst_dir = '/data/0dataset/', '/home/project/0dataset/'

train_transform = mtf.Compose([mtf.RandFlipd(keys=["im1", "im2"], prob=0.10, spatial_axis=0, allow_missing_keys=True),
                                mtf.RandFlipd(keys=["im1", "im2"], prob=0.10, spatial_axis=1, allow_missing_keys=True), 
                                mtf.RandFlipd(keys=["im1", "im2"], prob=0.10, spatial_axis=2, allow_missing_keys=True),
                                mtf.ToTensord(keys=["im1", "im2"], dtype=torch.float, allow_missing_keys=True)])
val_transform = mtf.Compose([mtf.ToTensord(keys=["im1", "im2"], dtype=torch.float, allow_missing_keys=True)])

train_transform_seg = mtf.Compose([mtf.RandFlipd(keys=["image", "imagePre", "seg"], prob=0.10, spatial_axis=0, allow_missing_keys=True),
                                    mtf.RandFlipd(keys=["image", "imagePre", "seg"], prob=0.10, spatial_axis=1, allow_missing_keys=True),
                                    mtf.RandFlipd(keys=["image", "imagePre", "seg"], prob=0.10, spatial_axis=2, allow_missing_keys=True),
                                    mtf.RandScaleIntensityd(keys=["image", "imagePre"], factors=0.01, prob=0.05, allow_missing_keys=True),
                                    mtf.RandShiftIntensityd(keys=["image", "imagePre"], offsets=0.01, prob=0.05, allow_missing_keys=True),
                                    mtf.ToTensord(keys=["image", "imagePre", "seg"], dtype=torch.float, allow_missing_keys=True)])
val_transform_seg = mtf.Compose([mtf.ToTensord(keys=["image", "imagePre", "seg"], dtype=torch.float, allow_missing_keys=True)])

train_transform_den = mtf.Compose([mtf.RandFlipd(keys=["input", "output", "input_pre"], prob=0.10, spatial_axis=0, allow_missing_keys=True),
                                   mtf.RandFlipd(keys=["input", "output", "input_pre"], prob=0.10, spatial_axis=1, allow_missing_keys=True),
                                   mtf.RandFlipd(keys=["input", "output", "input_pre"], prob=0.10, spatial_axis=2, allow_missing_keys=True),
                                   mtf.RandCoarseDropoutd(keys=["input"], holes=15, spatial_size=(32, 32, 16), fill_value=0.0, prob=0.7),
                                   mtf.ToTensord(keys=["input", "output", "input_pre"], dtype=torch.float, allow_missing_keys=True)])
val_transform_den = mtf.Compose([mtf.ToTensord(keys=["input", "output", "input_pre"], dtype=torch.float, allow_missing_keys=True)])

def get_date_from_path(path):
    m = re.search(r'[A-Za-z0-9]+_(\d{8})_\d+', path) # → YYYYMMDD
    if m: 
        return datetime.strptime(m.group(1), '%Y%m%d')
    m = re.search(r'[/_]\d+scans/(\d{4}-\d{2}-\d{2})/', path) # → YYYY-MM-DD
    if m: 
        return datetime.strptime(m.group(1), '%Y-%m-%d')
    m = re.search(r'(\d{2}-\d{2}-\d{4})-NA-', path) # MM-DD-YYYY-NA or MM-DD-YYYY (in folder names)
    if m: 
        return datetime.strptime(m.group(1), '%m-%d-%Y')
    assert False, f"Date not found in path: {path}"

def get_text_info(priorInfo, diagnoInfo, tokenizer, image_tokens, max_length, proj_out_num,
                  taskPrompt, taskPrompt_len, imagePre_tokens=None, text_path=None):
    answer = ' '.join(diagnoInfo.split())
    answer_len = len(tokenizer.encode(answer, add_special_tokens=False))
    pre_tokens_len = proj_out_num if imagePre_tokens is not None else 0
    prior_budget = max_length - proj_out_num - pre_tokens_len - taskPrompt_len - answer_len - 2 # +2: separator space + EOS token

    prior_ids = tokenizer.encode(" ".join(priorInfo.split()), add_special_tokens=False)
    if len(prior_ids) > prior_budget:
        prior_ids = prior_ids[:max(0, prior_budget)]
        priorInfo = tokenizer.decode(prior_ids, skip_special_tokens=True)
        print(taskPrompt, '; over length: ', text_path)

    question = image_tokens + taskPrompt + priorInfo
    if imagePre_tokens is not None:
        question = question + imagePre_tokens
    
    text_tensor = tokenizer(question + " " + answer, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt",)
    input_id = text_tensor["input_ids"][0] # size [4096] concatenated question+answer tokens
    attention_mask = text_tensor["attention_mask"][0] # tells which tokens are real, which are padding

    valid_len = torch.sum(attention_mask)
    if valid_len < len(input_id): input_id[valid_len] = tokenizer.eos_token_id # Forces an eos_token right after the last non-pad token
    else: raise ValueError(f"length should not be over after truncated: {text_path}")

    question_tensor = tokenizer(question, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt",)
    question_len = torch.sum(question_tensor["attention_mask"][0])

    label = input_id.clone() # label is masked so that only the answer portion is not -100 and used for loss computation
    label[:question_len] = -100
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        label[label == tokenizer.pad_token_id] = -100
        if valid_len < len(label): label[valid_len] = tokenizer.eos_token_id
    else:
        label[label == tokenizer.pad_token_id] = -100
    return question, answer, input_id, attention_mask, label


class CLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args, self.tokenizer, self.mode = args, tokenizer, mode
        set_track_meta(False)
        if mode == 'train': self.transform, self.data_list = train_transform, args.REPORT_train
        elif mode == 'val': self.transform, self.data_list = val_transform, args.REPORT_val
        elif mode == 'test': self.transform, self.data_list = val_transform, args.REPORT_test
        else: print("the mode train/val/test is not defined ! ")

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens, text_path):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        print('CLIP task: text over length: ', text_path)
        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            first = sentences.pop(0)
            selected_sentences.append(first)
            current_tokens = count_tokens(first)

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        data_list_idx = self.data_list[idx]
        pet_path, text_path, dose = data_list_idx["image"].replace(src_dir, dst_dir), data_list_idx["report"].replace(src_dir, dst_dir), data_list_idx["dose_tag"]
        
        priorInfo, diagnoInfo = get_text_from_path(text_path)

        image, _, _, _, _ = get_img_from_path(image_path=pet_path, currentTask='REPORT')
        image = self.transform({"im1": image})["im1"]
        # plt.imsave('y1.png',image[0,:,40,:], cmap='jet', vmin=0, vmax=0.2)
        
        text = self.truncate_text(priorInfo + ' ' + diagnoInfo, self.args.max_length, text_path)
        text_tensor = self.tokenizer(text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
        
        input_id = text_tensor["input_ids"][0]  # integers corresponding to tokens in text, based on the tokenizer’s vocabulary 
        attention_mask = text_tensor["attention_mask"][0] # indicates which tokens are real input and which are padding

        ret = {'images': image, 'text': text, 'dose': torch.tensor(dose, dtype=torch.float32), 'input_ids': input_id, 
               'attention_mask': attention_mask, "image_path": pet_path, "task_type": "CLIP"}
        return ret

class RefVAEDataset(Dataset):
    def __init__(self, args, mode="train", tokenizer=None):
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.image_tokens, self.imagePre_tokens = "<im_patch>" * args.proj_out_num, "<imPre_patch>" * args.proj_out_num
            self.task_prompt = ' Task: DEN. '
            self.taskPrompt_len = len(tokenizer.encode(self.task_prompt, add_special_tokens=False))
        set_track_meta(False)

        if mode == 'train': self.transform, self.data_list = train_transform_den, args.DEN_train
        elif mode == 'val': self.transform, self.data_list = val_transform_den, args.DEN_val
        elif mode == 'test': self.transform, self.data_list = val_transform_den, args.DEN_test
        else: print("the mode train/val/test is not defined ! ")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_list_idx = self.data_list[idx]
        pet_path, petHC_path, dose = data_list_idx["low_dose"].replace(src_dir, dst_dir), data_list_idx["high_dose"].replace(src_dir, dst_dir), data_list_idx["dose_tag"]
        pre_pet_path = data_list_idx["previous_high_doses"]
        
        if pre_pet_path and self.tokenizer is not None: 
            pre_pet_path = pre_pet_path[0].replace(src_dir, dst_dir)
            image, image_hc, image_pre, _, _ = get_img_from_path(image_path=pet_path, image_path_hc=petHC_path, 
                                                                 image_path_pre=pre_pet_path, currentTask='DEN')
            it = self.transform({"input": image, "output": image_hc, "input_pre": image_pre})
            image, image_hc, image_pre = it['input'], it['output'], it["input_pre"]
        else:
            image, image_hc, _, _, _ = get_img_from_path(image_path=pet_path, image_path_hc=petHC_path, currentTask='DEN')
            it = self.transform({"input": image, "output": image_hc})
            image, image_hc = it['input'], it['output']
            image_pre = None
        # plt.imsave('y1.png',image[0,:,128,:], cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y2.png',image_hc[0,:,128,:], cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y3.png',image[0,:,:,64], cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y4.png',image_hc[0,:,:,64], cmap='jet', vmin=0, vmax=5)

        ret = {'images': image, 'images_gt': image_hc, 'dose': torch.tensor(dose, dtype=torch.float32),
                "images_pre": image_pre, 
               "image_path": pet_path, "image_hc_path": petHC_path, "task_type": 'DEN'}
        
        assert torch.isfinite(image).all(), f"DEN image has NaN/Inf: {pet_path}"
        assert torch.isfinite(image_hc).all(), f"DEN image_hc has NaN/Inf: {petHC_path}"

        if self.tokenizer is not None:
            text_path = os.path.join(os.path.dirname(pet_path), 'report.txt')
            priorInfo, _ = get_text_from_path(text_path)

            if pre_pet_path:
                pre_text_path = os.path.join(os.path.dirname(pre_pet_path), 'report.txt')
                if os.path.exists(pre_text_path):
                    _, pre_diagnoInfo = get_text_from_path(pre_text_path)
                    date_cur, date_pre = get_date_from_path(text_path), get_date_from_path(pre_text_path)
                    timeDiff = (date_cur.year - date_pre.year) * 12 + (date_cur.month - date_pre.month)
                    assert timeDiff is not None
                    priorInfo = priorInfo + f' Report from {timeDiff} months ago: ' + pre_diagnoInfo + ' '

            question, answer, input_id, attention_mask, label = get_text_info(priorInfo=priorInfo, diagnoInfo='DEN result:',
                tokenizer=self.tokenizer, image_tokens=self.image_tokens,
                imagePre_tokens=self.imagePre_tokens if image_pre is not None else None,
                max_length=self.args.max_length, proj_out_num=self.args.proj_out_num,
                taskPrompt=self.task_prompt, taskPrompt_len=self.taskPrompt_len,
                text_path=os.path.join(os.path.dirname(pet_path), 'report.txt'))

            ret.update({'input_id': input_id, 'label': label, 'attention_mask': attention_mask, 'question': question, 'answer': answer})
        return ret

class RefSegDataset(Dataset):
    def __init__(self, args, mode="train", tokenizer=None):
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.image_tokens, self.imagePre_tokens = "<im_patch>" * args.proj_out_num, "<imPre_patch>" * args.proj_out_num
            self.task_prompt = ' Task: SEG. '
            self.task_prompt_len = len(tokenizer.encode(self.task_prompt, add_special_tokens=False))
        set_track_meta(False)

        if mode == 'train': self.transform, self.data_list = train_transform_seg, args.SEG_train
        elif mode == 'val': self.transform, self.data_list = val_transform_seg, args.SEG_val
        elif mode == 'test': self.transform, self.data_list = val_transform_seg, args.SEG_test
        else: print("the mode train/val/test is not defined ! ")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_list_idx = self.data_list[idx]
        pet_path, tumor_path, dose = data_list_idx["image"].replace(src_dir, dst_dir), data_list_idx["seg"].replace(src_dir, dst_dir), data_list_idx["dose_tag"]
        pre_pet_path = data_list_idx["previous_high_doses"]

        imagePre = None
        if pre_pet_path and self.tokenizer is not None:
            pre_pet_path = pre_pet_path[0].replace(src_dir, dst_dir)
            image, _, imagePre, seg, _ = get_img_from_path(image_path=pet_path, tumor_path=tumor_path, image_path_pre=pre_pet_path, currentTask='SEG')
            it = self.transform({"image": image, "seg": seg, "imagePre": imagePre})
            image, seg, imagePre = it['image'], it['seg'], it["imagePre"]
        else:
            image, _, _, seg, _ = get_img_from_path(image_path=pet_path, tumor_path=tumor_path, currentTask='SEG')
            it = self.transform({"image": image, "seg": seg})
            image, seg = it['image'], it['seg']
        # ttttttt = np.argwhere(seg[0,:,:,:]==1)
        # ttttttt = ttttttt[len(ttttttt)//2][1] if len(ttttttt)>=1 else seg.shape[1]//2
        # plt.imsave('y1.png', image[0,:,ttttttt,:], cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y2.png', seg[0,:,ttttttt,:], cmap='gray', vmin=0, vmax=1.0)

        ret = {'images': image, 'images_pre': imagePre, 'images_gt': seg, 'dose': torch.tensor(dose, dtype=torch.float32), "image_path": pet_path, "image_seg_path": tumor_path, "task_type": "SEG"}

        if not torch.isfinite(image).all(): raise ValueError(f"SEG image has NaN/Inf: {pet_path}")
        if not ((seg == 0) | (seg == 1)).all(): raise ValueError(f"SEG mask has invalid values: {tumor_path}")

        if self.tokenizer is not None:
            text_path = os.path.join(os.path.dirname(pet_path), 'report.txt')
            priorInfo, _ = get_text_from_path(text_path)
            
            if pre_pet_path:
                pre_text_path = os.path.join(os.path.dirname(pre_pet_path), 'report.txt')
                if os.path.exists(pre_text_path):
                    _, pre_diagnoInfo = get_text_from_path(pre_text_path)
                    date_cur, date_pre = get_date_from_path(text_path), get_date_from_path(pre_text_path)
                    timeDiff = (date_cur.year - date_pre.year) * 12 + (date_cur.month - date_pre.month)
                    assert timeDiff is not None
                    priorInfo = priorInfo + f' Report from {timeDiff} months ago: ' + pre_diagnoInfo + ' '

            question, answer, input_id, attention_mask, label = get_text_info(priorInfo=priorInfo, diagnoInfo='SEG result:',
                tokenizer=self.tokenizer, image_tokens=self.image_tokens,
                imagePre_tokens=self.imagePre_tokens if imagePre is not None else None,
                max_length=self.args.max_length, proj_out_num=self.args.proj_out_num,
                taskPrompt=self.task_prompt, taskPrompt_len=self.task_prompt_len,
                text_path=os.path.join(os.path.dirname(pet_path), 'report.txt'))

            ret.update({'input_id': input_id, 'label': label, 'attention_mask': attention_mask, 'question': question, 'answer': answer})
        return ret

class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args, self.tokenizer, self.mode = args, tokenizer, mode
        self.image_tokens, self.imagePre_tokens = "<im_patch>" * args.proj_out_num, "<imPre_patch>" * args.proj_out_num
        
        self.task_prompt = ' Task: Report. '
        self.taskPrompt_len = len(tokenizer.encode(self.task_prompt, add_special_tokens=False))
        set_track_meta(False)

        ids = tokenizer(self.image_tokens, add_special_tokens=False).input_ids
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids]
        assert len(ids) == args.proj_out_num, f"Expected {args.proj_out_num} image tokens, got {len(ids)}"
        assert set(tokens) == {"<im_patch>"}, f"Unexpected tokens: {set(tokens)}"
        
        ids = tokenizer(self.imagePre_tokens, add_special_tokens=False).input_ids
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids]
        assert len(ids) == args.proj_out_num, f"Expected {args.proj_out_num} imagePre tokens, got {len(ids)}"
        assert set(tokens) == {"<imPre_patch>"}, f"Unexpected tokens: {set(tokens)}"

        if mode == 'train': self.transform, self.data_list = train_transform, args.REPORT_train
        elif mode == 'val': self.transform, self.data_list = val_transform, args.REPORT_val
        elif mode == 'test': self.transform, self.data_list = val_transform, args.REPORT_test
        else: print("the mode train/val/test is not defined ! ")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_list_idx = self.data_list[idx]
        pet_path, text_path, dose = data_list_idx["image"].replace(src_dir, dst_dir), data_list_idx["report"].replace(src_dir, dst_dir), data_list_idx["dose_tag"]
        pre_pet_path, pre_text_path = data_list_idx["previous_high_doses"], data_list_idx["previous_reports"]
        
        priorInfo, diagnoInfo = get_text_from_path(text_path)
        
        imagePre = None
        if pre_pet_path:
            pre_pet_path, pre_text_path = pre_pet_path[0].replace(src_dir, dst_dir), pre_text_path[0].replace(src_dir, dst_dir)
            image, _ , imagePre, _, _ = get_img_from_path(image_path=pet_path, image_path_pre=pre_pet_path, currentTask='REPORT')
            assert imagePre.shape == image.shape
            it = self.transform({"im1": image, "im2": imagePre})
            image, imagePre = it["im1"], it["im2"]

            _, pre_diagnoInfo = get_text_from_path(pre_text_path)
            date_cur, date_pre = get_date_from_path(text_path), get_date_from_path(pre_text_path)
            timeDiff = (date_cur.year - date_pre.year) * 12 + (date_cur.month - date_pre.month)
            assert timeDiff is not None
            priorInfo = priorInfo + f' Report from {timeDiff} months ago: ' + pre_diagnoInfo + ' '
        else:
            image, _ , _, _, _ = get_img_from_path(image_path=pet_path, currentTask='REPORT')
            image = self.transform({"im1": image})["im1"]
        # plt.imsave('y1.png',image[0,:,40,:], cmap='jet', vmin=0, vmax=0.2)

        question, answer, input_id, attention_mask, label = get_text_info(priorInfo=priorInfo, diagnoInfo=diagnoInfo,
                tokenizer=self.tokenizer, image_tokens=self.image_tokens,
                imagePre_tokens=self.imagePre_tokens if imagePre is not None else None,
                max_length=self.args.max_length, proj_out_num=self.args.proj_out_num,
                taskPrompt=self.task_prompt, taskPrompt_len=self.taskPrompt_len, text_path=text_path)

        return {"images": image, 'dose': torch.tensor(dose, dtype=torch.float32), 
                "images_gt": None, "images_pre": imagePre, "task_type": "report", 
                "input_id": input_id, "label": label, "attention_mask": attention_mask, 
                "question": question, "answer": answer, "image_path": pet_path}

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args, self.tokenizer, self.mode = args, tokenizer, mode
        self.image_tokens, self.imagePre_tokens = "<im_patch>" * args.proj_out_num, "<imPre_patch>" * args.proj_out_num
        set_track_meta(False)
        if mode == "train":
            self.transform = train_transform
            hpv_list, cancer_list, relapse_list = getattr(args, "VQA_hpv_train", []), getattr(args, "VQA_cls_train", []), getattr(args, "VQA_relapse_train", [])
        elif mode == "val":
            self.transform = val_transform
            hpv_list, cancer_list, relapse_list = getattr(args, "VQA_hpv_val", []), getattr(args, "VQA_cls_val", []), getattr(args, "VQA_relapse_val", [])
        elif mode == "test":
            self.transform = val_transform
            hpv_list, cancer_list, relapse_list = getattr(args, "VQA_hpv_test", []), getattr(args, "VQA_cls_test", []), getattr(args, "VQA_relapse_test", [])
        else: raise ValueError(f"Unknown mode: {mode}")

        self.taskPrompt = {"VQA_hpv": " Task: Predict HPV status. ",
                            "VQA_cancer": " Task: Predict cancer type. ",
                            "VQA_relapse": " Task: Predict RFS status. "}
        self.taskPrompt_len = {"VQA_hpv":     len(tokenizer.encode(self.taskPrompt["VQA_hpv"],  add_special_tokens=False)),
                            "VQA_cancer":  len(tokenizer.encode(self.taskPrompt["VQA_cancer"],  add_special_tokens=False)),
                            "VQA_relapse": len(tokenizer.encode(self.taskPrompt["VQA_relapse"], add_special_tokens=False)),}

        self.data_list = [(tag, item['image'], item['patientInfo'], item['diagnosis'],
                            item.get('relapse_free_survival'), item['dose_tag'], 
                            item['previous_high_doses'], item['previous_diagnosis'])
                    for tag, lst in [("VQA_hpv", hpv_list), ("VQA_cancer", cancer_list), ("VQA_relapse", relapse_list)]
                    for item in lst]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        task_type, pet_path, patientInfo, diagnosis, relapse_free_survival, dose_tag, pet_path_pre, previous_diagnosis = self.data_list[idx]  # tag, image, patientInfo, diagnosis, rfs, dose, prev_doses, prev_diag
        answer = " ".join(diagnosis.split())

        imagePre = None
        if pet_path_pre:
            image, _ , imagePre, _, _ = get_img_from_path(image_path=pet_path.replace(src_dir, dst_dir),
                                                        image_path_pre=pet_path_pre[0].replace(src_dir, dst_dir),
                                                        currentTask='VQA')
            assert imagePre.shape == image.shape
            it = self.transform({"im1": image, "im2": imagePre})
            image, imagePre = it["im1"], it["im2"]

            date_cur, date_pre = get_date_from_path(pet_path), get_date_from_path(pet_path_pre[0])
            timeDiff = (date_cur.year - date_pre.year) * 12 + (date_cur.month - date_pre.month)
            assert timeDiff is not None
            patientInfo = patientInfo + f' Diagnosis from {timeDiff} months ago: ' + previous_diagnosis[0] + '. '
        else:
            image, _, _, _, _ = get_img_from_path(image_path=pet_path.replace(src_dir, dst_dir), currentTask='VQA')
            image = self.transform({"im1": image})["im1"]
        # plt.imsave('y1.png',image[0,:,128,:], cmap='jet', vmin=0, vmax=5)

        if task_type == "VQA_hpv":
            assert answer in ('positive', 'negative'), f"Bad HPV label {answer} for {pet_path}, answer='{answer}'"
        elif task_type == "VQA_cancer":
            assert answer in ("NEGATIVE", "LYMPHOMA", "LUNG_CANCER", "MELANOMA", "PROSTATE_CANCER"), \
                            f"Bad cancer label {answer} for {pet_path}, answer='{answer}'"
        elif task_type == "VQA_relapse":
            assert answer in ('Not relapsed.', 'Relapsed.'), f"Bad relapse label {answer} for {pet_path}, answer='{answer}'"
            assert relapse_free_survival is not None, f"Missing relapse_free_survival for {pet_path}"
            answer = answer + " RFS months: " + str(int(relapse_free_survival/30)) + "."
        else: raise ValueError(f"Unknown task_type: {task_type}")

        question, answer, input_id, attention_mask, label = get_text_info(priorInfo=patientInfo, diagnoInfo=answer,
                tokenizer=self.tokenizer, image_tokens=self.image_tokens,
                imagePre_tokens=self.imagePre_tokens if imagePre is not None else None,
                max_length=self.args.max_length, proj_out_num=self.args.proj_out_num,
                taskPrompt=self.taskPrompt[task_type], taskPrompt_len=self.taskPrompt_len[task_type],
                text_path=os.path.join(os.path.dirname(pet_path), 'report.txt'))

        return {"images": image, 'dose': torch.tensor(dose_tag, dtype=torch.float32), 
                "images_gt": None, "images_pre": imagePre, "task_type": task_type,
                "input_id": input_id, "label": label, "attention_mask": attention_mask,
                "question": question, "answer": answer, "image_path": pet_path}


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [CapDataset(args, tokenizer, mode),
                        VQADataset(args, tokenizer, mode)]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
