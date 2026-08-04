import argparse
import csv
import os
os.environ["CUDA_HOME"] = os.environ.get("EBROOTCUDA", "/apps/software/2024a/software/CUDA/12.6.0")
os.environ["HF_HUB_CACHE"] = "/home/project/huggingface/hub"
import json
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset.multiscan_dataset import VQADataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.llm_qwen import VLMQwenForCausalLM, VLMQwenConfig  # registers vlm_qwen with AutoModel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
    parser.add_argument("--model_name_or_path", type=str, default="/home/project/LongiPET-VLM-model_results/model")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--proj_out_num", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="/home/project/LongiPET-VLM-model_results/eval_vqa.csv",)
    return parser.parse_args(args)

def eval_collate_fn(batch):
    from torch.utils.data.dataloader import default_collate
    result = {}
    for key in batch[0].keys():
        vals = [d[key] for d in batch]
        result[key] = None if vals[0] is None else default_collate(vals)
    return result

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    with open("/home/project/LongiPET-VLM-dataPath/VQA_DiseaseDiagnosis.json", "r", encoding="utf-8") as f: 
        data = json.load(f)
    VQA_cls_train, VQA_cls_val, args.VQA_cls_test = data["train"], data["val"], data["test"]
    print("for test VQA_cls task:", len(args.VQA_cls_test))
    
    with open("/home/project/LongiPET-VLM-dataPath/VQA_HPVstatus.json", "r", encoding="utf-8") as f: 
        data = json.load(f)
    VQA_hpv_train, VQA_hpv_val, args.VQA_hpv_test = data["train"], data["val"], data["test"]
    print("for test VQA_hpv task:", len(args.VQA_hpv_test))

    with open("/home/project/LongiPET-VLM-dataPath/VQA_RFSstatus.json", "r", encoding="utf-8") as f: 
        data = json.load(f)
    VQA_relapse_train, VQA_relapse_val, args.VQA_relapse_test = data["train"], data["val"], data["test"]
    print("for test VQA_relapse task:", len(args.VQA_relapse_test))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=args.max_length,
                                              padding_side="right", use_fast=False,)
    assert tokenizer.convert_tokens_to_ids("<im_patch>") == 151669 and tokenizer.convert_tokens_to_ids("<imPre_patch>") == 151670
    assert len(tokenizer) == 151671
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    model = model.to(device=device)
    model.eval()

    test_dataset = VQADataset(args, tokenizer=tokenizer, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False, drop_last=False,
                                 collate_fn=eval_collate_fn,)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Answer", "Pred", "Correct", "ImagePath"])
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            answer = sample["answer"]
            image_path = sample['image_path']
            
            image = sample["images"].to(device=device)
            images_pre = sample["images_pre"]
            if images_pre is None:
                images_pre, images_pre_mask = None, None
            else:
                has_pre = [x is not None for x in images_pre]
                if any(has_pre):
                    ref = next(x for x in images_pre if x is not None).to(device=device)
                    images_pre = torch.stack([x if x is not None else torch.zeros_like(ref) for x in images_pre]).to(device=device)
                    images_pre_mask = torch.tensor(has_pre, dtype=torch.bool).to(device=device)
                else:
                    images_pre, images_pre_mask = None, None
            
            dose = sample["dose"].to(dtype=torch.float32, device=image.device)
            task_type = sample["task_type"]

            question_tensor = tokenizer(question, return_tensors="pt", padding=True)
            input_id = question_tensor["input_ids"].to(device=device)
            attention_mask = question_tensor["attention_mask"].to(device=device)

            with torch.no_grad():
                generation = model.generate(images=image, images_pre=images_pre, images_pre_mask=images_pre_mask,
                                            input_ids=input_id, dose=dose, attention_mask=attention_mask,
                                            task_type='report',
                                            do_sample=args.do_sample, 
                                            top_p=args.top_p, temperature=args.temperature,
                                            max_new_tokens=args.max_new_tokens,)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            generated_text = generated_texts[0].strip()

            if task_type[0] == 'VQA_hpv' or task_type[0] == 'VQA_cancer':
                correct = 1.0 if generated_text == answer[0] else 0

            elif task_type[0] == 'VQA_relapse':
                correct = 0.5 if generated_text.split('.')[0] == answer[0].split('.')[0] else 0.0

                try: months_pred = float(generated_text.split('months:')[-1].replace('.', '').strip())
                except: months_pred = 0.0
                months_label = float(answer[0].split('months:')[-1].replace('.', '').strip())
                if abs(months_pred - months_label) / max(abs(months_label), 1e-6) < 0.15:
                    correct += 0.5
            else:
                assert False, "Unknown task type: {}".format(sample["task_type"][0])

            writer.writerow([' '.join(question[0].replace("<im_patch>" * args.proj_out_num, "").replace("<imPre_patch>" * args.proj_out_num, "").split()),
                            answer[0],
                            ' '.join(generated_text.split()),
                            correct,
                            image_path[0].replace('/home/project/0dataset/','')])
            print('finished: ', image_path)
            del image, dose, input_id, attention_mask, generation
            torch.cuda.empty_cache()
        
    with open(args.output_path, mode="r") as infile:
        reader = csv.DictReader(infile)
        total = 0
        correct = 0
        for row in reader:
            total += 1
            if row["Correct"] == "1.0":
                correct += 1
            if row["Correct"] == "0.5":
                correct += 0.5
    print('accuracy:', correct / total)
    

if __name__ == "__main__":
    main()
