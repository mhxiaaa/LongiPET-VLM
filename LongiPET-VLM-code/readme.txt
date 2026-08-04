Data Structure
    As shown in LongiPET-VLM-dataPath/, prepare ***.json files for various tasks.
    PET image is in SUV unit.



step0_vision.py ---> vision pretraining
step1_pretrain_projector.py  --->  projector pretraining
step2_trainVLM.py  --->  language model finetuning
step3_trainVLM_vision.py  --->   vision decoders finetuning
step4_mergeWeights.py  --->   get the model
