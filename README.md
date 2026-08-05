# Code for LongiPET-VLM model

Dataset Structure
    As shown in LongiPET-VLM-dataPath/, prepare ***.json files for various tasks.
    PET images are in SUV units.


step0_vision.py ---> vision pretraining
step1_pretrain_projector.py  --->  projector pretraining
step2_trainVLM.py  --->  language model finetuning
step3_trainVLM_vision.py  --->   vision decoders finetuning
step4_mergeWeights.py  --->   get the model

Related Publications:
https://jnm.snmjournals.org/content/67/supplement_1/261701.abstract
(and a journal paper in preparation)
