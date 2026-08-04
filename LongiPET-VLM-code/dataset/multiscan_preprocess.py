import numpy as np
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
from skimage.transform import resize
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
target_shape = (288, 176, 128)

def crop_image_zeroOut3D(image, threshold=0.2, image_hc=None, seg=None, ct=None, image_pre=None):
    s0, s1, s2 = np.max(image, (1,2)) > threshold, np.max(image, (0,2)) > threshold, np.max(image, (0,1)) > threshold
    x,y,z = image.shape
    s0_start, s0_end = s0.argmax(), x-s0[::-1].argmax()
    s1_start, s1_end = s1.argmax(), y-s1[::-1].argmax()
    s2_start, s2_end = s2.argmax(), z-s2[::-1].argmax()
    image_crop = image[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end]
    image_hc_crop = image_hc[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end] if image_hc is not None else None
    seg_crop = seg[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end] if seg is not None else None
    ct_crop = ct[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end] if ct is not None else None
    image_pre_crop = image_pre[s0_start:s0_end, s1_start:s1_end, s2_start:s2_end] if image_pre is not None else None
    return image_crop, image_hc_crop, seg_crop, ct_crop, image_pre_crop

def move_axis_with_fewest_slices_to_last(image, image_hc=None, seg=None, ct=None, image_pre=None):
    assert image.ndim == 3, f"Expected 3D volume, got {image.ndim}D"
    
    valid_counts = [image.shape[0], image.shape[1], image.shape[2]]
    moved_axis = int(np.argmin(valid_counts))
    perm = [ax for ax in range(3) if ax != moved_axis] + [moved_axis]

    image = np.transpose(image, perm)
    if image_hc is not None: image_hc = np.transpose(image_hc, perm)
    if seg is not None: seg = np.transpose(seg, perm)
    if ct is not None: ct = np.transpose(ct, perm)
    if image_pre is not None: image_pre = np.transpose(image_pre, perm)
    return image, image_hc, seg, ct, image_pre

def getValidBodyMask(img, threshold=0.1):
    mask = binary_fill_holes(img > threshold)
    for ii in range(mask.shape[2]): mask[:,:,ii] = binary_fill_holes(mask[:,:,ii])
    for ii in range(mask.shape[1]): mask[:,ii,:] = binary_fill_holes(mask[:,ii,:])
    for ii in range(mask.shape[0]): mask[ii,:,:] = binary_fill_holes(mask[ii,:,:])
    return mask

def extract_top_valid_slices(image, axis=0, num_slices=300, threshold=0.2, fluctuation_win=30, seg=None,):
    assert len(image.shape) == 3, "image must be 3D"
    assert axis in [0, 1, 2], "axis must be 0, 1, or 2"
    axis_len = image.shape[axis]
    assert axis_len >= num_slices, "Image too small along chosen axis"
    
    if seg is not None:
        assert seg.shape == image.shape, "seg must have same shape as image"
        if axis == 0: per_slice_counts = np.sum(seg > 0.5, axis=(1, 2))
        elif axis == 1: per_slice_counts = np.sum(seg > 0.5, axis=(0, 2))
        else: per_slice_counts = np.sum(seg > 0.5, axis=(0, 1))

        if np.any(per_slice_counts > 0):
            window_sums = np.convolve(per_slice_counts, np.ones(num_slices, dtype=np.int64), mode="valid")
            best_start = int(np.argmax(window_sums))
            start_slice = best_start + np.random.randint(-fluctuation_win, fluctuation_win + 1)
            start_slice = max(0, min(start_slice, axis_len - num_slices))
            return [start_slice, start_slice + num_slices]

    if axis == 0: per_slice_counts = np.sum(image > threshold, axis=(1, 2))
    elif axis == 1: per_slice_counts = np.sum(image > threshold, axis=(0, 2))
    else: per_slice_counts = np.sum(image > threshold, axis=(0, 1))

    window_sums = np.convolve(per_slice_counts, np.ones(num_slices, dtype=np.int64), mode="valid")
    best_start = int(np.argmax(window_sums))
    start_slice = best_start + np.random.randint(-fluctuation_win, fluctuation_win + 1)
    start_slice = max(0, min(start_slice, axis_len - num_slices))
    
    return [start_slice, start_slice + num_slices]

def random_crop_3d(image, target_shape, image_hc=None, seg=None, ct=None, image_pre=None):
    ih, iw, id_ = image.shape
    th, tw, td = target_shape

    pad = [(max(0, th - ih), 0), (max(0, tw - iw), 0), (max(0, td - id_), 0)]
    if any(p[0] > 0 for p in pad):
        image = np.pad(image, pad, mode='constant', constant_values=image.min())
        if image_hc is not None: image_hc = np.pad(image_hc, pad, mode='constant', constant_values=image_hc.min())
        if seg is not None: seg = np.pad(seg, pad, mode='constant', constant_values=seg.min())
        if ct is not None: ct = np.pad(ct, pad, mode='constant', constant_values=ct.min())
        if image_pre is not None: image_pre = np.pad(image_pre, pad, mode='constant', constant_values=image_pre.min())

    ih, iw, id_ = image.shape
    h0, w0, d0 = random.randint(0, ih - th), random.randint(0, iw - tw), random.randint(0, id_ - td)
    image = image[h0:h0+th, w0:w0+tw, d0:d0+td]
    if image_hc is not None: image_hc = image_hc[h0:h0+th, w0:w0+tw, d0:d0+td]
    if seg is not None: seg = seg[h0:h0+th, w0:w0+tw, d0:d0+td]
    if ct is not None: ct = ct[h0:h0+th, w0:w0+tw, d0:d0+td]
    if image_pre is not None: image_pre = image_pre[h0:h0+th, w0:w0+tw, d0:d0+td]
    return image, image_hc, seg, ct, image_pre


def pad_or_crop_to_shape(image, image_hc=None, seg=None, ct=None, image_pre=None,
                        currentTask='REPORT', target_shape=target_shape, threshold=0.2):
    oriShape = image.shape
    if currentTask == 'REPORT' or currentTask == 'VQA':
        if oriShape[0] > target_shape[0] + 50:
            cropBox = extract_top_valid_slices(image, axis=0, num_slices=target_shape[0]+50, threshold=threshold, seg=seg)
            image = image[cropBox[0]:cropBox[1], :, :]
            if image_hc is not None: image_hc = image_hc[cropBox[0]:cropBox[1], :, :]
            if seg is not None: seg = seg[cropBox[0]:cropBox[1], :, :]
            if ct is not None: ct = ct[cropBox[0]:cropBox[1], :, :]
            if image_pre is not None: image_pre = image_pre[cropBox[0]:cropBox[1], :, :]
        if oriShape[1] > target_shape[1] + 50: 
            cropBox = extract_top_valid_slices(image, axis=1, num_slices=target_shape[1]+50, threshold=threshold, seg=seg)
            image = image[:, cropBox[0]:cropBox[1], :]
            if image_hc is not None: image_hc = image_hc[:, cropBox[0]:cropBox[1], :]
            if seg is not None: seg = seg[:, cropBox[0]:cropBox[1], :]
            if ct is not None: ct = ct[:, cropBox[0]:cropBox[1], :]
            if image_pre is not None: image_pre = image_pre[:, cropBox[0]:cropBox[1], :]
        if oriShape[2] > target_shape[2] + 50:
            cropBox = extract_top_valid_slices(image, axis=2, num_slices=target_shape[2]+50, threshold=threshold, seg=seg)
            image = image[:, :, cropBox[0]:cropBox[1]]
            if image_hc is not None: image_hc = image_hc[:, :, cropBox[0]:cropBox[1]]
            if seg is not None: seg = seg[:, :, cropBox[0]:cropBox[1]]
            if ct is not None: ct = ct[:, :, cropBox[0]:cropBox[1]]
            if image_pre is not None: image_pre = image_pre[:, :, cropBox[0]:cropBox[1]]
        image = resize(image, target_shape, anti_aliasing=False)
        if image_hc is not None: image_hc = resize(image_hc, target_shape, anti_aliasing=False)
        if seg is not None: seg = resize(seg, target_shape, anti_aliasing=False, order=0)
        if ct is not None: ct = resize(ct, target_shape, anti_aliasing=False)
        if image_pre is not None: image_pre = resize(image_pre, target_shape, anti_aliasing=False)
    
    elif currentTask == 'SEG' or currentTask == 'DEN':
        image, image_hc, seg, ct, image_pre = random_crop_3d(image=image, target_shape=target_shape, image_hc=image_hc, seg=seg, ct=ct, image_pre=image_pre)
    else: raise ValueError(f"Unknown currentTask: {currentTask}")
    
    return image, image_hc, seg, ct, image_pre

def get_img_from_path(image_path, image_path_hc=None, currentTask='REPORT', target_shape=target_shape, suvThreshold=0.2,
                        image_path_pre=None, tumor_path=None, ct_path=None):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # read image (suv unit)
    if np.max(image) < 5.0: suvThreshold=0.1

    if image_path_hc is not None: 
        image_hc = sitk.GetArrayFromImage(sitk.ReadImage(image_path_hc)) 
        assert image.shape == image_hc.shape
    else: # if exist higher-quality image
        image_hc = None
    if ct_path is not None: 
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        assert image.shape == ct.shape
        validBodyMask = getValidBodyMask(image, threshold=suvThreshold); 
        ct[validBodyMask<0.5] = ct.min()
        ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-15)
    else: 
        ct = None
    if tumor_path is not None:
        seg = sitk.GetArrayFromImage(sitk.ReadImage(tumor_path))
        assert image.shape == seg.shape
        seg = np.where(seg>=0.5, 1.0, 0.0)
    else:
        seg = None
    if image_path_pre is not None: # if exist history scan
        image_pre = sitk.GetArrayFromImage(sitk.ReadImage(image_path_pre))
        # plt.imsave('y1.png',image[:,:,100], cmap='jet', vmin=0, vmax=5)
        # plt.imsave('y2.png',image_pre[-image.shape[0]::,:,100], cmap='jet', vmin=0, vmax=5)
        image_pre = image_pre[-image.shape[0]::, :, :]
        image_pre = resize(image_pre, image.shape, anti_aliasing=False)
    else:
        image_pre = None
    
    image, image_hc, seg, ct, image_pre = crop_image_zeroOut3D(image=image, threshold=suvThreshold, image_hc=image_hc, seg=seg, ct=ct, image_pre=image_pre)
    image, image_hc, seg, ct, image_pre = move_axis_with_fewest_slices_to_last(image=image, image_hc=image_hc, seg=seg, ct=ct, image_pre=image_pre)
    image, image_hc, seg, ct, image_pre = pad_or_crop_to_shape(image=image, image_hc=image_hc, seg=seg, ct=ct, image_pre=image_pre,
                                                    currentTask=currentTask, target_shape=target_shape, threshold=suvThreshold)
    
    # plt.imsave('y1.png',image[:,128,:], cmap='jet', vmin=0, vmax=5)
    return (image[np.newaxis, ...], 
            image_hc[np.newaxis, ...] if image_hc is not None else None,
            image_pre[np.newaxis, ...] if image_pre is not None else None,
            seg[np.newaxis, ...]     if seg     is not None else None,
            ct[np.newaxis, ...]      if ct      is not None else None)

def get_text_from_path(text_path):
    priorInfo, diagnoInfo = '', ''

    if not os.path.exists(text_path):
        if 'mCT_dataset/' in text_path:
            tracer = text_path.split('02.Yale_mCT_dataset/')[-1].split('/')[0].replace('_noReports','')
            priorInfo = 'Imaging device: Siemens Biograph mCT. Tracer: '+ tracer + '.'

        elif 'uExplorer_FDG' in text_path:
            priorInfo = 'Imaging device: United Imaging uEXPLORER. Tracer: 18F-FDG.'
        
        elif 'Vision_dataset' in text_path:
            priorInfo = 'Imaging device: Siemens Biograph Vision Quadra. Tracer: 18F-FDG.'
    
    else:
        with open(text_path, 'r') as text_file:
            raw_text = text_file.read()
        cleaned_text = raw_text.replace('#', '').replace('Radiology Report','').replace('\n',' ').replace('-',' ')
        cleaned_text = re.sub(r'\s+', ' ', re.sub(r'\.+', '.', cleaned_text)).strip() # remove multiple spaces and dots

        try: # text after Patient Information, before Scanning Technique
            priorInfo = cleaned_text.split('**Patient Information:**')[1].split('**Scanning Technique:**')[0]
            if '**Comparison:**' in priorInfo: priorInfo = priorInfo.split('**Comparison:**')[0]
            if '**Impression:**' in priorInfo: priorInfo = priorInfo.split('**Impression:**')[0]
            priorInfo = priorInfo.replace('*', '')
        except IndexError:
            print("Failed to extract priorInfo from: ", text_path)
        
        try: # text after Impression
            diagnoInfo = cleaned_text.split('**Impression:**')[1]
            diagnoInfo = diagnoInfo.replace('*', '')
        except IndexError:
            print("Failed to extract diagnoInfo from: ", text_path)

        if '_mCT_dataset/' in text_path:
            tracer = text_path.split('_mCT_dataset/')[-1].split('/')[0].replace('_noReports','')
            priorInfo = re.sub(r'\s+', ' ', re.sub(r'\.+', '.', priorInfo.strip() + '. Imaging device: Siemens Biograph mCT. Tracer: '+ tracer + '.')).strip()

    return priorInfo.strip(), diagnoInfo.strip()