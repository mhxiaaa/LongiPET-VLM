import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass

@dataclass
class CropMeta:
    orig_shape: tuple         # original image shape (x, y, z)
    box: list                 # [x0, x1, y0, y1, z0, z1] (end-exclusive)
    slices: tuple             # (slice(x0,x1), slice(y0,y1), slice(z0,z1))
    pad_before: tuple         # (px0, py0, pz0)
    pad_after: tuple          # (px1, py1, pz1)
    pad_value: float = 0.0

def crop_image_zeroOut3D(img, tol=0, min_size=(0,0,0), pad_value=0):
    x, y, z = img.shape
    a1, a2, a3 = min_size

    s0, s1, s2 = np.max(img, axis=(1,2)) > tol, np.max(img, axis=(0,2)) > tol, np.max(img, axis=(0,1)) > tol

    def _tight_bounds(s, dim, min_len):
        if s.any():
            start = int(s.argmax())
            end = int(dim - s[::-1].argmax())  # exclusive
        else:
            start = max(0, (dim - min_len)//2)
            end = min(dim, start + min_len)
        end = max(end, start+1)
        return start, end

    x0, x1 = _tight_bounds(s0, x, max(1,a1))
    y0, y1 = _tight_bounds(s1, y, max(1,a2))
    z0, z1 = _tight_bounds(s2, z, max(1,a3))

    def _expand(start, end, dim, need):
        cur = end - start
        if cur >= need: return start, end
        add = need - cur
        left = add//2; right = add - left
        start -= left; end += right
        start = max(0, start); end = min(dim, end)
        return start, end

    x0, x1 = _expand(x0,x1,x,a1)
    y0, y1 = _expand(y0,y1,y,a2)
    z0, z1 = _expand(z0,z1,z,a3)

    slices = (slice(x0,x1), slice(y0,y1), slice(z0,z1))
    crop = img[slices]

    # pad if clipped by borders
    need_x, need_y, need_z = max(0, a1 - crop.shape[0]), max(0, a2 - crop.shape[1]), max(0, a3 - crop.shape[2])

    px0, px1 = need_x//2, need_x - need_x//2
    py0, py1 = need_y//2, need_y - need_y//2
    pz0, pz1 = need_z//2, need_z - need_z//2

    if need_x or need_y or need_z:
        crop = np.pad(crop, ((px0, px1),(py0, py1),(pz0, pz1)), mode="constant", constant_values=pad_value)

    meta = CropMeta(orig_shape=img.shape, box=[x0,x1,y0,y1,z0,z1], slices=slices,
                    pad_before=(px0,py0,pz0), pad_after=(px1,py1,pz1), pad_value=pad_value)
    return crop, meta

def uncrop_image_zeroOut3D(crop_like: np.ndarray, meta: CropMeta):
    """
    Inverse of crop+pad:
      - strip the padding recorded in meta
      - paste back into a zeros canvas of meta.orig_shape at meta.slices
    """
    x0,x1,y0,y1,z0,z1 = meta.box
    px0,py0,pz0 = meta.pad_before
    px1,py1,pz1 = meta.pad_after

    # 1) remove padding added during crop (if any)
    x_end = None if px1 == 0 else -px1
    y_end = None if py1 == 0 else -py1
    z_end = None if pz1 == 0 else -pz1
    unpadded = crop_like[px0:x_end, py0:y_end, pz0:z_end]
    # Sanity: match the intended box size
    bx, by, bz = (x1-x0, y1-y0, z1-z0)
    assert unpadded.shape == (bx,by,bz), f"Shape mismatch after unpad: {unpadded.shape} vs box {(bx,by,bz)}"
    # 2) paste into original space
    canvas = np.full(meta.orig_shape, meta.pad_value, dtype=unpadded.dtype)
    canvas[x0:x1, y0:y1, z0:z1] = unpadded
    return canvas

def crop_like_with_meta(arr, meta, pad_value=None):
    """
    Crop `arr` with meta.slices and pad to the same shape as the cropped image.
    Works with dict-like or dataclass meta.
    """
    # support both dict and dataclass
    slices     = meta["slices"]      if isinstance(meta, dict) else meta.slices
    pad_before = meta["pad_before"]  if isinstance(meta, dict) else meta.pad_before
    pad_after  = meta["pad_after"]   if isinstance(meta, dict) else meta.pad_after
    if pad_value is None:
        pad_value = meta["pad_value"] if isinstance(meta, dict) else meta.pad_value

    cropped = arr[slices]

    # pad if the original crop added padding
    px0, py0, pz0 = pad_before
    px1, py1, pz1 = pad_after
    if px0 or py0 or pz0 or px1 or py1 or pz1:
        cropped = np.pad(cropped, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant", constant_values=pad_value,)
    return cropped

def get_patchList_Test(img, patch_size, patch_moving_stride, validValueThresh=0.15):
    img_size = img.shape
    imax, jmax, kmax = img_size[1] - patch_size[1], img_size[2] - patch_size[2], img_size[0] - patch_size[0]
    irange = list(range(0, imax+1, patch_moving_stride[1]))
    jrange = list(range(0, jmax+1, patch_moving_stride[2]))
    krange = list(range(0, kmax+1, patch_moving_stride[0]))
    if irange[-1] != imax:
        irange.append(imax)
    if jrange[-1] != jmax:
        jrange.append(jmax)
    if krange[-1] != kmax:
        krange.append(kmax)
    patchIdx = []
    for k in krange:
        for j in jrange:
            for i in irange:
                box = [k, k+patch_size[0], i, i+patch_size[1], j, j+patch_size[2]]
                imC = img[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
                if imC.max() > validValueThresh:
                    patchIdx.append(box)
    return patchIdx

def reverseImg(img_size, predictions, patchIdx):
    denoised_img = np.float32(np.zeros(img_size))
    
    patch_size = predictions[0].shape
    gaussian_mask = np.zeros(patch_size, dtype=float)
    gaussian_mask[int(patch_size[0] / 2 - 1):int(patch_size[0] / 2 + 1),
                  int(patch_size[1] / 2 - 1):int(patch_size[1] / 2 + 1),
                  int(patch_size[2] / 2 - 1):int(patch_size[2] / 2 + 1)] = 1
    gaussian_mask = gaussian_filter(gaussian_mask, patch_size[0] / 4, truncate=2, mode='nearest')
    gaussian_mask = gaussian_mask / np.amax(gaussian_mask)
    
    blending_mask = np.zeros(img_size, dtype=float)
    for ppIdx in range(len(patchIdx)):
        box = patchIdx[ppIdx]
        patchC = predictions[ppIdx]
        blending_mask[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += gaussian_mask
        denoised_img[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += patchC * gaussian_mask
    blending_mask = 1 / blending_mask
    denoised_img = denoised_img * blending_mask
    denoised_img[np.where(np.isnan(denoised_img))] = 0
    denoised_img[np.where(denoised_img <= 0)] = 0
    return denoised_img

def reverseSeg(img_size, predictions, patchIdx):
    numClasses = predictions[0].shape[0]
    
    patch_size = predictions[0][0, :, :, :].shape
    gaussian_mask = np.zeros(patch_size, dtype=float)
    gaussian_mask[int(patch_size[0] / 2 - 1):int(patch_size[0] / 2 + 1),
                  int(patch_size[1] / 2 - 1):int(patch_size[1] / 2 + 1),
                  int(patch_size[2] / 2 - 1):int(patch_size[2] / 2 + 1)] = 1
    gaussian_mask = gaussian_filter(gaussian_mask, patch_size[0] / 4, truncate=2, mode='nearest')
    gaussian_mask = gaussian_mask / np.amax(gaussian_mask)

    seg_all = []
    for c in range(numClasses):
        denoised_img = np.float32(np.zeros(img_size))
        blending_mask = np.zeros(img_size, dtype=float)
        for ppIdx in range(len(patchIdx)):
            box = patchIdx[ppIdx]
            patchC = predictions[ppIdx][c, :, :, :]
            blending_mask[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += gaussian_mask
            denoised_img[box[0]:box[1], box[2]:box[3], box[4]:box[5]] += patchC * gaussian_mask
        blending_mask = 1 / blending_mask
        denoised_img = denoised_img * blending_mask
        denoised_img[np.where(np.isnan(denoised_img))] = 0
        denoised_img[np.where(denoised_img <= 0)] = 0
        
        seg_all.append(denoised_img)
    return np.array(seg_all)
