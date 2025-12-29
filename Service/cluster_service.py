# -*- coding: utf-8 -*-
"""
Service / cluster_service.py
YuNet(检测) + SFace(特征) 的人脸聚类与相册导出
变更：
- 无人脸图片复制到所有“有人”的子集
- 重命名合并：同名目标合并内容，不再生成 _1
- 支持“重建输出”（控制层调用时可先清空再生成）
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import shutil
import uuid

import cv2  # 需要 opencv-contrib-python(-headless)
from Utils.helpers import ensure_dir, copy_or_link

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# 模型路径（确保是“非量化” .onnx）
YUNET_MODEL = os.path.join("models", "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join("models", "face_recognition_sface_2021dec.onnx")


def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _raise_if_model_missing():
    miss = []
    if not os.path.isfile(YUNET_MODEL): miss.append(YUNET_MODEL)
    if not os.path.isfile(SFACE_MODEL): miss.append(SFACE_MODEL)
    if miss:
        raise FileNotFoundError("缺少模型：\n  - " + "\n  - ".join(miss))


def _init_yunet(input_size=(320, 320)):
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "当前 OpenCV 缺少 FaceDetectorYN；请安装 opencv-contrib-python(-headless)。"
        )
    try:
        det = cv2.FaceDetectorYN_create(
            YUNET_MODEL, "", input_size, score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
    except cv2.error as e:
        raise RuntimeError(
            "YuNet ONNX 解析失败：请使用非量化模型，或升级到 opencv-contrib-python-headless==4.10.0.84"
        ) from e
    return det


def _init_sface():
    return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")


def _imread_any(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_embeddings_sface(image_paths, align=True):
    """
    返回：
      feats: (N,D)
      metas: [{"img_path", "face_index", "box_x","box_y","box_w","box_h"} ...]
      face_image_set: set(含有人脸的原图路径)
    """
    _raise_if_model_missing()
    det = _init_yunet()
    recog = _init_sface()

    feats, metas = [], []
    face_image_set = set()

    for p in image_paths:
        img = _imread_any(p)
        if img is None:  # 读图失败
            continue
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        ret = det.detect(img)
        faces = ret[1] if ret is not None else None
        if faces is None or len(faces) == 0:
            continue

        face_image_set.add(p)

        for idx, f in enumerate(faces):
            if align:
                aligned = recog.alignCrop(img, f)
            else:
                x, y, bw, bh = f[:4].astype(int)
                x, y = max(0, x), max(0, y)
                aligned = img[y:y+bh, x:x+bw].copy()

            feat = recog.feature(aligned).astype(np.float32)  # (1,D)
            feats.append(feat)
            x, y, bw, bh = f[:4].astype(int)
            metas.append({
                "img_path": p,
                "face_index": int(idx),
                "box_x": int(x), "box_y": int(y),
                "box_w": int(bw), "box_h": int(bh),
            })

    if not feats:
        return np.zeros((0, 128), dtype=np.float32), [], face_image_set

    feats = np.vstack(feats)
    feats = l2_normalize(feats, axis=1)
    return feats, metas, face_image_set


def save_thumbnail(img_path, box, out_dir, out_name):
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        x, y, w, h = box
        cx, cy = x + w/2, y + h/2
        side = int(max(w, h) * 1.25)
        x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
        x2, y2 = min(W, cx + side//2), min(H, cy + side//2)
        crop = im.crop((x1, y1, x2, y2))
        crop.thumbnail((256, 256))
        ensure_dir(out_dir)
        crop.save(os.path.join(out_dir, out_name))
    except Exception:
        pass


def _unique_dst(path):
    """若文件存在则在扩展名前加短UUID，避免覆盖"""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    return f"{root}__{uuid.uuid4().hex[:6]}{ext}"


def run_face_cluster_once(
    input_dir,
    out_dir,
    model="SFace",
    detector_backend="yunet",
    cluster_method="dbscan",
    dbscan_eps=0.6,
    min_samples=3,
    save_thumbs=True,
    copy_mode="copy",
    rebuild=True,  # True: 清空 albums/thumbnails 后重建（追加照片时建议 True）
):
    """
    - 从 input_dir 读取全部图片，重新聚类并导出到 out_dir。
    - 无人脸的原图，会被复制到所有“有人”的子集。
    - 返回 clusters & name_template 供前端展示/命名。
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")
    image_paths = [str(p) for p in Path(input_dir).rglob("*") if str(p).endswith(exts)]
    image_paths = sorted(image_paths)
    if not image_paths:
        return {"clusters": [], "name_template": []}

    feats, metas, face_image_set = extract_embeddings_sface(image_paths, align=True)

    thumbs_root = os.path.join(out_dir, "thumbnails")
    albums_root = os.path.join(out_dir, "albums")
    if rebuild:
        shutil.rmtree(thumbs_root, ignore_errors=True)
        shutil.rmtree(albums_root, ignore_errors=True)
    ensure_dir(thumbs_root)
    ensure_dir(albums_root)

    if not metas:
        # 没有人脸：创建空的“noise”即可，并把所有图片复制到 noise（或按需到每人，此处只有无人脸）
        noise_dir = os.path.join(albums_root, "noise")
        ensure_dir(noise_dir)
        for img in image_paths:
            dst = _unique_dst(os.path.join(noise_dir, os.path.basename(img)))
            copy_or_link(img, dst, mode=copy_mode)
        return {"clusters": [{"cluster_id": -1, "folder": "noise", "thumb_files": []}],
                "name_template": []}

    # 聚类
    if cluster_method == "hdbscan" and HAS_HDBSCAN:
        labels = hdbscan.HDBSCAN(min_cluster_size=max(2, min_samples), metric="euclidean").fit_predict(feats)
    else:
        labels = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric="euclidean").fit_predict(feats)

    df = pd.DataFrame(metas)
    df["cluster_id"] = labels

    clusters = []
    people_folders = []  # 仅“有人”的文件夹（不含 noise）
    for cid, g in df.groupby("cluster_id"):
        folder = "noise" if cid == -1 else f"person_{int(cid):03d}"
        c_dir = os.path.join(albums_root, folder)
        ensure_dir(c_dir)

        # 把该簇涉及到的原图复制/链接进相册
        uniq_imgs = sorted(set(g["img_path"].tolist()))
        for img in uniq_imgs:
            dst = _unique_dst(os.path.join(c_dir, os.path.basename(img)))
            copy_or_link(img, dst, mode=copy_mode)

        # 缩略图
        thumbs = []
        if save_thumbs:
            tdir = os.path.join(thumbs_root, folder)
            ensure_dir(tdir)
            for k, (_, r) in enumerate(g.head(12).iterrows()):
                img = r["img_path"]
                box = (r["box_x"], r["box_y"], r["box_w"], r["box_h"])
                out_name = f"{k:02d}_{os.path.basename(img)}"
                save_thumbnail(img, box, tdir, out_name)
                thumbs.append(f"{folder}/{out_name}")

        clusters.append({"cluster_id": int(cid), "folder": folder, "thumb_files": thumbs})
        if cid != -1:
            people_folders.append(folder)

    # === 无人脸图片分发：复制到每个“有人”的相册 ===
    no_face_images = sorted(set(image_paths) - set(face_image_set))
    if people_folders and no_face_images:
        for folder in people_folders:
            c_dir = os.path.join(albums_root, folder)
            for img in no_face_images:
                dst = _unique_dst(os.path.join(c_dir, os.path.basename(img)))
                copy_or_link(img, dst, mode=copy_mode)

    # 命名模板
    template = []
    for c in clusters:
        if c["cluster_id"] == -1:
            continue
        template.append({"cluster_id": c["cluster_id"], "person_name": c["folder"]})

    return {"clusters": clusters, "name_template": template}


def _merge_move(src_dir, dst_dir):
    """
    合并文件夹：把 src_dir 的内容并入 dst_dir，文件名冲突自动改名。
    """
    ensure_dir(dst_dir)
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isdir(s):
            _merge_move(s, d)
        else:
            d = d if not os.path.exists(d) else _unique_dst(d)
            shutil.move(s, d)
    shutil.rmtree(src_dir, ignore_errors=True)


def rename_albums_with_map(albums_dir, name_map):
    """
    将 albums/person_xxx 重命名为用户输入的名字；
    - 若目标已存在：**合并**内容（不再生成 _1）
    - 名字为空/空白：跳过
    """
    for cid, name in name_map.items():
        old_name = f"person_{int(cid):03d}"
        src = os.path.join(albums_dir, old_name)
        if not os.path.isdir(src):
            continue
        new_name = name.strip()
        if not new_name:
            continue
        dst = os.path.join(albums_dir, new_name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            _merge_move(src, dst)   # 合并
        else:
            os.rename(src, dst)
