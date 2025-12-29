# # -*- coding: utf-8 -*-
# """
# 基于 DeepFace 的相册按人分发（题目 1）
# Author: <冯基桐>
# Python 3.8+
#
# 依赖:
#   pip install deepface opencv-python-headless scikit-learn pandas tqdm pillow
# 可选:
#   pip install hdbscan
#
# 使用示例:
#   python face_cluster_distributor.py --input_dir ./photos --out_dir ./output \
#     --model ArcFace --detector_backend retinaface --dbscan_eps 0.6 --min_samples 3 \
#     --copy_mode copy --save_thumbs
#
# 二次命名导出（使用你手工填写的名字映射）:
#   python face_cluster_distributor.py --input_dir ./photos --out_dir ./output_named \
#     --model ArcFace --detector_backend retinaface --dbscan_eps 0.6 --min_samples 3 \
#     --copy_mode copy --save_thumbs --name_map ./output/names_template.csv
# """
#
# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path
# import Utils
#
# from sklearn.cluster import DBSCAN
# try:
#     import hdbscan  # optional
#     HAS_HDBSCAN = True
# except Exception:
#     HAS_HDBSCAN = False
#
# from deepface import DeepFace
#
#
#
#
#
# # ---------------------------
# # 主流程
# # ---------------------------
# def extract_embeddings(image_paths, model_name="ArcFace", detector_backend="retinaface",
#                        align=True, enforce_detection=False, prog=True):
#     """
#     用 DeepFace.represent 提取每张图中每张脸的特征向量。
#     返回:
#       feats: (N, D) np.array
#       metas: list(dict)  每个脸的元信息: {img_path, face_index, box(x,y,w,h)}
#     """
#     feats = []
#     metas = []
#
#     for img_path in tqdm(image_paths, desc="Extracting faces", disable=not prog):
#         # DeepFace.represent 可能返回 list[dict] 或 dict（版本差异）
#         try:
#             reps = DeepFace.represent(
#                 img_path=img_path,
#                 model_name=model_name,
#                 detector_backend=detector_backend,
#                 align=align,
#                 enforce_detection=enforce_detection,
#                 # normalization='base'  # 可选
#             )
#         except Exception as e:
#             reps = []
#
#         if isinstance(reps, dict):
#             reps = [reps]
#
#         face_idx = 0
#         for rep in reps:
#             # 常见字段: "embedding", "facial_area"={"x","y","w","h"}
#             emb = rep.get("embedding", None)
#             area = rep.get("facial_area", {})
#             if emb is None:
#                 continue
#             emb = np.array(emb, dtype=np.float32).reshape(1, -1)
#             feats.append(emb)
#             metas.append({
#                 "img_path": img_path,
#                 "face_index": face_idx,
#                 "box_x": int(area.get("x", 0)),
#                 "box_y": int(area.get("y", 0)),
#                 "box_w": int(area.get("w", 0)),
#                 "box_h": int(area.get("h", 0)),
#             })
#             face_idx += 1
#
#     if len(feats) == 0:
#         return np.zeros((0, 512), dtype=np.float32), []
#
#     feats = np.vstack(feats)
#     feats = Utils.l2_normalize(feats, axis=1)
#     return feats, metas
#
# def do_clustering(feats, method="dbscan", eps=0.6, min_samples=3):
#     """
#     返回 labels: shape (N,)
#       -1 表示噪声/未归入任何簇
#     """
#     if len(feats) == 0:
#         return np.array([], dtype=int)
#
#     if method == "hdbscan" and HAS_HDBSCAN:
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_samples),
#                                     metric="euclidean")
#         labels = clusterer.fit_predict(feats)
#     else:
#         clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
#         labels = clusterer.fit_predict(feats)
#     return labels
#
# def export_results(metas, labels, out_dir, copy_mode="copy",
#                    save_thumbs=False, name_map=None):
#     """
#     将**原始照片**按人分发到 albums/<person>/，并保存缩略图到 thumbnails/<person>/。
#     同一张照片若包含多个人，会被复制到多人的文件夹中。
#     """
#     Utils.ensure_dir(out_dir)
#     albums_dir = os.path.join(out_dir, "albums")
#     thumbs_dir = os.path.join(out_dir, "thumbnails")
#     Utils.ensure_dir(albums_dir)
#     if save_thumbs:
#         Utils.ensure_dir(thumbs_dir)
#
#     df = pd.DataFrame(metas)
#     df["cluster_id"] = labels
#
#     # 生成 names_template.csv 供人工命名
#     uniq = sorted([int(x) for x in df["cluster_id"].unique().tolist()])
#     template = []
#     for cid in uniq:
#         if cid == -1:
#             continue
#         person = name_map.get(cid, f"person_{cid:03d}") if name_map else f"person_{cid:03d}"
#         template.append({"cluster_id": cid, "person_name": person})
#     if template:
#         pd.DataFrame(template).to_csv(os.path.join(out_dir, "names_template.csv"),
#                                       index=False, encoding="utf-8-sig")
#
#     # 保存脸级别的聚类结果
#     df.to_csv(os.path.join(out_dir, "clusters.csv"), index=False, encoding="utf-8-sig")
#
#     # 为每个 cluster 收集其涉及的图片
#     grouped = df.groupby("cluster_id")
#     for cid, g in grouped:
#         if cid == -1:
#             person = "noise"
#         else:
#             if name_map and cid in name_map:
#                 person = name_map[cid]
#             else:
#                 person = f"person_{cid:03d}"
#
#         # 该簇的原始图片集合
#         img_set = sorted(set(g["img_path"].tolist()))
#
#         # 分发原图
#         for img in img_set:
#             dst = os.path.join(albums_dir, person, os.path.basename(img))
#             Utils.copy_or_link(img, dst, mode=copy_mode)
#
#         # 缩略图
#         if save_thumbs:
#             # 每个 cluster 挑前若干张脸保存缩略图
#             sub = g.sort_values(["img_path", "face_index"]).head(12)
#             k = 0
#             for _, r in sub.iterrows():
#                 img = r["img_path"]
#                 box = (r["box_x"], r["box_y"], r["box_w"], r["box_h"])
#                 thumb_path = os.path.join(thumbs_dir, person, f"{k:02d}_" + os.path.basename(img))
#                 Utils.save_thumbnail(img, box, thumb_path)
#                 k += 1
#
# def run_face_cluster(
#     input_dir,
#     out_dir,
#     model="ArcFace",
#     detector_backend="retinaface",
#     align=True,
#     cluster_method="dbscan",
#     dbscan_eps=0.6,
#     min_samples=3,
#     copy_mode="copy",
#     save_thumbs=False,
#     name_map_path="",
#     enforce_detection=False,
# ):
#     """
#     输入:
#       input_dir: 包含原始照片的文件夹
#       out_dir: 输出目录
#       model: DeepFace 模型名 (ArcFace, SFace, Facenet, VGG-Face...)
#       detector_backend: retinaface / mtcnn / opencv / yolov8...
#       align: 是否做人脸对齐
#       cluster_method: "dbscan" 或 "hdbscan"
#       dbscan_eps: DBSCAN eps 参数
#       min_samples: DBSCAN/HDBSCAN 的 min_samples
#       copy_mode: "copy" / "hardlink" / "symlink"
#       save_thumbs: 是否保存脸部缩略图
#       name_map_path: 可选，CSV 文件映射 cluster_id→person_name
#       enforce_detection: 是否强制要求检测到人脸
#
#     输出:
#       out_dir 下生成:
#         albums/      按人分发的原始照片
#         thumbnails/  （可选）每人几张脸部缩略图
#         clusters.csv 每张脸的聚类结果
#         names_template.csv 模板供你人工填写人名
#     """
#     # 收集图片
#     exts = (".jpg",".jpeg",".png",".bmp",".webp",".JPG",".JPEG",".PNG",".BMP",".WEBP")
#     image_paths = [str(p) for p in Path(input_dir).rglob("*") if str(p).endswith(exts)]
#     image_paths = sorted(image_paths)
#     if not image_paths:
#         print("未在输入目录找到图片。")
#         return
#
#     # 提取嵌入
#     feats, metas = extract_embeddings(
#         image_paths,
#         model_name=model,
#         detector_backend=detector_backend,
#         align=align,
#         enforce_detection=enforce_detection,
#         prog=True
#     )
#     if len(metas) == 0:
#         print("未检测到任何人脸。")
#         return
#
#     # 聚类
#     labels = do_clustering(
#         feats,
#         method=("hdbscan" if (cluster_method=="hdbscan" and HAS_HDBSCAN) else "dbscan"),
#         eps=dbscan_eps,
#         min_samples=min_samples
#     )
#     print(f"共检测到 {len(metas)} 张人脸；聚类簇数（不含噪声）= {len(set(labels) - {-1})}")
#
#     # 可选姓名映射
#     name_map = {}
#     if name_map_path and os.path.isfile(name_map_path):
#         name_map = Utils.load_name_map(name_map_path)
#
#     # 导出
#     export_results(
#         metas=metas,
#         labels=labels,
#         out_dir=out_dir,
#         copy_mode=copy_mode,
#         save_thumbs=save_thumbs,
#         name_map=name_map
#     )
#     print("完成。请查看输出目录：", out_dir)