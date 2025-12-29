#
# import os
# import shutil
# import numpy as np
# from pathlib import Path
#
# import pandas as pd
# from PIL import Image
#
# # ---------------------------
# # 小工具
# # ---------------------------
# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)
#
# def l2_normalize(x, axis=1, eps=1e-12):
#     n = np.linalg.norm(x, axis=axis, keepdims=True)
#     return x / np.maximum(n, eps)
#
# def copy_or_link(src, dst, mode="copy"):
#     ensure_dir(os.path.dirname(dst))
#     if mode == "copy":
#         shutil.copy2(src, dst)
#     elif mode == "hardlink":
#         if os.path.exists(dst):
#             os.remove(dst)
#         os.link(src, dst)
#     elif mode == "symlink":
#         if os.path.lexists(dst):
#             os.remove(dst)
#         os.symlink(os.path.abspath(src), dst)
#     else:
#         shutil.copy2(src, dst)
#
#
# def save_thumbnail(img_path, box, out_path, expand=0.25):
#     """
#     根据检测框保存脸部缩略图，box: (x, y, w, h) in original image coordinates
#     """
#     try:
#         im = Image.open(img_path).convert("RGB")
#         W, H = im.size
#         x, y, w, h = box
#         # 扩大一点边界
#         cx, cy = x + w/2, y + h/2
#         side = max(w, h) * (1 + expand)
#         x1 = int(max(0, cx - side/2))
#         y1 = int(max(0, cy - side/2))
#         x2 = int(min(W, cx + side/2))
#         y2 = int(min(H, cy + side/2))
#         face = im.crop((x1, y1, x2, y2))
#         face.thumbnail((256, 256))
#         ensure_dir(os.path.dirname(out_path))
#         face.save(out_path)
#     except Exception as e:
#         pass
#
# def load_name_map(csv_path):
#     """
#     CSV 格式: cluster_id, person_name
#     """
#     mp = {}
#     df = pd.read_csv(csv_path)
#     for _, r in df.iterrows():
#         cid = int(r["cluster_id"])
#         name = str(r["person_name"]).strip()
#         if name and name.lower() != "nan":
#             mp[cid] = name
#     return mp