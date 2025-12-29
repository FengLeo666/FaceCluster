# -*- coding: utf-8 -*-
import os, zipfile,tempfile,json
from pathlib import Path
from werkzeug.utils import secure_filename

ALLOWED_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".zip",".JPG",".JPEG",".PNG",".BMP",".WEBP"}

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def allowed_image(filename):
    return any(filename.endswith(ext) for ext in ALLOWED_EXTS)

def secure_filename_keep_ext(filename):
    ext = os.path.splitext(filename)[1]
    base = os.path.splitext(filename)[0]
    return secure_filename(base) + ext

def unzip_if_zip(path, out_dir):
    if not path.lower().endswith(".zip"):
        return
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(out_dir)
        os.remove(path)
    except Exception:
        pass

def copy_or_link(src, dst, mode="copy"):
    ensure_dir(os.path.dirname(dst))
    try:
        if mode == "hardlink":
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        elif mode == "symlink":
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        else:
            import shutil
            shutil.copy2(src, dst)
    except Exception:
        # 退化为复制
        import shutil
        shutil.copy2(src, dst)


def write_json_atomic(path, data, ensure_ascii=False, indent=2, encoding="utf-8"):
    """
    原子写：先写到同目录的临时文件，再 os.replace 覆盖正式文件，避免读到半成品。
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # 原子替换（Windows / POSIX 都安全）
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def read_json_safe(path, default=None, encoding="utf-8"):
    """
    容错读：文件不存在 / 空文件 / JSON 格式错误时，返回 default。
    """
    if not os.path.isfile(path):
        return default
    try:
        if os.path.getsize(path) == 0:
            return default
    except Exception:
        return default
    try:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
    except Exception:
        return default