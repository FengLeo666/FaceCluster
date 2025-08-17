# -*- coding: utf-8 -*-
import os, uuid, shutil, json, time
from datetime import datetime
from flask import Blueprint, current_app, render_template, request, redirect, url_for, send_from_directory, flash
from Service.cluster_service import run_face_cluster_once, rename_albums_with_map
from Utils.helpers import allowed_image, unzip_if_zip, secure_filename_keep_ext, write_json_atomic, read_json_safe

from flask import send_file
import tempfile, zipfile

bp = Blueprint("web", __name__, template_folder="../templates", static_folder="../static")

# ---------- 会话列表 ----------
def _list_sessions():
    from Utils.helpers import read_json_safe
    workroot = current_app.config['WORKDIR']
    items = []
    if not os.path.isdir(workroot):
        return items

    for sid in sorted(os.listdir(workroot)):
        sdir = os.path.join(workroot, sid)
        meta_path = os.path.join(sdir, "session.json")
        if not os.path.isfile(meta_path):
            continue

        meta = read_json_safe(meta_path, default={}) or {}
        title = meta.get("title") or sid

        albums_dir = os.path.join(sdir, "output", "albums")
        album_count = 0
        if os.path.isdir(albums_dir):
            album_count = sum(
                1 for d in os.listdir(albums_dir)
                if os.path.isdir(os.path.join(albums_dir, d))
            )

        try:
            created_ts = os.path.getmtime(meta_path)
        except Exception:
            created_ts = 0

        items.append({
            "sid": sid,
            "title": title,
            "created": datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S") if created_ts else "",
            "album_count": album_count,
        })

    items.sort(key=lambda x: x["created"], reverse=True)
    return items


@bp.route("/", methods=["GET"])
def sessions_index():
    # 访问根路径时，先展示会话选择页
    items = _list_sessions()
    return render_template("sessions.html", sessions=items)

@bp.route("/new", methods=["GET"])
def new_session():
    # 上传入口
    return render_template("upload.html")



# ---------- 标注命名 ----------
# Controller/web.py
@bp.route("/label/<sid>", methods=["GET"])
def label_clusters(sid):
    from Utils.helpers import read_json_safe
    workroot = current_app.config['WORKDIR']
    session_dir = os.path.join(workroot, sid)
    meta_path   = os.path.join(session_dir, "session.json")
    if not os.path.isfile(meta_path):
        return "无效会话", 404

    meta = read_json_safe(meta_path, default={}) or {}
    title    = meta.get("title") or sid
    clusters = meta.get("clusters") or []      # ← 不再抛 KeyError
    return render_template("clusters.html", sid=sid, title=title, clusters=clusters)


@bp.route("/apply_names/<sid>", methods=["POST"])
def apply_names(sid):
    workroot = current_app.config['WORKDIR']
    session_dir = os.path.join(workroot, sid)
    meta_path = os.path.join(session_dir, "session.json")
    if not os.path.isfile(meta_path):
        return "无效会话", 404
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    name_map = {}
    for c in meta["clusters"]:
        cid = int(c["cluster_id"])
        key = f"name_{cid}"
        val = request.form.get(key, "").strip()
        if val:
            name_map[cid] = val

    rename_albums_with_map(meta["albums_dir"], name_map)
    flash("已应用命名与文件夹重命名。")
    return redirect(url_for("web.label_clusters", sid=sid))

# ---------- 浏览相册 ----------
@bp.route("/browse/<sid>", methods=["GET"])
def browse_session(sid):
    from Utils.helpers import read_json_safe
    workroot = current_app.config['WORKDIR']
    session_dir = os.path.join(workroot, sid)

    # 读取会话标题（无则回退为 sid）
    meta = read_json_safe(os.path.join(session_dir, "session.json"), default={}) or {}
    title = meta.get("title") or sid

    albums_dir = os.path.join(session_dir, "output", "albums")
    if not os.path.isdir(albums_dir):
        return "该会话暂无相册", 404

    thumbs_dir = os.path.join(session_dir, "output", "thumbnails")
    albums = sorted([d for d in os.listdir(albums_dir) if os.path.isdir(os.path.join(albums_dir, d))])

    album_cards = []
    for alb in albums:
        thumb = None
        td = os.path.join(thumbs_dir, alb)
        if os.path.isdir(td):
            imgs = sorted(os.listdir(td))
            if imgs:
                thumb = f"/thumbs/{sid}/{alb}/{imgs[0]}"
        if not thumb:
            ad = os.path.join(albums_dir, alb)
            imgs = [x for x in sorted(os.listdir(ad)) if x.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
            if imgs:
                thumb = f"/albums/{sid}/{alb}/{imgs[0]}"
        album_cards.append({"name": alb, "thumb": thumb})

    return render_template("browse.html", sid=sid, title=title, albums=album_cards)


@bp.route("/album/<sid>/<path:album>", methods=["GET"])
def view_album(sid, album):
    workroot = current_app.config['WORKDIR']
    a_dir = os.path.join(workroot, sid, "output", "albums", album)
    if not os.path.isdir(a_dir):
        return "相册不存在", 404
    imgs = [x for x in sorted(os.listdir(a_dir)) if x.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    return render_template("album.html", sid=sid, album=album, imgs=imgs)

# ---------- 静态文件（缩略图 / 相册原图） ----------
@bp.route("/thumbs/<sid>/<path:filename>")
def thumbs(sid, filename):
    workroot = current_app.config['WORKDIR']
    thumbs_dir = os.path.join(workroot, sid, "output", "thumbnails")
    return send_from_directory(thumbs_dir, filename, as_attachment=False)

@bp.route("/albums/<sid>/<path:filename>")
def albums_file(sid, filename):
    workroot = current_app.config['WORKDIR']
    albums_dir = os.path.join(workroot, sid, "output", "albums")
    return send_from_directory(albums_dir, filename, as_attachment=False)

@bp.route("/add/<sid>", methods=["GET"])
def add_photos_form(sid):
    return render_template("upload.html", sid=sid)  # 复用上传页模板，带隐藏 sid

@bp.route("/upload", methods=["POST"])
def upload_photos():
    flash("正在处理上传的照片，请稍候…")  # 上传后提示
    sid = request.form.get("sid", "").strip()

    files = request.files.getlist('files')
    if not files:
        flash("没有选择文件")
        return redirect(url_for("web.sessions_index"))

    workroot = current_app.config['WORKDIR']

    # 新建或定位会话
    if sid:
        session_dir = os.path.join(workroot, sid)
        if not os.path.isdir(session_dir):
            flash("会话不存在")
            return redirect(url_for("web.sessions_index"))
    else:
        sid = str(uuid.uuid4())[:8]
        session_dir = os.path.join(workroot, sid)
        os.makedirs(session_dir, exist_ok=True)

    raw_dir = os.path.join(session_dir, "raw")
    out_dir = os.path.join(session_dir, "output")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 保存上传文件
    for f in files:
        if not f or f.filename == "":
            continue
        fname = secure_filename_keep_ext(f.filename)
        save_path = os.path.join(raw_dir, fname)
        f.save(save_path)
        unzip_if_zip(save_path, raw_dir)

    # 重新聚类并重建相册
    result = run_face_cluster_once(
        input_dir=raw_dir,
        out_dir=out_dir,
        dbscan_eps=0.6,
        min_samples=3,
        save_thumbs=True,
        copy_mode="hardlink",
        rebuild=True
    )

    # 更新 session.json
    session_meta = {
        "sid": sid,
        "out_dir": out_dir,
        "albums_dir": os.path.join(out_dir, "albums"),
        "thumbs_dir": os.path.join(out_dir, "thumbnails"),
        "clusters": result["clusters"],
        "name_template": result["name_template"]
    }
    meta_path = os.path.join(session_dir, "session.json")
    write_json_atomic(meta_path, session_meta, ensure_ascii=False, indent=2)

    flash("照片处理完成！")  # 聚类完成后提示
    return redirect(url_for("web.sessions_index"))

@bp.route("/session/<sid>/rename", methods=["POST"])
def rename_session(sid):
    title = request.form.get("title", "").strip()
    workroot = current_app.config['WORKDIR']
    meta_path = os.path.join(workroot, sid, "session.json")
    if not os.path.isfile(meta_path):
        return "无效会话", 404
    meta = read_json_safe(meta_path, default={}) or {}
    meta["title"] = title

    write_json_atomic(meta_path, meta, ensure_ascii=False, indent=2)
    meta["title"] = title
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    flash("已更新会话名称")
    return redirect(url_for("web.sessions_index"))



@bp.route("/download/<sid>/<path:album>", methods=["GET"])
def download_album(sid, album):
    workroot = current_app.config['WORKDIR']
    a_dir = os.path.join(workroot, sid, "output", "albums", album)
    if not os.path.isdir(a_dir):
        return "相册不存在", 404
    # 临时 zip
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{album}.zip")
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(a_dir)):
            p = os.path.join(a_dir, name)
            if os.path.isfile(p):
                zf.write(p, arcname=name)
    return send_file(tmp.name, as_attachment=True, download_name=f"{album}.zip")

@bp.route("/processing/<sid>")
def processing(sid):
    return render_template("processing.html", sid=sid)

@bp.route("/rebuild/<sid>", methods=["POST"])
def rebuild_session(sid):
    from Utils.helpers import read_json_safe, write_json_atomic
    workroot = current_app.config['WORKDIR']
    session_dir = os.path.join(workroot, sid)
    raw_dir = os.path.join(session_dir, "raw")
    out_dir = os.path.join(session_dir, "output")
    meta_path = os.path.join(session_dir, "session.json")

    if not os.path.isdir(raw_dir):
        flash("该会话还没有原始图片！")
        return redirect(url_for("web.sessions_index"))

    result = run_face_cluster_once(
        input_dir=raw_dir,
        out_dir=out_dir,
        dbscan_eps=0.6,
        min_samples=3,
        save_thumbs=True,
        copy_mode="hardlink",
        rebuild=True,             # 清空后重建
    )

    meta = read_json_safe(meta_path, default={}) or {}
    meta.update({
        "sid": sid,
        "out_dir": out_dir,
        "albums_dir": os.path.join(out_dir, "albums"),
        "thumbs_dir": os.path.join(out_dir, "thumbnails"),
        "clusters": result["clusters"],
        "name_template": result["name_template"],
    })
    write_json_atomic(meta_path, meta, ensure_ascii=False, indent=2)
    flash("已重新聚类。")
    return redirect(url_for("web.label_clusters", sid=sid))
