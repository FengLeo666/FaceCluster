
from flask import Flask
from Controller.web import bp as web_bp
import os


# if __name__ == "__main__":
#     Service.run_face_cluster(
#         input_dir="./photos",
#         out_dir="./output",
#         model="ArcFace",
#         detector_backend="retinaface",
#         dbscan_eps=0.6,
#         min_samples=3,
#         copy_mode="copy",
#         save_thumbs=True
#     )


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret'
    app.config['WORKDIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
    os.makedirs(app.config['WORKDIR'], exist_ok=True)
    app.register_blueprint(web_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)