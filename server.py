from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging, os
from find_by_color import ColorSelector


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
app.logger.addHandler(fh_debug)


def lk4some_by_color(img, H, S):
    app.logger.debug('searching begin')
    cs = ColorSelector(img, H, S)
    cs.get_color_mask()
    cs.get_contour()
    contours = cs.contours
    img_height = cs.img0.height
    img_width = cs.img0.width
    app.logger.debug('searching end')

    return contours, img_height, img_width


@app.route('/')
def index():
    app.logger.debug("index")
    return "Index Page"


@app.route('/upload', methods=['POST'])
def upload_img():
    img = request.files['search_area']
    chosen_color = request.form['color']
    if chosen_color == 'red':
        H = 0
        S = 80
        # H = 8
        # S = 63
    contours, img_height, img_width = lk4some_by_color(img, H, S)
    app.logger.debug(f"{contours}, {img_height}, {img_width}")

    return jsonify({
                   "contours": contours,
                   "img_height": img_height, "img_width": img_width})
