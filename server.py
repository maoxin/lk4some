from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging, os, time
from yaml import load
from find_by_color import ColorSelector
import json


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
app.logger.addHandler(fh_debug)

with open('color_table.yml') as f:
    color_table = load(f)


def lk4some_by_color(img_path, chosen_color_name):
    app.logger.debug('searching begin')
    t0 = time.time()

    cs = ColorSelector(img_path, chosen_color_name)
    cs.get_color_distance_channels()
    cs.get_color_prob_channels()
    cs.classify()
    cs.get_chosen_color_mask()
    cs.get_contour()
    contours = cs.contours
    img_height = cs.init_img.height
    img_width = cs.init_img.width

    t1 = time.time()
    app.logger.debug(f'searching end, cost {t1 - t0}s')

    return contours, img_height, img_width


@app.route('/')
def index():
    app.logger.debug("index")
    return "Index Page"


@app.route('/upload', methods=['POST'])
def upload_img():
    img_path = request.files['search_area']
    
    chosen_color_name = request.form['color']

    app.logger.debug(f"color_name: {chosen_color_name}")
    contours, img_height, img_width = lk4some_by_color(img_path, chosen_color_name)
    app.logger.debug(f"{contours}, {img_height}, {img_width}")

    return jsonify({
                   "contours": contours,
                   "img_height": img_height, "img_width": img_width})
