from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging, os, time
from yaml import load
from find_by_color import ColorSelector


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
app.logger.addHandler(fh_debug)

with open('color_table.yml') as f:
    color_table = load(f)


def lk4some_by_color(img, H_range, S_range, V_range, mode="HS"):
    app.logger.debug('searching begin')
    t0 = time.time()

    cs = ColorSelector(img, H_range, S_range, V_range)
    cs.get_color_mask(mode=mode)
    cs.get_contour()
    contours = cs.contours
    img_height = cs.img0.height
    img_width = cs.img0.width

    t1 = time.time()
    app.logger.debug(f'searching end, cost {t1 - t0}s')

    return contours, img_height, img_width


@app.route('/')
def index():
    app.logger.debug("index")
    return "Index Page"


@app.route('/upload', methods=['POST'])
def upload_img():
    img = request.files['search_area']
    chosen_color = request.form['color']

    chosen_color = chosen_color.replace("色", "")
    if chosen_color not in color_table['color_list']:
        chosen_color = color_table['translate'][chosen_color]
    else:
        pass

    H_range = color_table['color'][chosen_color].get("H")
    S_range = color_table['color'][chosen_color].get("S")
    V_range = color_table['color'][chosen_color].get("V")

    if chosen_color in color_table['special_color_list']:
        if chosen_color == "黑":
            mode = "black"
        elif chosen_color in ["白", "灰"]:
            mode = "white_or_grey"
    elif chosen_color in color_table['normal_color_list']:
        mode = "normal"

    app.logger.debug(f"color: {chosen_color}, mode: {mode}")
    contours, img_height, img_width = lk4some_by_color(img, H_range, S_range, V_range, mode)
    app.logger.debug(f"{contours}, {img_height}, {img_width}")

    return jsonify({
                   "contours": contours,
                   "img_height": img_height, "img_width": img_width})
