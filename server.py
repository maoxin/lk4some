from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging, os
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


def lk4some_by_color(img, H, S, V, mode="HS"):
    app.logger.debug('searching begin')
    cs = ColorSelector(img, H, S, V)
    cs.get_color_mask(mode=mode)
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

    chosen_color = chosen_color.replace("色", "")
    if chosen_color not in color_table['color_list']:
        chosen_color = color_table['translate'][chosen_color]
    else:
        pass

    if chosen_color in color_table['special'].keys():
        H, S, V = color_table['special'][chosen_color]
        mode = "HV"
    elif chosen_color in color_table['normal']:
        H, S, V = color_table['normal'][chosen_color]
        if V < 100:
            mode = "HSV"
        else:
            mode = "HS"

    app.logger.debug(f"color: {chosen_color}, mode: {mode}")
    contours, img_height, img_width = lk4some_by_color(img, H, S, V, mode)
    app.logger.debug(f"{contours}, {img_height}, {img_width}")

    return jsonify({
                   "contours": contours,
                   "img_height": img_height, "img_width": img_width})
