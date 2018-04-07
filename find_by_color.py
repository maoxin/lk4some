from PIL import Image, ImageDraw, ImageFilter
from skimage import measure
import numpy as np
import logging
import os
from yaml import load


fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
logger = logging.getLogger('find_by_color')
logger.setLevel(logging.DEBUG)
logger.addHandler(fh_debug)

with open('color_table.yml') as f:
    color_table = load(f)


class ColorSelector(object):
    """return contour for chosen color"""

    def __init__(self, img_path, chosen_h_range, chosen_s_range, chosen_v_range):
        self.img_path = img_path
        if chosen_h_range:
            self.chosen_h_upper = chosen_h_range[1] / 360 * 2 * np.pi
            self.chosen_h_lower = chosen_h_range[0] / 360 * 2 * np.pi
        if chosen_v_range:
            self.chosen_v_upper = chosen_v_range[1]
            self.chosen_v_lower = chosen_v_range[0]
        if chosen_s_range:
            self.chosen_s_upper = chosen_s_range[1]
            self.chosen_s_lower = chosen_s_range[0]

        self.img0 = Image.open(self.img_path).convert(mode="HSV")
        logger.debug(f"image size {self.img0.size}")

        if self.img0.height > 128 and self.img0.width > 128:
            self.img = self.img0.resize((128, int(128 * self.img0.height / self.img0.width)))
            # self.img = self.img0.resize((256, int(256 * self.img0.height / self.img0.width)))
            # self.img = self.img0.copy()
        else:
            self.img = self.img0.copy()
        self.height_ratio = self.img0.height / self.img.height
        self.width_ratio = self.img0.width / self.img.width
        self.ar_img = np.array(self.img)
        self.ar_h = self.ar_img[:, :, 0] / 255 * 2 * np.pi
        self.ar_s = self.ar_img[:, :, 1] / 255 * 100
        self.ar_v = self.ar_img[:, :, 2] / 255 * 100

    def get_color_mask(self, mode="HS"):
        if mode == "normal":
            if self.chosen_h_lower > self.chosen_h_upper:
                h_mask = (self.ar_h <= self.chosen_h_upper) | (self.ar_h >= self.chosen_h_lower)
            else:
                h_mask = (self.ar_h <= self.chosen_h_upper) & (self.ar_h >= self.chosen_h_lower)
            s_mask = (self.ar_s <= self.chosen_s_upper) & (self.ar_s >= self.chosen_s_lower)
            v_mask = (self.ar_v <= self.chosen_v_upper) & (self.ar_s >= self.chosen_v_lower)

            color_mask = (h_mask & s_mask & v_mask).astype('uint8')

        elif mode == "black":
            v_mask = (self.ar_v <= self.chosen_v_upper) & (self.ar_v >= self.chosen_v_lower)

            color_mask = v_mask.astype('uint8')

        elif mode == "white_or_grey":
            v_mask = (self.ar_v <= self.chosen_v_upper) & (self.ar_v >= self.chosen_v_lower)
            s_mask = (self.ar_s <= self.chosen_s_upper) & (self.ar_s >= self.chosen_s_lower)

            color_mask = (v_mask & s_mask).astype('uint8')

        else:
            return 1

        # img_color_mask = Image.fromarray(color_mask)
        # color_mask = np.array(self.connect_gap(img_color_mask))
        # fill the gap causing by noise

        self.color_mask = color_mask

        return 0

    def distance_H(self):
        ar_distance = abs(self.ar_h - self.chosen_h)
        ar_distance[ar_distance > np.pi] = 2 * np.pi - ar_distance[ar_distance > np.pi]

        ar_distance = ar_distance - ar_distance.min()

        return ar_distance

    def distance_HS(self):
        x0 = self.chosen_s * np.cos(self.chosen_h)
        y0 = self.chosen_s * np.sin(self.chosen_h)
        x1 = self.ar_s * np.cos(self.ar_h)
        y1 = self.ar_s * np.sin(self.ar_h)
        ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        ar_distance = ar_distance - ar_distance.min()

        return ar_distance

    def distance_HSV(self):
        x0 = self.chosen_s * np.cos(self.chosen_h)
        y0 = self.chosen_s * np.sin(self.chosen_h)
        z0 = self.chosen_v
        x1 = self.ar_s * np.cos(self.ar_h)
        y1 = self.ar_s * np.sin(self.ar_h)
        z1 = self.ar_v
        ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        ar_distance = ar_distance - ar_distance.min()

        return ar_distance

    def distance_HV(self):
        x0 = self.chosen_s
        x1 = self.ar_s
        y0 = self.chosen_v
        y1 = self.ar_v
        ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        ar_distance = ar_distance - ar_distance.min()

        return ar_distance

    def distance_V(self):
        ar_distance = abs(self.ar_v - self.chosen_v)

        ar_distance = ar_distance - ar_distance.min()

        return ar_distance

    def get_contour(self, min_pts=12):
        # get connected region
        blobs_labels = measure.label(self.color_mask, background=0)
        Cs = []
        for label in np.unique(blobs_labels):
            if label != 0:
                row, column = np.where(blobs_labels == label)
                if len(row) >= min_pts:
                    Cs.append(np.vstack((row, column)).T)

        self.Cs = Cs

        contours = []
        for C in self.Cs:
            up = int(C[:, 0].min() * self.height_ratio)
            down = int(C[:, 0].max() * self.height_ratio)
            left = int(C[:, 1].min() * self.width_ratio)
            right = int(C[:, 1].max() * self.width_ratio)

            contours.append([(left, up), (right, down)])

        self.contours = contours

        return 0

    def connect_gap(self, img, round_max=1, round_min=1):
        for i in range(round_max):
            img = img.filter(ImageFilter.MaxFilter)
        for i in range(round_min):
            img = img.filter(ImageFilter.MinFilter)

        return img


if __name__ == "__main__":
    def draw_rec(chosen_color, img_path):
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

        cs = ColorSelector(img_path, H_range, S_range, V_range)
        cs.get_color_mask(mode=mode)
        cs.get_contour()
        contours = cs.contours

        img = cs.img0.convert("RGB")
        draw = ImageDraw.Draw(img)
        for c in contours:
            draw.rectangle(c, outline='red')

        return img

    real_img_fn = os.listdir("img/现实组")
    art_img_fn = os.listdir("img/艺术组")
    colors = color_table['color_list']

    for fn in real_img_fn:
        for cl in colors:
            img = draw_rec(cl, f'img/现实组/{fn}')
            img.save(f"img/结果/现实组/{cl}_{fn}")

    for fn in art_img_fn:
        for cl in colors:
            img = draw_rec(cl, f'img/艺术组/{fn}')
            img.save(f"img/结果/艺术组/{cl}_{fn}")
