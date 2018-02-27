from PIL import Image, ImageDraw, ImageFilter
from skimage import measure
import numpy as np
import logging


fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
logger = logging.getLogger('find_by_color')
logger.setLevel(logging.DEBUG)
logger.addHandler(fh_debug)


class ColorSelector(object):
    """return contour for chosen color"""

    def __init__(self, img_path, chosen_h, chosen_s, chosen_v=100):
        self.img_path = img_path
        self.chosen_h = chosen_h / 360 * 2 * np.pi
        self.chosen_v = chosen_v
        self.img0 = Image.open(self.img_path).convert(mode="HSV")
        logger.debug(f"image size {self.img0.size}")

        if self.img0.height > 128 and self.img0.width > 128:
            self.img = self.img0.resize((128, int(128 * self.img0.height / self.img0.width)))
        else:
            self.img = self.img0.copy()
        self.height_ratio = self.img0.height / self.img.height
        self.width_ratio = self.img0.width / self.img.width
        self.ar_img = np.array(self.img)
        self.ar_h = self.ar_img[:, :, 0] / 255 * 2 * np.pi
        self.ar_s = self.ar_img[:, :, 1] / 255 * 100
        self.ar_v = self.ar_img[:, :, 2] / 255 * 100

        # self.chosen_s = self.ar_v.copy()
        # self.chosen_s[self.ar_v > chosen_s] = chosen_s
        # S cannot be greater than V (inverted cone space of HSV), so I use the method above.method
        # But the reality is that S can be greater tan V according to the wiki. So I give up
        self.chosen_s = chosen_s

    def get_color_mask(self, mode="HS"):
        if mode == "H":
            ar_distance = self.distance_H()
            threshold = 2 * np.pi / 36
        elif mode == "HS":
            ar_distance = self.distance_HS()
            # threshold = 8
            # 人眼能观察的颜色认为有150种，r = R / 150^0.5 = 100 * 0.08 = 8
            threshold = 16
            # 认为是36种提高容错率。。。。
        elif mode == "HSV":
            ar_distance = self.distance_HS()
            # threshold = 28
            # d = 100 * 3∫ (π / 150) = 28
            threshold = 45
            # d = 100 * 3∫ (π / 36) = 45
        else:
            return 1

        color_mask = np.zeros(ar_distance.shape, dtype="uint8")
        color_mask[ar_distance <= threshold] = 255

        img_color_mask = Image.fromarray(color_mask)
        color_mask = np.array(self.connect_gap(img_color_mask))
        # fill the gap causing by noise

        self.ar_distance = ar_distance
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

    def get_contour(self, gap_ratio=4.24, min_pts=9):
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
    # img_path = "img/IMG_7270.jpg"
    img_path = "img/IMG_7237.jpg"
    # img_path = "img/IMG_7271.jpg"
    # cs = ColorSelector(img_path, 8, 63)
    cs = ColorSelector(img_path, 0, 80)

    # for H in range(360):
        # for S in range(100):
            # cs = ColorSelector(img_path, H, S)
            # cs.get_color_mask()
            # cs.get_contour()

    cs.get_color_mask(mode="HS")
    color_mask = cs.color_mask
    cs.get_contour()
    contours = cs.contours

    img = cs.img0.convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in contours:
        draw.rectangle(c, outline='red')
