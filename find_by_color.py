from PIL import Image, ImageDraw, ImageFilter
from skimage import measure
import numpy as np
import logging
import yaml
import visdom
import pydensecrf.densecrf as dcrf


fh_debug = logging.FileHandler("debug.log")
fh_debug.setLevel(logging.DEBUG)
fmt_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh_debug.setFormatter(fmt_fh)
logger = logging.getLogger('find_by_color')
logger.setLevel(logging.DEBUG)
logger.addHandler(fh_debug)

with open('color_table.yml') as f:
    color_table = yaml.load(f)


class ColorSelector(object):
    """return contour for chosen color"""

    def __init__(self, img_path, chosen_color_name='红'):
        # load color table, parse chosen color
        with open('color_table.yml', 'r') as f:
            self.color_table = yaml.load(f.read())
        self.chosen_color_name = chosen_color_name.replace("色", "")
        if self.chosen_color_name not in self.color_table['color_list']:
            try:
                self.chosen_color_name = self.color_table['translate'][self.chosen_color_name]
            except KeyError:
                raise Exception(f"Cannot search {self.chosen_color_name}, please try another one")

        # load img, resize it
        self.img_path = img_path
        self.init_img = Image.open(self.img_path).convert(mode="HSV")
        logger.debug(f"image size {self.init_img.size}")
        if self.init_img.height > 128 and self.init_img.width > 128:
            self.img = self.init_img.resize((128, int(128 * self.init_img.height / self.init_img.width)))
        else:
            self.img = self.init_img.copy()
        self.height_ratio = self.init_img.height / self.img.height
        self.width_ratio = self.init_img.width / self.img.width

        # scale H, S, V to standard ones
        self.ar_img = np.array(self.img)
        self.ar_h = self.ar_img[:, :, 0] / 255 * 2 * np.pi
        self.ar_s = self.ar_img[:, :, 1] / 255 * 100
        self.ar_v = self.ar_img[:, :, 2] / 255 * 100

    def get_chosen_color_mask(self):
        try:
            chosen_color_index = np.where(self.color_names == self.chosen_color_name)[0][0]
        except IndexError:
            raise Exception(f"{self.chosen_color_name} cannot be found...")

        chosen_color_mask = np.zeros(self.classified_index_crf.shape, dtype="uint8")
        chosen_color_mask[self.classified_index_crf==chosen_color_index] = 255

        chosen_img_color_mask = Image.fromarray(chosen_color_mask)
        chosen_img_color_mask = self.connect_gap(chosen_img_color_mask)
        chosen_color_mask = np.array(self.remove_noise(chosen_img_color_mask))
        # fill the gap causing by noise, and remove other noise

        self.chosen_color_mask = chosen_color_mask

        return 0

    def classify(self):
        self.classified_index_softmax = self.color_softmax_channels.argmax(0)
        self.classified_index_crf = self.color_crf_channels.argmax(0)

        return 0

    def get_color_prob_channels(self):
        epsilon = 0.1

        self.color_softmax_channels = np.exp(1/(self.color_distance_channels + epsilon))
        self.color_softmax_channels = self.color_softmax_channels / self.color_softmax_channels.sum(0)

        nlabels = len(self.color_table['color_point'])
        d = dcrf.DenseCRF2D(self.img.width, self.img.height, nlabels)
        U = -np.log(self.color_softmax_channels).astype('float32').reshape((nlabels, -1))
        d.setUnaryEnergy(U)
        im = np.array(self.img.convert(mode="RGB"))

        # d.addPairwiseGaussian(sxy=20, compat=3)
        d.addPairwiseBilateral(sxy=8, srgb=3, rgbim=im, compat=10)

        Q = d.inference(20)
        self.color_crf_channels = np.array(Q).reshape((nlabels, self.img.height, self.img.width))
        
        return 0

    def get_color_distance_channels(self):
        # 由于同一个颜色可能对应多个标准颜色，我们只需要比较相对的可能性
        #  所以不用设置什么范围（这样的强行切割可能有害）
        # 可以使用环境饱和度和暗度来矫正，这个作为下一阶段工作
        color_names = []
        color_distance_channels = []
        for color_name, color_detail in self.color_table['color_point'].items():
            color_names.append(color_name)

            mode = color_detail[0]
            H, S, V = color_detail[1:]
            H = H / 360 * 2 * np.pi
            color_distance_channels.append(self.get_color_distance(H, S, V, mode))

        self.color_names = np.array(color_names)
        self.color_distance_channels = np.array(color_distance_channels)

        return 0
    
    def get_color_distance(self, H, S, V, mode):
        if mode == 'HSV':
            x0 = S * np.cos(H)
            y0 = S * np.sin(H)
            z0 = V
            x1 = self.ar_s * np.cos(self.ar_h)
            y1 = self.ar_s * np.sin(self.ar_h)
            z1 = self.ar_v
            ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
            ar_distance = ar_distance * 1.5
        
        elif mode == 'HS':
            x0 = S * np.cos(H)
            y0 = S * np.sin(H)
            x1 = self.ar_s * np.cos(self.ar_h)
            y1 = self.ar_s * np.sin(self.ar_h)
            ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        elif mode == 'V':
            ar_distance = abs(self.ar_v - V)
            ar_distance = ar_distance * 1.5

        else:
            raise Exception(f"Cannot use mode {mode}")

        return ar_distance

    def get_contour(self, min_pts=12):
        # get connected region
        blobs_labels = measure.label(self.chosen_color_mask, background=0)
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

    def connect_gap(self, img, round_max=2, round_min=2):
        for i in range(round_max):
            img = img.filter(ImageFilter.MaxFilter)
        for i in range(round_min):
            img = img.filter(ImageFilter.MinFilter)

        return img

    def remove_noise(self, img, round_max=1, round_min=1):
        for i in range(round_min):
            img = img.filter(ImageFilter.MinFilter)
        for i in range(round_max):
            img = img.filter(ImageFilter.MaxFilter)

        return img


if __name__ == "__main__":
    # img_path = "img/IMG_7270.jpg"
    # img_path = "img/IMG_7237.jpg"
    # img_path = "img/IMG_7271.jpg"
    img_path = "img/现实组/1.jpg"

    cs = ColorSelector(img_path, '蓝')
    cs.get_color_distance_channels()
    cs.get_color_prob_channels()
    cs.classify()
    cs.get_chosen_color_mask()
    cs.get_contour()


    img = cs.img
    color_names = cs.color_names
    color_distance_channels = cs.color_distance_channels
    color_softmax_channels = cs.color_softmax_channels
    color_crf_channels = cs.color_crf_channels
    classified_index_softmax = cs.classified_index_softmax
    classified_index_crf = cs.classified_index_crf
    chosen_color_mask = cs.chosen_color_mask
    contours = cs.contours

    vis = visdom.Visdom()
    if vis.check_connection():
        vis.image(np.transpose(np.array(img.convert(mode="RGB")), [2, 0, 1]), opts=dict(title='Image'),
                win='Image')

        # distance channels
        # for name, channel in zip(color_names, color_distance_channels):
        #     vis.heatmap(channel[::-1, :], opts=dict(title=name), win=name)

        # softmax channnels
        # for name, softmax_channel in zip(color_names, color_softmax_channels):
            # vis.heatmap(softmax_channel[::-1, :], opts=dict(title=name), win=name)

        # crf channels
        # for name, crf_channel in zip(color_names, color_crf_channels):
            # vis.heatmap(crf_channel[::-1, :], opts=dict(title=name), win=name)

        # classified indices
        # vis.heatmap(classified_index_softmax[::-1, :], opts=dict(title='classified_index_softmax'), win='classified_index_softmax')
        # vis.heatmap(classified_index_crf[::-1, :], opts=dict(title='classified_index_crf'), win='classified_index_crf')
        # s = "<!doctype html><html><body><ul>"
        # index = np.arange(0, len(color_names))
        # for n, i in zip(color_names, index):
        #     s += f"<li>{i}: {n}</li>"
        # s += "</ul></body></html>"
        # vis.text(s, win='legend')

        # chosen color mask
        vis.image(chosen_color_mask[None, :, :], opts=dict(title='chosen_color_mask'), win='chosen_color_mask')
            

    # for H in range(360):
        # for S in range(100):
            # cs = ColorSelector(img_path, H, S)
            # cs.get_color_mask()
            # cs.get_contour()
# ----
    # cs.get_color_mask(mode="HS")
    # color_mask = cs.color_mask
    # cs.get_contour()
    # contours = cs.contours

    img = cs.init_img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in contours:
        draw.rectangle(c, outline='red')
    img.show()
