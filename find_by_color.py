from PIL import Image, ImageDraw
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

    def __init__(self, img_path, chosen_h, chosen_s):
        self.img_path = img_path
        self.chosen_h = chosen_h / 360 * 2 * np.pi
        self.img0 = Image.open(self.img_path).convert(mode="HSV")
        logger.debug(f"image size {self.img0.size}")

        self.img = self.img0.resize((128, int(128 * self.img0.height / self.img0.width)))
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

    def get_color_mask(self, s_ratio=0.7):
        x0 = self.chosen_s * np.cos(self.chosen_h)
        y0 = self.chosen_s * np.sin(self.chosen_h)
        x1 = self.ar_s * np.cos(self.ar_h)
        y1 = self.ar_s * np.sin(self.ar_h)
        ar_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        ar_distance = ar_distance - ar_distance.min()
        ar_distance = ar_distance.astype("uint8")

        color_mask = np.zeros(ar_distance.shape, dtype="uint8")
        # color_mask[ar_distance <= 8] = 255
        # 人眼能观察的颜色认为有150种，r = R / 150^0.5 = 100 * 0.08 = 8
        color_mask[ar_distance <= 16] = 255
        # 认为是36种提高容错率。。。。

        self.ar_distance = ar_distance
        self.color_mask = color_mask

        return 0

    def get_contour(self):
        # use AGNES
        row, column = np.where(self.color_mask == 255)
        data_set = np.vstack((row, column)).T

        ag = AGNES(gap_ratio=4.24, min_pts=9)
        ag.fit(data_set)

        contours = []
        for C in ag.Cs:
            up = int(C[:, 0].min() * self.height_ratio)
            down = int(C[:, 0].max() * self.height_ratio)
            left = int(C[:, 1].min() * self.width_ratio)
            right = int(C[:, 1].max() * self.width_ratio)

            contours.append([(left, up), (right, down)])

        self.contours = contours
        self.ag = ag

        return 0


class AGNES(object):
    def __init__(self, gap_ratio, min_pts):
        self.gap_ratio = gap_ratio
        self.min_pts = min_pts

    def fit(self, X):
        # X is a (N, 2) array
        # a C is a (n, 2) array, Cs is the list of C
        Cs = list(X.reshape((X.shape[0], 1, 2)))

        x = X[:, 0]
        y = X[:, 1]
        M = np.sqrt((x[:, np.newaxis] - x)**2 + (y[:, np.newaxis] - y)**2)
        np.fill_diagonal(M, np.inf)

        min_distance = M.min()
        min_distance_old = min_distance

        while (min_distance / min_distance_old <= self.gap_ratio) and (M.shape[0] >= 2):
            ar_indices_star = np.vstack(np.where(M == M.min())).T[0]
            i_star = min(ar_indices_star)
            j_star = max(ar_indices_star)

            Cs[i_star] = np.vstack((Cs[i_star], Cs[j_star]))
            Cs.pop(j_star)

            M[i_star] = np.vstack((M[i_star], M[j_star])).min(axis=0)
            M[:, i_star] = M[i_star]
            M[i_star, i_star] = np.inf

            M = np.delete(M, j_star, axis=0)
            M = np.delete(M, j_star, axis=1)

            min_distance = M.min()

        Cs = sorted(Cs, key=lambda x: x.shape[0], reverse=True)

        while Cs:
            if Cs[-1].shape[0] < self.min_pts:
                Cs.pop(-1)
            else:
                break

        self.Cs = Cs

        return 0

    def min_distance_sets(self, C0, C1):
        x_center0, y_center0 = C0.mean(axis=0)
        x_center1, y_center1 = C1.mean(axis=0)
        v_length = np.sqrt((x_center1 - x_center0)**2 + (y_center1 - y_center0)**2)
        cos_theta = (x_center1 - x_center0) / v_length
        sin_theta = (y_center1 - y_center0) / v_length

        C0_new = self.rotate_axis(C0, cos_theta, sin_theta)
        C1_new = self.rotate_axis(C1, cos_theta, sin_theta)

        min_distance = np.sqrt(((C1_new[C1_new.argmin(axis=0)[0]] - C0_new[C0_new.argmax(axis=0)[0]])**2).sum())

        return min_distance

    def rotate_axis(self, C, cos_theta, sin_theta):
        C_new = np.zeros(C.shape)
        C_new[:, 0] = C[:, 0] * cos_theta + C[:, 1] * sin_theta
        C_new[:, 1] = -C[:, 0] * sin_theta + C[:, 1] * cos_theta

        return C_new


if __name__ == "__main__":
    img_path = "img/IMG_7270.jpg"
    cs = ColorSelector(img_path, 8, 63)
    # cs = ColorSelector(img_path, 0, 80)

    # for H in range(360):
        # for S in range(100):
            # cs = ColorSelector(img_path, H, S)
            # cs.get_color_mask()
            # cs.get_contour()

    cs.get_color_mask()
    cs.get_contour()
    ag = cs.ag
    contours = cs.contours

    img = cs.img0.convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in contours:
        draw.rectangle(c, outline='red')
