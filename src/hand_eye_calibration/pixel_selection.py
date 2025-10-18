import cv2
from .utils import quit_keypress


class PixelSelector:
    def __init__(self, win='pixel_selector'):
        self.win = win
        self.clicks = []

    def load_image(self, img, recrop=False, x=700, y=300, width=400, height=300, out_size=(640, 480)):
        if recrop:
            img = img[y:y+height, x:x+width]
        if out_size:
            img = cv2.resize(img, out_size)
        self.img = img
        self.vis = img.copy()  # drawing surface

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            cv2.circle(self.vis, (x, y), 3, (255, 255, 0), -1)

    def run(self, img, **load_kwargs):
        self.load_image(img, **load_kwargs)
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.mouse_callback)

        while not quit_keypress():
            cv2.imshow(self.win, self.vis)

        cv2.destroyWindow(self.win)
        return self.clicks
