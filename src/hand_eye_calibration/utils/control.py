import cv2


def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC
    return key == 27