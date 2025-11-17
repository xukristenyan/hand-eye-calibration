from abc import ABC, abstractmethod
from .camera_interface import BaseCamera
from zed_toolbox import ZedCamera

class Zed(BaseCamera):
    def __init__(self, serial, config):
        self.serial = serial
        self.config = config
        self.camera = ZedCamera(self.serial, self.config)

    def launch(self):
        self.camera.launch()

    def get_camera_image(self):
        color_image, _ = self.camera.get_rgbd()
        return color_image

    def get_camera_intrinsics(self):
        K, dist = self.camera.get_intrinsics()
        return K, dist

    def shutdown(self):
        self.camera.shutdown()

    def deproject_pixel_to_point(self, pixel):
        point = self.camera.deproject_to_3d(pixel)
        return point