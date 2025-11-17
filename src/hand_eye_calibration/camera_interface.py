from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):
    '''
    An abstract interface for all cameras to use the hand-eye calibration tool.
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def launch(self):
        '''Connects to the camera hardware.'''
        pass

    @abstractmethod
    def get_camera_image(self) -> np.ndarray:
        '''Fetches the latest RGB image.'''
        pass

    @abstractmethod
    def get_camera_intrinsics(self):
        '''Returns the camera's intrinsic parameters.'''
        pass

    @abstractmethod
    def shutdown(self):
        '''Disconnects from the camera.'''
        pass

    @abstractmethod
    def deproject_pixel_to_point(self):
        pass

# ==================================================================
    # def get_camera_image(self):         # realsense
    #     color_image, _ = self.camera.get_images()
    #     return color_image


    # def get_camera_intrinsics(self):         # realsense
    #     intrinsics = self.camera.intrinsics
    #     return intrinsics["matrix"], intrinsics["coeffs"]
