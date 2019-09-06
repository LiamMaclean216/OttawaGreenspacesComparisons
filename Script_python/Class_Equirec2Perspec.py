# -*- coding: utf-8 -*-
"""
Definition file of the Equirectangular class.

This class is used to project equirectangular panorama into perspective images

This class has been taken from this Github project : https://github.com/fuenwang/Equirec2Perspec
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Class definition
# ----------------------------------------------------------------------------------------------------------------------
class Equirectangular:
    """
    Definition of Equirectangular class.

    This class is used to project equirectangular panorama into perspective images
    """
    def __init__(self, img_name):
        self.img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self.img.shape

    def get_perspective(self, fov, theta, phi, height, width, radius=128):
        """
        Project equirectangular panorama into a perspective image in the direction described by the parameters.

        :param fov: field of view angle in degree
        :type fov: float
        :param theta: horizontal angle in degree
        :type theta: float
        :param phi:  vertical angle in degree
        :type phi: float
        :param height: output perspective image height
        :type height: int
        :param width: output perspective image width
        :type width: int
        :param radius:
        :type radius:
        :return: perspective array
        :rtype: np.array
        """

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        w_fov = fov
        h_fov = float(height) / width * w_fov

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - w_fov) / 2.0
        w_len = 2 * radius * np.sin(np.radians(w_fov / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - h_fov) / 2.0
        h_len = 2 * radius * np.sin(np.radians(h_fov / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + radius
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        dist = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (radius / dist * x_map)[:, :]
        xyz[:, :, 1] = (radius / dist * y_map)[:, :]
        xyz[:, :, 2] = (radius / dist * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [rot_1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
        [rot_2, _] = cv2.Rodrigues(np.dot(rot_1, y_axis) * np.radians(-phi))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(rot_1, xyz)
        xyz = np.dot(rot_2, xyz).T
        lat = np.arcsin(xyz[:, 2] / radius)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        perspective = cv2.remap(self.img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return perspective
