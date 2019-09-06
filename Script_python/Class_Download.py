# -*- coding: utf-8 -*-
"""
Definition file of the Download class.
This class is used to :
    - download the images from the following sources : Google Street View, Mapillary
    - create simple images from Mapillary panoramas
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import time

import progressbar
import streetview
from osgeo import ogr
from pymapillary import Mapillary
from pymapillary.utils import download_image_by_key

from Class_Image import Panorama
from utils_class import timeit, safe_folder_creation


# ----------------------------------------------------------------------------------------------------------------------
# Class definition
# ----------------------------------------------------------------------------------------------------------------------
class Download(object):
    """
    Download class definition.

    This class is used to :
        - download images from Mapillary or Google Street View at locations described in a shapefile.
        - create simple images from Mapillary panoramas

    WARNING : Google Street View imagery might not be usable for research applications.
    Check Google policies at https://www.google.com/permissions/geoguidelines/.
    """
    def __init__(self, main_dir, layer_path):
        self._main_dir = main_dir
        self.layer_path = layer_path

        self.gsv_dir = os.path.join(main_dir, "GSV_images")
        self.mapillary_dir = os.path.join(main_dir, "Mapillary_images")
        self.mapillary_dir_pano = os.path.join(main_dir, "Mapillary_panoramas")
        self.mapillary_dir_images_from_pano = os.path.join(main_dir, "Mapillary_images_from_panoramas")

        self.nb_mapillary_jpg = 0
        self.nb_mapillary_panos = 0
        self.nb_gsv = 0
        self.nb_mapillary_from_pano = 0

    def __str__(self):
        string = "=== Download information === "
        string += "\nDownload directory path : " + self.layer_path
        string += "\nLayer path : " + self.layer_path
        string += "\nNumber of Mapillary images downloaded : " + str(self.nb_mapillary_jpg)
        string += "\nNumber of Mapillary panoramas downloaded : " + str(self.nb_mapillary_panos)
        string += "\nNumber of images produced from Mapillary panoramas : " + str(self.nb_mapillary_from_pano)
        string += "\nNumber of GSV images downloaded : " + str(self.nb_gsv)

        return string

    @timeit
    def download_gsv(self, gsv_key, start=0, end=-1):
        """
        Downloads Google Street View (GSV) images points in Ottawa from a shapefile layer.

        WARNING: the filename format is currently not compatible with the class Image definition.

        :param gsv_key: GSV API key
        :type gsv_key: str
        :param start: index of the feature of the layer from where download starts
        :type start: int
        :param end: index of the feature of the layer from where download ends
        :type end: int 
        """

        # Creation of directory
        self.gsv_dir = safe_folder_creation(self.gsv_dir)

        #  Convert layer file
        ds = ogr.Open(self.layer_path)
        layer = ds.GetLayer()

        # Determine the number of locations to download
        loc_max = len(layer)
        if start < end < len(layer):
            stop = end
        else:
            stop = loc_max
        n_loc = stop - start

        # Display advancement of downloading
        pbar = progressbar.ProgressBar()
        for i in pbar(range(start, stop)):
            # Get location
            feature = layer[i]
            lon = feature.GetGeometryRef().GetX()
            lat = feature.GetGeometryRef().GetY()

            # Get the closest panoramas from the location
            pano_id = streetview.panoids(lat, lon, closest=True)

            # Check if there is a pano
            if len(pano_id):
                # Create filename
                image_key = pano_id[0]["panoid"]
                if pano_id[0]["month"] < 10:
                    image_date = str(pano_id[0]["year"]) + "-0" + str(pano_id[0]["month"]) + "-01T00-00-00"
                else:
                    image_date = str(pano_id[0]["year"]) + "-" + str(pano_id[0]["month"]) + "-01T00-00-00"
                image_lon = "{0:.6f}".format(lon)
                image_lat = "{0:.6f}".format(lat)
                image_filename = '{}_{}_{}_{}'.format(image_lon, image_lat, image_key, image_date)
                # Download one image
                try:
                    streetview.api_download(image_key, 90, self.gsv_dir, gsv_key, fov=80, pitch=0,
                                            fname=image_filename)
                    self.nb_gsv += 1
                except Exception as err:
                    print(err)
                    print("Error on feature {}, lat = {}, lon = {} ".format(i, lat, lon))
                    continue

        # Display information
        print("Number of locations         : {}".format(n_loc))
        print("Number of images downloaded : {}".format(self.nb_gsv))
        print("Ratio : {}%".format((self.nb_gsv / n_loc) * 100))

    @timeit
    def download_mapillary(self, mapillary_key, start=0, end=-1):
        """
        Downloads Mapillary images of points from a ArcGis layer

        :param mapillary_key: Mapillary API key
        :type mapillary_key: str
        :param start: index of the feature of the layer from where download starts
        :type start: int
        :param end: index of the feature of the layer where download ends
        :type end: int
        """
        # Initialization of variables
        self.nb_mapillary_jpg = 0
        self.nb_mapillary_panos = 0
        cmp_err = 0
        err_list = []

        # Creation of directory
        self.mapillary_dir = safe_folder_creation(self.mapillary_dir)
        self.mapillary_dir_pano = safe_folder_creation(self.mapillary_dir_pano)

        # Convert layer file
        ds = ogr.Open(self.layer_path)
        layer = ds.GetLayer()

        # Determine the number of locations to download
        loc_max = len(layer)
        if start < end < len(layer):
            stop = end
        else:
            stop = loc_max
        n_loc = stop - start

        # Create a Mapillary Object
        mapillary = Mapillary(mapillary_key)

        # Display advancement of downloading
        pbar = progressbar.ProgressBar()
        for i in pbar(range(start, stop)):
            # Get location
            feature = layer[i]
            lon = feature.GetGeometryRef().GetX()
            lat = feature.GetGeometryRef().GetY()
            close_to = "{},{}".format(lon, lat)

            # Catch metadata search errors
            try:
                raw_json = mapillary.search_images(closeto=close_to,
                                                   per_page=1,
                                                   radius=10)
                raw_json_pano = mapillary.search_images(closeto=close_to,
                                                        per_page=1,
                                                        radius=10,
                                                        pano="true")
            except Exception as err:
                print(err)
                print("Error during metadata search on feature {}, lat = {}, lon = {} ".format(i, lat, lon))
                cmp_err += 1
                err_list.append(i)
                continue

            # Check if there is a image at this location and download it
            try:
                if len(raw_json['features']):
                    image_key = raw_json['features'][0]['properties']['key']
                    image_date = raw_json['features'][0]['properties']['captured_at'][:-5]
                    image_date = image_date.replace(":", "-")
                    image_lon = raw_json['features'][0]['geometry']['coordinates'][0]
                    image_lat = raw_json['features'][0]['geometry']['coordinates'][1]
                    image_lon = "{0:.6f}".format(image_lon)
                    image_lat = "{0:.6f}".format(image_lat)

                    image_filename = '{}_{}_{}_000_{}.jpg'.format(image_lon, image_lat, image_key, image_date)
                    image_path = os.path.join(self.mapillary_dir, image_filename)
                    download_image_by_key(image_key, 640, image_path)
                    self.nb_mapillary_jpg += 1
            except Exception as err:
                print(err)
                print("Error when downloading image on feature {}, lat = {}, lon = {} ".format(i, lat, lon))
                cmp_err += 1
                err_list.append(i)
                continue

            # Check if there is a pano at this location and download it
            try:
                if len(raw_json_pano['features']):
                    pano_key = raw_json_pano['features'][0]['properties']['key']
                    pano_date = raw_json_pano['features'][0]['properties']['captured_at'][:-5]
                    pano_date = pano_date.replace(":", "-")
                    pano_lon = raw_json_pano['features'][0]['geometry']['coordinates'][0]
                    pano_lat = raw_json_pano['features'][0]['geometry']['coordinates'][1]
                    pano_lon = "{0:.6f}".format(pano_lon)
                    pano_lat = "{0:.6f}".format(pano_lat)

                    pano_filename = '{}_{}_{}_999_{}.jpg'.format(pano_lon, pano_lat, pano_key, pano_date)
                    pano_path = os.path.join(self.mapillary_dir_pano, pano_filename)
                    download_image_by_key(pano_key, 2048, pano_path)
                    self.nb_mapillary_panos += 1
            except Exception as err:
                print(err)
                print("Error when downloading panorama on feature {}, lat = {}, lon = {} ".format(i, lat, lon))
                cmp_err += 1
                err_list.append(i)
                continue

        # Display information
        print("Number of locations            : {}".format(n_loc))
        print("Number of images downloaded    : {}".format(self.nb_mapillary_jpg))
        print("Number of panoramas downloaded : {}".format(self.nb_mapillary_panos))
        print("Number of errors caught : {}".format(cmp_err))
        print("List of features index with errors : {}".format(err_list))

    @timeit
    def create_images_from_mapillary_panos(self):
        """
        Create for 4 simple images for all the downloaded Mapillary panoramas.

        """
        # Creation of directory
        self.mapillary_dir_images_from_pano = safe_folder_creation(self.mapillary_dir_images_from_pano)

        # Display advancement of processing
        pbar = progressbar.ProgressBar()
        panos = os.listdir(self.mapillary_dir_pano)
        for pano in pbar(panos):
            # Create instance of panorama and its 4 images
            p = Panorama(os.path.join(self.mapillary_dir_pano, pano))
            p.create_4_images(self.mapillary_dir_images_from_pano)

        # Print results
        self.nb_mapillary_from_pano = 4 * len(panos)
        print("Number of images produced : {}".format(self.nb_mapillary_from_pano))
