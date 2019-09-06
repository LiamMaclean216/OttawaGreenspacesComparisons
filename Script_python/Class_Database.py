# -*- coding: utf-8 -*-
"""
Definition file of the Database class.

This class is used to complete the database used by the web interface to store the comparisons.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import glob
import csv

import progressbar
import mysql.connector

from utils_class import mapillary_to_sql_date, safe_folder_creation
from Class_Image import Image as Ci


# ----------------------------------------------------------------------------------------------------------------------
# Class definition
# ----------------------------------------------------------------------------------------------------------------------
class Database:
    """
    Definition of the database class.
    """
    def __init__(self, db_name, host, user, password, local, ssl_ca, img_dir):
        self.db_name = db_name
        self.host = host
        self.user = user
        self.password = password
        self.ssl_ca = ssl_ca
        self.local = local
        self.img_dir = img_dir

    def __str__(self):
        string = "=== Database information === "
        string += "\nName : " + self.db_name
        string += "\nServer name : " + self.host
        string += "\nUsername : " + self.user
        string += "\nPassword : " + self.password
        string += "\nPath of the ssl certificate : " + self.ssl_ca
        string += "\nPath of the linked image directory : " + self.img_dir
        return string

    def complete_images_table(self, table):
        """
        Complete the images table of the database with the relevant values of the directory images.

        :param table: SQL table name
        :type table: str
        """
        # Variable initialization
        cmp_double = 0
        cmp_img = 0

        # Connection to database
        conn, cursor = connection_database(self.db_name, self.host, self.user, self.password, self.local, self.ssl_ca)

        # Get images paths
        images = glob.glob(self.img_dir + "/*jpg")
        pbar = progressbar.ProgressBar()
        for image_path in pbar(images):
            image_name = os.path.basename(image_path)
            img = Ci(image_path)

            # Get image information
            lon, lat, k, h, datetime, global_key = img.get_info_from_filename()
            date_sql = mapillary_to_sql_date(datetime)

            columns = ("image_key", "latitude", "longitude", "datetime", "filename")
            values = (global_key, lat, lon, date_sql, image_name)

            try:
                insert_into(cursor, table, columns, values)
                cmp_img += 1
            except mysql.connector.errors.IntegrityError:
                # print("Image already in table", image_name)
                cmp_double += 1
                continue

        # Validate queries
        commit_query(conn)

        # End connection to database
        end_connection(conn, cursor)

        # Display information
        print("Number of images inserted into database: {} ".format(cmp_img))
        print("Number of images already inserted in database : {} ".format(cmp_double))

    def get_duels(self, csv_file, table):
        """
        Connects to the SQL server and stores the results of the comparisons in a csv.

        :param csv_file: path of the csv file produced
        :type csv_file: str
        :param table: SQL table name
        :type table: str
        """
        conn, curs = connection_database(self.db_name, self.host, self.user, self.password, self.local, self.ssl_ca)
        query = "SELECT * FROM {};".format(table)
        curs.execute(query)
        result = curs.fetchall()
        end_connection(conn, curs)

        # Write comparisons results in csv file
        with open(csv_file, mode='w') as file:
            file_writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for comparison in result:
                file_writer.writerow(comparison)


# ----------------------------------------------------------------------------------------------------------------------
# Functions definition
# ----------------------------------------------------------------------------------------------------------------------
def connection_database(db_name, host, user, password, local, ssl_ca):
    """
    Creates mysql.connector object to connect to a mysql database.
    :param db_name: database name
    :type db_name: str
    :param host: server address
    :type host: str
    :param user: username
    :type user: str
    :param password: user's password
    :type password: str
    :param local: true if connection to localhost database, false if connection to a remote server
    :type local: bool
    :param ssl_ca: path to ssl certificate
    :type ssl_ca: str
    :return: connection and associated cursor
    """
    if local:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            passwd=password,
            database=db_name
        )
    else:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            passwd=password,
            database=db_name,
            ssl_ca=ssl_ca
        )

    cursor = conn.cursor()
    return conn, cursor


def commit_query(conn):
    """
    Commit queries of a connect instance.
    :param conn: connection to database
    :type conn: mysql.connector.connect
    """
    conn.commit()


def end_connection(conn, cur):
    """
    Closes connection to a database of a connect instance.
    :param conn: connection to database
    :type conn: mysql.connector.connect
    :param cur: cursor of the connection
    :type cur: mysql.connector.connect.cursor
    """
    cur.close()
    conn.close()


def insert_into(cursor, table, columns, values):
    """
    Execute a SQL INSERT INTO queries
    :param cursor: cursor of the connect instance
    :type cursor: mysql.connector.connect.cursor
    :param table: table in which values are inserted
    :type table: str
    :param columns: columns names
    :type columns: tuple(str)
    :param values: values of the row inserted
    :type values: tuple
    """
    columns_str = columns[0]
    values_str = "%s"
    for col in columns[1:]:
        columns_str += ", " + col
        values_str += ", %s"
    query = "INSERT INTO {} ({}) VALUES ({})".format(table, columns_str, values_str)
    cursor.execute(query, values)
