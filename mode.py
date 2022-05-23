# -*- coding: utf-8 -*-
"""
Created on Sat May  8 21:42:21 2021

@author: HP
"""

import enum


class Mode(enum.Enum):
    NONE = 0
    ONE_PATH_FIXED = 1
    ONE_PATH_RANDOM = 2
    TWO_PATHS = 3
    ALL_PATHS = 4