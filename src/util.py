"""
collection of helper utils
"""
from gensim.utils import tokenize
import torch
import sys
import operator
import base64
import json
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)


