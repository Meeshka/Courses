from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

import matplotlib
import matplotlib.pyplot as plt


def findFiles(path):
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


if __name__ == "__main__":

    print(findFiles('data/names/*.txt'))


    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)


    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    print(unicodeToAscii('Ślusàrski'))
    # Read a file and split into lines

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)