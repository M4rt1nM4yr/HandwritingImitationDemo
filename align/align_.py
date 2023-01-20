import numpy as np
import matplotlib.pyplot as plt
from skimage import io

plt.gray()
from skimage.transform import rescale


def get_index(img):
    addition_per_row = np.sum(img, axis=1)
    m = addition_per_row.min()
    m_index = np.argmin(addition_per_row)
    std = addition_per_row.std()
    for i, val in enumerate(addition_per_row):
        if val > m + 2 * std and i > m_index:
            break
    return i


def clean_word(im):
    addition_per_column = np.sum(im, axis=0)
    std = addition_per_column.std()
    mean = addition_per_column.mean()
    for i, val in enumerate(addition_per_column):
        if val > mean + std * 2 and i > im.shape[1] / 2:
            im = im[:, :i]
            break
    return im


def put_in_canvas(words):
    size = 900
    canvas = np.ones((size, size))
    r = 90
    c = 20
    gap = 20
    for word in words:
        word = clean_word(word)
        if c + word.shape[1] + gap > size:
            r += 90
            c = 5
        i = get_index(word)
        canvas[r - i:r + word.shape[0] - i, c:c + word.shape[1]] = word
        c += word.shape[1] + gap

    return canvas


if __name__ == "__main__":
    words = []
    for i in range(10):
        im = rescale(io.imread(str(i) + '.png', as_gray=True), 0.9)
        words.append(im)
    put_in_canvas(words)