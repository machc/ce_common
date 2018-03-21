import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    print('Cannot load matplotlib; continuing...')

import skimage.color
import skimage.draw


def draw_marker_img(im, pos, radius=5, color=(1., 0, 0)):
    """ Return image with marker on (x,y) position pos. """
    imout = im.copy()
    if imout.ndim == 2:
        imout = skimage.color.gray2rgb(imout)

    rr, cc = skimage.draw.circle(int(pos[1]), int(pos[0]), radius, im.shape)
    imout[rr, cc, :] = color
    return imout


def draw_vertical_line_img(im, j, color=(1., 0, 0)):
    """ Return image with vertical line at col j. """
    imout = im.copy()
    if imout.ndim == 2:
        imout = skimage.color.gray2rgb(imout)
    rr, cc = skimage.draw.line(0, int(j), im.shape[0]-1, int(j))
    imout[rr, cc, :] = color
    return imout


def draw_horizontal_line_img(im, i, color=(1., 0, 0)):
    """ Return image with vertical line at row i. """
    imout = im.copy()
    if imout.ndim == 2:
        imout = skimage.color.gray2rgb(imout)
    rr, cc = skimage.draw.line(int(i), 0, int(i), im.shape[1]-1)
    imout[rr, cc, :] = color
    return imout


def draw_box_img(im, r, c, w, h, color=(1., 0, 0)):
    """ Return image box drawn. """
    imout = im.copy()
    if imout.ndim == 2:
        imout = skimage.color.gray2rgb(imout)
    rr, cc = skimage.draw.polygon_perimeter([r, r, r+h, r+h, r], [c, c+w, c+w, c, c])
    imout[rr, cc, :] = color
    return imout


def draw_cross_img(im, pos, radius=5, color=(1., 0, 0)):
    """ Return image with cross on (x,y) position pos. """
    imout = im.copy()
    if imout.ndim == 2:
        imout = skimage.color.gray2rgb(imout)

    i, j, r = pos[1], pos[0], radius
    rr, cc = skimage.draw.line(int(i), int(j-r/2), int(i), int(j+r/2))
    valid = (rr > 0) & (rr < im.shape[1]) & (cc > 0) & (cc < im.shape[1])
    imout[rr[valid], cc[valid], :] = color
    rr, cc = skimage.draw.line(int(i-r/2), int(j), int(i+r/2), int(j))
    valid = (rr > 0) & (rr < im.shape[1]) & (cc > 0) & (cc < im.shape[1])
    imout[rr[valid], cc[valid], :] = color

    return imout


def tile_imgs(imgs, gap=1, fill=np.nan):
    """ Create a single image by tiling nrow x ncol images.

    Args:
        imgs (list of lists)
    """
    if not isinstance(imgs[0], (list, tuple)):
        imgs = [imgs]

    assert imgs[0][0].dtype == 'uint8'

    nrows = len(imgs)
    ncols = len(imgs[0])

    try:
        H, W = imgs[0][0].shape
        figout = np.empty((nrows*H + (nrows-1)*gap,
                           ncols*W + (ncols-1)*gap),
                          dtype='uint8')
    except ValueError:
        H, W, C = imgs[0][0].shape
        figout = np.empty((nrows*H + (nrows-1)*gap,
                           ncols*W + (ncols-1)*gap,
                           C),
                          dtype='uint8')

    figout.fill(fill)

    for i in range(nrows):
        for j in range(ncols):
            top = i*H + i*gap
            left = j*W + j*gap
            figout[top:top+H, left:left+W, ...] = imgs[i][j]

    return figout


def savefig_tight(fname):
    """ Save current figure with the tighest whitespace possible. """
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.margins(0, 0)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.0)


def fig_to_array(fig):
    """ From: http://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array. """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data
def confusion_matrix(matrix, labels=[],
                     title='cols are predictions; rows are real labels',
                     saveto=''):
    import seaborn as sns
    fig = plt.figure()
    f = sns.heatmap(matrix, annot=True, fmt='.2f')
    f.set_xticklabels(labels, rotation='vertical')
    f.set_yticklabels(labels[::-1], rotation='horizontal')
    f.set_title(title)
    if saveto:
        plt.savefig(saveto, bbox_inches='tight', pad_inches=0.0)

    return fig

