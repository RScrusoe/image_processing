import numpy as np
from PIL import Image
import scipy.fftpack as fp
from matplotlib import pyplot as plt


def shift_image(im, x, y):
    '''
    im s=must be greyscale image with only one channel
    downwards   y is +ve
    rightwords  x is +ve
    '''
    tx = np.array([0 for j in range(abs(x))])
    ty = np.array([0 for j in range(im.shape[1])])
    t = []
    if x >= 0:
        for i in range(im.shape[0]):
            t.append(np.array(np.hstack((tx, im[i]))[:-x]))
    else:
        for i in range(im.shape[0]):
            t.append(np.array(np.hstack((im[i][-x:], tx))))

    if y > 0:
        for i in range(abs(y)):
            t = np.vstack((ty, t))
        t = t[:-abs(y)]
    elif y < 0:
        for i in range(abs(y)):
            t = np.vstack((t, ty))
        t = t[abs(y):]
    return np.array(t)


def do_fft(nparray):
    '''
    accepts the no array of the image
    and return the DFT of it
    # l2 = np.log(k*abs(do_fft(nparray))+1)
    this is used to show the image of frequency transform
    '''
    return np.fft.fftshift(np.fft.fft2(nparray))  # fftshift shifts the spectrum to the centre


def do_ifft(nparray):
    return abs(np.fft.ifft2(np.fft.ifftshift(nparray)))


def showimg(l):
    l = np.uint8(l)
    img = Image.fromarray(l)
    img.show()
    return img


def do_gamma(nparray, gamma):
    nparray = nparray ** gamma
    c = 255 / (np.max(nparray) - np.min(nparray))
    nparray = nparray * c
    nparray = np.uint8(nparray)
    return nparray

# Example of gamma transform
# l = do_gamma(l,0.3)
# showimg(l)
# print(np.max(l))


def do_log(nparray):
    nparray = np.log(np.ones(nparray.shape) + nparray)
    c = 255 / (np.max(nparray) - np.min(nparray))
    nparray = nparray * c
    nparray = np.uint8(nparray)
    return nparray
# Example of log transform
# l = do_log(l)
# showimg(l)
# print(np.max(l))


def non_zero_min(l):
    for i in l:
        if i == 0:
            continue
        else:
            return i


def guassian_kernel(size, sig=1):
    '''
    returns the guassian kernel (square sized)
    assume size is odd number
    if not, it will just add one to it
    '''
    if size % 2 == 0:
        size += 1
    tmp = (size - 1) / 2
    x = np.arange(-1 * tmp, tmp + 1)
    y = np.arange(-1 * tmp, tmp + 1)
    x, y = np.meshgrid(x, y)
    a1 = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    a2 = 2 * np.pi * sig ** 2
    kernel = a1 / a2
    kernel = kernel / (np.sum(kernel))
    return kernel


def do_guassian_blur(l, size, sig=1):
    '''
    assumes kernel is square matrix
    '''
    kernel = guassian_kernel(size,sig)
    size = kernel.shape[0]
    cstart = np.int((size - 1) / 2)         # col starts here
    cstop = np.int(l.shape[1] - cstart)     # col stops here
    rstart = np.int((size - 1) / 2)         # row starts here
    rstop = np.int(l.shape[0] - rstart)     # row stops here
    tmp = np.int((size - 1) / 2)

    for i in range(rstart, rstop):
        for j in range(cstart, cstop):
            box = l[i - tmp:i + tmp + 1, j - tmp:j + tmp + 1]
            l[i][j] = np.sum(box * kernel)

    return l


def do_hist_equalize(l):
    hist, bins = np.histogram(l.flat, np.arange(256), normed=False)
    cs = np.cumsum(hist)
    lflat = l.flat
    new_intensity = []
    for i in range(255):
        new_intensity.append(
            np.int((cs[i] - non_zero_min(cs)) * 255 / (l.shape[0] * l.shape[1] - 1)))

    print(new_intensity[254])
    print(new_intensity[253])
    for i in range(len(lflat)):
        if lflat[i] == 255:
            lflat[i] = 254
        lflat[i] = new_intensity[lflat[i]]
    lnew = np.array(lflat).reshape(l.shape)
    return lnew

# example of do_hist_equalize()
# check with car.png or eye.png
# lnew = do_hist_equalize(l)
# showimg(lnew)
# hist2, bins = np.histogram(lnew.flat, np.arange(256), normed=False)
# print(bins)
# plt.plot(hist2)
# plt.show()


def do_padding(l, left, top, right, bottom, pad=0):
    '''
    l is np array of image
    '''
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    r, c = l.shape[0], l.shape[1]
    newl = np.zeros((r + top + bottom, c + left + right))
    r2, c2 = newl.shape[0], newl.shape[1]
    newl[top:r2 - bottom, left:c2 - right] = l
    return newl


def undo_all():
    l = np.array(img)
    return l


def plot_histo(l):
    plt.hist(l.flatten(), bins=256, range=(0.0, 256.0), fc='k', ec='k')
    plt.xlabel('Intensity level')
    plt.ylabel('Number of pixels')
    plt.title("Histogram")
    plt.show()


def display_fft_magn(l):
    l = do_fft(l)
    l = abs(l)
    maxi = np.max(l.ravel())
    mini = np.min(l.ravel())
    l = l - mini
    l = l / maxi
    l = l * 255
    showimg(l)


def display_fft_phase(l):
    l = do_fft(l)

    def phase(i):
        '''
        returns the phase of imaginary number i
        '''
        return np.arctan2(i.imag, i.real)

    #np.vectorize helps effective operation of phase method to each of the element in array 'l'
    phase = np.vectorize(phase, otypes=[np.float]) 
    l = phase(l)
    
    maxi = np.max(l.ravel())
    mini = np.min(l.ravel())
    l = l - mini
    l = l / maxi
    l = l * 255
    showimg(l)
