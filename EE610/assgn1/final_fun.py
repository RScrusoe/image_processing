import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def showimg(l):
    '''
    accepts np array type object and constructs image from it
    and displays the image
    '''
    l = np.uint8(l)  # converting the values in integers in range of 0-255
    img = Image.fromarray(l)
    img.show()
    return img


def do_fft(nparray):
    '''
    accepts the np array of the image
    and return the DFT of it
    '''
    return np.fft.fftshift(np.fft.fft2(nparray))  # fftshift shifts the spectrum to the centre


def do_ifft(nparray):
    '''
    accepts the np array type object
    and return the inverse fourier transform of it
    '''
    return np.abs(np.fft.ifft2(np.fft.ifftshift(nparray)))


def do_gamma(nparray, gamma):
    '''
    accepts the np array type object and gamma value
    and return the np array type object with gamma correction done
    '''
    nparray = nparray ** gamma
    # maps values in the 0-255 intensity range
    c = 255 / (np.max(nparray) - np.min(nparray))
    nparray = nparray * c
    nparray = np.uint8(nparray)
    return nparray


def do_log(nparray):
    '''
    accepts the np array type object
    and return the np array type object with log transform done
    '''
    nparray = np.log(np.ones(nparray.shape) + nparray)
    # maps values in the 0-255 intensity range
    c = 255 / (np.max(nparray) - np.min(nparray))
    nparray = nparray * c
    nparray = np.uint8(nparray)
    return nparray


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

    # a1,a2 are intermediate terms in calculating gaussian

    a1 = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    a2 = 2 * np.pi * sig ** 2
    kernel = a1 / a2
    kernel = kernel / (np.sum(kernel))
    return kernel


def do_guassian_blur(l, size, sig=1):
    '''
    accepts the np array type object, size of the kernel and sigma which is by default 1
    and return the np array type object with guassian blurring done
    '''
    kernel = guassian_kernel(size, sig)
    l = do_padding(l,50,50,50,50)
    size = kernel.shape[0]
    cstart = np.int((size - 1) / 2)         # col starts here
    cstop = np.int(l.shape[1] - cstart)     # col stops here
    rstart = np.int((size - 1) / 2)         # row starts here
    rstop = np.int(l.shape[0] - rstart)     # row stops here
    tmp = np.int((size - 1) / 2)

    for i in range(rstart, rstop):
        for j in range(cstart, cstop):
            box = l[i - tmp:i + tmp + 1, j - tmp:j + tmp + 1]  # selecting box
            # multiplying box with kernel to update the original value
            l[i][j] = np.sum(box * kernel)
    l = remove_padding(l,50,50,50,50)
    return l


def non_zero_min(l):
    '''
    function to return non zero minimum in the np array
    this will be used in histogram equalization filter
    '''
    for i in l:
        if i == 0:
            continue
        else:
            return i

def do_hist_equalize(l):
    '''
    accepts the np array type object
    and return the np array type object with histogram equalization done
    source for the algorith used here: https://en.wikipedia.org/wiki/Histogram_equalization
    '''
    hist,bins = np.histogram(l.flat, np.arange(256), normed = False)
    cs = np.cumsum(hist)        #calculates cdf
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


def do_padding(l, left, top, right, bottom):
    '''
    accepts the np array type object, left,top,right,bottom are the number of pixels to pad with zeros
    returns np array type object wirh zeros padded
    '''
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    r, c = l.shape[0], l.shape[1]
    newl = np.zeros((r + top + bottom, c + left + right))
    r2, c2 = newl.shape[0], newl.shape[1]
    newl[top:r2 - bottom, left:c2 - right] = l
    return newl


def undo_all():
    '''
    function simply reads original image in the np array otype object and returns it
    '''
    l = np.array(img)
    return l


def plot_histo(l):
    '''
    accepts np array type object
    simple histogram plotting method
    '''

    plt.hist(l.flatten(), bins=256, range=(0.0, 256.0), fc='k', ec='k')
    plt.xlabel('Intensity level')
    plt.ylabel('Number of pixels')
    plt.title("Histogram")
    plt.show()


def display_fft_magn(l):
    '''
    method for displaying magnitude of fourier transform
    accepts np array type object
    and shows the magnitude plot of the fourier transform
    '''
    l = do_fft(l)
    l = abs(l)

    # Remapping values n the range of 0-255 in order to diplay
    maxi = np.max(l)
    mini = np.min(l)
    l = l - mini
    l = l / maxi
    l = l * 255
    showimg(l)



def display_fft_phase(l):
    '''
    method for displaying phase of fourier transform
    accepts np array type object
    and shows the phase plot of the fourier transform
    '''
    l = do_fft(l)

    def phase(i):
        '''
        returns the phase of imaginary number i
        '''
        return np.arctan2(i.imag, i.real)

    # np.vectorize helps effective operation of phase method to each of the element in array 'l'
    phase = np.vectorize(phase, otypes=[np.float]) 
    l = phase(l)
    
    # Remapping values n the range of 0-255 in order to diplay
    maxi = np.max(l.ravel())
    mini = np.min(l.ravel())
    l = l - mini
    l = l / maxi
    l = l * 255
    showimg(l)

def butterworth_filter(u,v,centeru,centerv,d0,n):
    '''
    butterworth filter:
    accepts u,v, centeru, centerv, d0,order (n)
    and returns value of filter
    '''

    dist = np.sqrt((u-centeru)**2 + (v-centerv)**2)
    if dist == 0:
        return 1
    else:
        return 1/(1+ (d0/dist)**(2*n) )

def remove_padding(l,left,top,right,bottom):
    '''
    accepts the np array type object, left,top,right,bottom are the number of pixels to remove padding
    returns np array type object wirh padding
    '''
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    r, c = l.shape[0], l.shape[1]
    newl = l[top:l.shape[0]-bottom,left:l.shape[1]-right]
    return newl


def do_butterworth_sharpening(l,d0,n):
    '''
    Butterworth sharpening filter
    accpets np array type object, d0 and n 
    returns new np array type object and shows butterworth sharperned image
    '''
    oldl = l
    l = do_padding(l,l.shape[1]/2,l.shape[0]/2,l.shape[1]/2,l.shape[0]/2)
    l = do_fft(l)
    centeru,centerv = int(l.shape[0]/2), int(l.shape[1]/2)

    for r in range(l.shape[0]):
        for c in range(l.shape[1]):
            l[r][c] = l[r][c] * butterworth_filter(r,c,centeru,centerv,d0,n)

    l = do_ifft(l)
    l = remove_padding(l,oldl.shape[1]/2,oldl.shape[0]/2,oldl.shape[1]/2,oldl.shape[0]/2)
    l = l-np.min(l)
    l = l/np.max(l)
    l = l*255
    showimg(l)
    return l
