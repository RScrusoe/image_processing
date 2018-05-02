from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/mona.jpg').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with


##Test 1:
##to plot histograms:

plot_histo(l)


##Test 2:
##to equalize histogram and show images:
##use eye.png image for good visualization of the method

l = do_hist_equalize(l)
showimg(l)



##Test 3:
##gamma correction
l = do_gamma(l,0.5)
showimg(l)


##Test 4:
##log transform
l = do_log(l)
showimg(l)


##Test 5:
##gaussian blur 
l = do_guassian_blur(l,5)
showimg(l)



##Test 6:
##displaying magnitude of fourier transform 
l = display_fft_magn(l)


##Test 7:
##displaying phase of fourier transform 
l = display_fft_phase(l)



##Test 8:
##Undoo all changes and show original image
l = undo_all(l)
showimg(l)


##Test 9:
##use ns.jpg for this example
##Apply butterworth sharpening filter
l = do_butterworth_sharpening(l,15,2)
