from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/zebra.png').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with

##Test 7:
##displaying phase of fourier transform 
showimg(l)
l = display_fft_phase(l)