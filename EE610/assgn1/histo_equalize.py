from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/eye.png').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with



##Test 2:
##to equalize histogram and show images:
##use eye.png image for good visualization of the method
showimg(l)
plot_histo(l)
l = do_hist_equalize(l)
showimg(l)
plot_histo(l)
