from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/ns.jpg').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with



##Test 9:
##use ns.jpg for this example
##Apply butterworth sharpening filter
d0 = float(input('Input D0 :'))
n = float(input('Input order n :'))
showimg(l)

l = do_butterworth_sharpening(l,d0,n)
