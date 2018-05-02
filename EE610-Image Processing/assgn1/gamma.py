from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/mona.jpg').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with


##Test 3:
##gamma correction

gamma = float(input('Input Gamma Value: '))
showimg(l)
l = do_gamma(l,gamma)
showimg(l)