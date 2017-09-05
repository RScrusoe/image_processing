from final_fun import *


#this will open image
#various image files are located in './../src/' 
img = Image.open('./../src/mona.jpg').convert('L')      #opening any image by converting in greyscale
l = np.array(img)       #converting image into np array type object to play with


##Test 5:
##gaussian blur 
size = int(input('Input size of the kerner (odd number) : '))
sig = float(input('Input sigma for the kerner : '))
showimg(l)
l = do_guassian_blur(l,15,2)
showimg(l)