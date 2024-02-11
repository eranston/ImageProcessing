
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal



def blur_image(image ,kernel):
    row_blured = np.array(scipy.signal.convolve2d(image,kernel,mode="same" , boundary="symm"),dtype=np.float32)
    blured_image = np.array(scipy.signal.convolve2d(row_blured,kernel.T,mode="same" , boundary="symm"),dtype=np.float32)
    return blured_image


def blur_and_shrink(image,kernel=np.array([[1,2,1]])*(0.25)):
    image = np.array(image)
    blured_image = np.array(blur_image(image,kernel))
    return np.array(blured_image[::2,::2])

def blur_rgb_image_and_shrink(image):
    image = np.array(image)
    new_image_shape= (image.shape[0]//2,image.shape[1]//2,image.shape[2])
    new_image = np.zeros(new_image_shape)
    for color in range(3):
        new_image[:,:,color] =np.array(blur_and_shrink(image[:,:,color]))
       

    return np.array(new_image)

def resize_image(image , size ):
    new_image = np.zeros(size)
    for color in range(size[2]):
        new_image[:,:,color] = np.array(Image.fromarray(image[:,:,color]).resize((size[0],size[1])))
    return new_image

def expend_image(image):
    new_image = np.zeros((image.shape[0]*2,image.shape[1]*2,3))
    for color in range(3):
        array = np.zeros((image.shape[0]*2,image.shape[1]*2))
        array[::2,::2] = image[:,:,color]
        new_image[:,:,color] = array
    return new_image

def blur_expend(image):
    kernel = np.array([0.5,1,0.5])
    for color in range(len(np.array(image).shape)):
        array = np.array(image[:,:,color])
        for row in range(len(image)):
                array[row] = np.convolve(array[row], kernel ,"same")
        array = np.transpose(array)
        for row in range(len(image)):
                array[row] = np.convolve(array[row], kernel ,"same")
        image[:,:,color] = np.transpose(array)
    return image

def expend_and_blur(image):
    image = expend_image(image)
    image = blur_expend(image)
    return np.array(image)

def merge_pyramids(pyramid1 , pyramid2 , mask_gausian_pyramid):
    laplacian_pyramid_merged_image = []
    laplacian_pyramid_image1 = pyramid1
    laplacian_pyramid_image2 = pyramid2
    for k in range(len(laplacian_pyramid_image1)):
        a = laplacian_pyramid_image1[k]
        b  = laplacian_pyramid_image2[k]
        mask = mask_gausian_pyramid[k]
        image = np.zeros(a.shape)
        image = mask*b + (1-mask)*a
        laplacian_pyramid_merged_image.append(image)

    return laplacian_pyramid_merged_image
            
def mask_blur(image,kernel=np.array([[1.0,2,1.0]])*0.25):
    new_image = np.zeros((image.shape[0]//2,image.shape[1]//2,image.shape[2]), dtype=np.float32)
    for color in range(image.shape[2]):
        new_image[:,:,color] = np.array(blur_and_shrink(image[:,:,color],kernel),dtype=np.float32)
   
    return new_image
        
def create_mask_pyramid(image , level):
    gaussian_pyramid = [image]
    for i in range(level):
        image = np.array(mask_blur(image),dtype=np.float32)
        gaussian_pyramid.append(np.array(image,dtype=np.float32))

        
    return gaussian_pyramid

def blend_mask(size):
    mask = np.zeros(size).astype(np.float32)
    mask[:,size[1]//2:] = 1.0
    return mask

def create_laplacian_pyramid(image1,level ):
    gaussian_pyramid_image1 = [np.array(image1)]
    
    
    laplacian_pyramid_image1 = []
    blured_image1 = image1
    for i in range(level):

        blured_image1 =blur_rgb_image_and_shrink(np.array(image1))
        temp1 = np.array(blured_image1)
        expened_image1 = expend_and_blur(np.array(temp1))
        lalpacian_image1 = np.array(image1) - np.array(expened_image1)
        
        
        laplacian_pyramid_image1.append(lalpacian_image1)
        gaussian_pyramid_image1.append(blured_image1)
        image1 = blured_image1
        
    
    laplacian_pyramid_image1.append(np.array(blured_image1))


    return laplacian_pyramid_image1
   



image_size = (1024,1024,3)
mask = blend_mask(image_size)
mask_gausian_pyramid = create_mask_pyramid(mask,7)


image1 = Image.open("apple.jpg")
image2 = Image.open("orange.jpg")

image1 = image1.resize(image_size[:2])
image2 = image2.resize(image_size[:2])



laplacian_pyramid_image1= create_laplacian_pyramid(np.array(image1),7)
laplacian_pyramid_image2= create_laplacian_pyramid(np.array(image2),7)
laplacian_pyramid_merged_image = merge_pyramids(laplacian_pyramid_image1 , laplacian_pyramid_image2 ,mask_gausian_pyramid)
image1 = np.array(image1)
image2 = np.array(image2)   



# merged_last_image = np.concatenate((image1[:,:image1.shape[1]//2],image2[:,image2.shape[1]//2:]),axis=1)
# laplacian_pyramid_merged_image.append(merged_last_image)


# need to check why from pyramid to image not working
new_image = laplacian_pyramid_merged_image[-1]

print(len(laplacian_pyramid_image1))
print(len(laplacian_pyramid_merged_image))


for k in range(len(laplacian_pyramid_image1)-2 , -1 , -1):

    image = np.array(laplacian_pyramid_merged_image[k])
    new_image = expend_and_blur(new_image)

    new_image = new_image + image

clipped_image = np.clip(new_image,0,255).astype(np.uint8)







                
            
            




