from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mediapy as video


FRAME_SCENCE_CUT_VALUE = 1000000



def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
 

    # Load an image using PIL (Python Imaging Library)
    video_path = 'C:/Users/1eran/OneDrive/Desktop/ImageProcessing/ex1/Exercise Inputs-20240107/video3_category2.mp4'

    v = video.read_video(video_path)
    image_matrix = np.array(v)

    ind1 , ind2 = category_images(image_matrix)
    print_image_charcteristics(image_matrix[ind1],1)
    print_image_charcteristics(image_matrix[ind2],2)
    return (ind1, ind2)
        

    

def category_images(image_matrix):
    maxvalue = 0
    for frame_index in range(len(image_matrix)-1):
        frames_distance = cumeltive_histogram_diffrence(image_matrix[frame_index] , image_matrix[frame_index+1])
        print(f"frame number: {frame_index} , diff value: {frames_distance} ")
        if(frames_distance > maxvalue):
            maxindex = frame_index
            maxvalue = frames_distance
    return (maxindex , maxindex+1)





def rgb_to_grrayscale(rgb_frame):
    
    # Convert RGB frame to grayscale using the luminosity method
    r, g, b = rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2]
    grayscale_frame = 0.299 * r + 0.587 * g + 0.114 * b
    
    return grayscale_frame.astype(np.uint8)



def cumeltive_histogram_diffrence(frame1 , frame2):
    
    frame1_histogram= make_graycale_histogram(frame1)
    frame2_histogram = make_graycale_histogram(frame2)
    frame1_cumeltive_histogram = np.cumsum(frame1_histogram)
    frame2_cumeltive_histogram = np.cumsum(frame2_histogram)
    
    sum = np.sum(np.abs(frame1_cumeltive_histogram-frame2_cumeltive_histogram))
    return sum




def make_graycale_histogram(image_array):
    gray_scale_image = rgb_to_grrayscale(image_array)
    return np.histogram(gray_scale_image, bins=256, range=(0, 256))[0]



def print_image_charcteristics(image , n):
    frame = Image.fromarray(image , 'RGB')

    frame.show()
    
    counts, bins = np.histogram(image)
    plt.stairs(counts, bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig(f"his{n}.png")
    plt.clf()


frame1 , frame2 = main("","")

print(frame1 , frame2)


