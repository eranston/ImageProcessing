
import numpy as np
import mediapy as video






def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
 
    #load video
    v = video.read_video(video_path)
    image_matrix = np.array(v)
    ind1 , ind2 = cut_scene(image_matrix)
    return (ind1, ind2)
        


def cut_scene(image_matrix):
    """
    the function finds where there is a cut scence by calculating the diffrence between the cumultive histograms of
    the grayscale for each near frames in the video
    :param image_matrix: array of frames that each frame represnted as 3d matrix representing the values of the pixels
    of each color
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    maxvalue = 0
    for frame_index in range(len(image_matrix)-1):
        frames_distance = cumeltive_histogram_diffrence(image_matrix[frame_index] , image_matrix[frame_index+1])
        
        if(frames_distance > maxvalue):
            maxindex = frame_index
            maxvalue = frames_distance
    return (maxindex , maxindex+1)




#transforming color image to grayscale image
def rgb_to_grrayscale(rgb_frame):
    """
    Main entry point for ex1
    :param rgb_frame: color image(3d matrix)
    :return: 2d matrix representing the grayscale image of the color image
    """
    
    # Convert RGB frame to grayscale using the luminosity method
    r, g, b = rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2]
    grayscale_frame = 0.299 * r + 0.587 * g + 0.114 * b
    
    return grayscale_frame.astype(np.uint8)



def cumeltive_histogram_diffrence(frame1 , frame2):
    """
    calculate the distance between 2 cumeltive histograms of the grayscale image
    :param frame1:  image
    :param frame2: the next image
    :return: the distance of the grayscale images
    """
    
    frame1_histogram= make_graycale_histogram(frame1)
    frame2_histogram = make_graycale_histogram(frame2)
    frame1_cumeltive_histogram = np.cumsum(frame1_histogram)
    frame2_cumeltive_histogram = np.cumsum(frame2_histogram)
    
    sum = np.sum(np.abs(frame1_cumeltive_histogram-frame2_cumeltive_histogram))
    return sum




def make_graycale_histogram(image_array):
    """
    create the grayscale histogram of an image
    :param image_array: 3d matrix of represent color image
    :return: the histogram of the grayscale of the image
    """
    
    gray_scale_image = rgb_to_grrayscale(image_array)
    
    return np.histogram(gray_scale_image, bins=256, range=(0, 256))[0]


