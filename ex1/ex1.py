from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mediapy as video






def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
 

    # Load an image using PIL (Python Imaging Library)
    video_path = [
                    'C:/Users/1eran/OneDrive/Desktop/ImageProcessing/ex1/Exercise Inputs-20240107/video3_category2.mp4' ]
    solution = []
    counter = 1
    for s in video_path:
        v = video.read_video(s)
        image_matrix = np.array(v)
        solution.append(category_images(image_matrix))
        print_image_charcteristics(image_matrix[solution[-1][0]],counter)
        # print_image_charcteristics(image_matrix[solution[-1][1]],counter+1)
        counter+= 2
    return solution
        

    

def category_images(image_matrix):
    maxvalue = 0
    for frame_index in range(len(image_matrix)-1):
        frames_distance = cumeltive_histogram_diffrence(image_matrix[frame_index] , image_matrix[frame_index+1])
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
    
    sum = np.sum(np.abs(frame1_histogram-frame2_histogram))
    return sum




def make_graycale_histogram(image_array):
    gray_scale_image = rgb_to_grrayscale(image_array)
    return np.histogram(gray_scale_image, bins=256, range=(0, 255))[0]



def print_image_charcteristics(image , n):
    
    frame = Image.fromarray(image , 'RGB')
    frame.save(f"frame{n}.png")
    frame.show()

    counts = np.histogram((image[0]), bins=256, range=(0, 255))[0]
    plt.hist(np.array(image[:,:,0]).flatten(), bins=256)
    plt.savefig(f"his{n*10}.png")
    plt.clf()
    plt.hist(np.array(image[:,:,1]).flatten(), bins=256)
    plt.savefig(f"his{(n+1)*10}.png")
    plt.clf()
    plt.hist(np.array(image[:,:,2]).flatten(), bins=256)
    plt.savefig(f"his{(n+2)*10}.png")
    plt.clf()


solutions = main("","")

print(solutions)


