import cv2 
import numpy as np
import math
import random
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from website implemenation of SIFT 
# https://www.geeksforgeeks.org/python-opencv-bfmatcher-function/
def SIFT(img1, img2, file_name):
    
    #convert images to grayscale 
    img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
    img2_grayscale = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    
    #make sift 
    sift = cv2.SIFT_create()
    
    #get keypoints and descriptors 
    kp1, descriptors1 = sift.detectAndCompute(img1_grayscale, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_grayscale, None)
    
    #create matcher 
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck =True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda val:val.distance)
    matches = matches[:200] #get best 200 matches
    
    points1 = []
    points2 = []

    #get the points in two list for matches
    for match in matches:
        # (x1,y1) = kp1[match.queryIdx].pt
        # (x2,y2) = kp2[match.trainIdx].pt
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    
    #draw the matches
    out = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    
    #save the image 
    cv2.imwrite(file_name + '_SIFT.jpeg', out)  
    
    return points1, points2 

def create_homography_estimation(x , x_prime):
    
    x = np.array(x)
    y = np.array(x_prime)
    x_len = len(x)

    # b = A^-1 * c
    A = np.zeros((2 * x_len, 8))  
    c = np.array(x_prime)
    c = c.reshape((2 * x_len, 1))
    
    #make A 
    for i in range(x_len):
        A[2 * i] = [x[i][0], x[i][1], 1, 0, 0, 0, -y[i][0] * x[i][0], y[i][0] * x[i][1]]
        A[2 * i + 1] = [0, 0, 0, x[i][0], x[i][1], 1, -y[i][1] * x[i][0], -y[i][1] * x[i][1]]

    # get inverse of A
    A_inv = np.linalg.pinv(A)
    
    #compute b which is the homography values
    b = np.dot(A_inv, c)
    
    #construct homography 3x3 matrix
    homography = np.zeros((3, 3))
    homography.flat[:b.size] = b
    homography[2][2] = 1

    return homography

def calculate_inlier(x, x_prime, H, delta):
    
    #make homogenous 
    x_3d = np.hstack((x, np.ones((x.shape[0], 1))))  
    prime_3d = np.hstack((x_prime, np.ones((x_prime.shape[0], 1))))  
    
    #set values
    distance = []
    inlier1 = []
    inlier2 = [] 
    outlier1 = [] 
    outlier2 = []
    
    #loop through and get number of distance less than delta
    for i, j in zip(x_3d, prime_3d):
        
        #apply homography
        est = np.matmul(H,i)
        est /= est[2]
        
        #calc distance between points 
        distance = np.sqrt((est[0] - j[0])**2 + (est[1] - j[1])**2)
        
        #inlier if less than delta and outlier otherwise
        if distance < delta:
            inlier1.append(i)
            inlier2.append(j)
        else:
            outlier1.append(i)
            outlier2.append(j)
    
    #num of inliers
    count = np.sum(inlier1)
    
    return inlier1, inlier2, outlier1, outlier2, count

def RANSAC(x, x_prime):    
    
    #initialize the varaibles for RANSAC 
    n = 4
    n_total = len(x)
    sigma = 1
    delta = sigma * 3 
    epsilon = 0.5
    p = 0.99
    N = int(math.log(1-p)/math.log(1-(1-epsilon)**n))
    M = int((1-epsilon)* n_total)
    error = False
    
    # set lists to be numpy  
    x = np.array(x)
    x_prime = np.array(x_prime)
    
    #loop through number of trials N
    for i in range(N):
        #get random samples points
        rand_idx = random.sample(range(0, n_total), n)
        src_random = [x[idx] for idx in rand_idx]
        dest_random = [x_prime[idx] for idx in rand_idx]
        
        #calc homography matrix using rand points
        H = create_homography_estimation(src_random, dest_random)
        
        #find inliers 
        in1, in2, out1, out2, count = calculate_inlier(x, x_prime, H, delta)

        #check M 
        if count > M:
            M = count
            best_inliers1 = in1
            best_inliers2 = in2
            best_outliers1 = out1
            best_outliers2 = out2
            best_H = H 
            error = True
    
    if error == False:
        print("ERROR: M too small")
        return 
    
    print("FOUND VALUE")
    return best_inliers1, best_inliers2, best_outliers1, best_outliers2, best_H

def draw_inlier_image(img1, img2, inlier_x, inlier_x_prime, outlier_x, outlier_x_prime, file_name):
    
    #get size of images and get colors for inlier/outliers
    height1, width1 = img1.shape[0:2]
    height2, width2 = img2.shape[0:2] 
    
    #make a new image 
    image_combined = np.concatenate((img1, img2), axis=1)
    
    #add points to image 
    for i in range(len(inlier_x)):
        point_img1 = inlier_x[i]
        point_img2 = inlier_x_prime[i]
        #add offset for 2nd image
        point_img2[0] = point_img2[0] + width1
        point_img1 = point_img1[:2]
        point_img2 = point_img2[:2]
        #create line and points
        cv2.circle(image_combined, tuple(point_img1.astype(int)), radius=3, 
                   color=(0,255,0), thickness=-1)
        cv2.circle(image_combined, tuple(point_img2.astype(int)), radius=3, 
                   color=(0,255,0), thickness=-1)
        cv2.line(image_combined, tuple(point_img1.astype(int)), 
                 tuple(point_img2.astype(int)), color=(0,255,0), thickness=1)
    for i in range(len(outlier_x)):
        point_img1 = outlier_x[i]
        point_img2 = outlier_x_prime[i]
        #add offset for 2nd image
        point_img2[0] = point_img2[0] + width1
        point_img1 = point_img1[:2]
        point_img2 = point_img2[:2]
        #create line and points
        cv2.circle(image_combined, tuple(point_img1.astype(int)), radius=3, 
                   color=(0,0,255), thickness=-1)
        cv2.circle(image_combined, tuple(point_img2.astype(int)), radius=3, 
                   color=(0,0,255), thickness=-1)
        cv2.line(image_combined, tuple(point_img1.astype(int)), 
                 tuple(point_img2.astype(int)), color=(0,0,255), thickness=1)
        
    #save the image 
    cv2.imwrite(file_name + '_RANSAC.jpeg', image_combined)  
        
    return 

#cost function for LM optimization 
def cost_function(h, x, x_prime):
    
    x = np.array(x)
    x_prime = np.array(x_prime)
    h = h.reshape(3, 3)
 
    # Apply the homography transformation
    transformed_points = np.matmul(h,x.T).T  
    last_coordinate = transformed_points[:, -1].reshape(-1, 1)
    normalized_points = transformed_points / last_coordinate 
    f_p = normalized_points.astype(int)  

    #get transformed x and y coordinates
    f_x = f_p[:, 0]
    f_y = f_p[:, 1]

    #calc cost using norm 
    cost = np.sqrt((x_prime[:, 0] - f_x) ** 2 + (x_prime[:, 1] - f_y) ** 2)
    
    return(cost)

#this is vectorized code for putting images together 
def apply_one_img2(img, homography, final_img, min_width, min_height):
    
    #get final image mesh
    mesh_x, mesh_y = np.meshgrid(np.arange(final_img.shape[1]), np.arange(final_img.shape[0]))
    final_pts = np.stack([mesh_x.ravel() + min_width, mesh_y.ravel() 
                          + min_height, np.ones(mesh_x.size)], axis=0)

    #apply homography
    inverse_homography = np.linalg.inv(homography)
    original_coords = np.matmul(inverse_homography, final_pts)
    original_coords /= original_coords[2, :]  

    #get x and y values 
    x_orig = original_coords[0, :].astype(int)
    y_orig = original_coords[1, :].astype(int)

    #get valid indicies if x_orig and y_orig in bounds
    valid = (x_orig >= 0) & (x_orig < img.shape[1]) & (y_orig >= 0) & (y_orig < img.shape[0])

    #get final image using the valid indicies
    fin_valid_y = mesh_y.ravel()[valid]
    fin_valid_x = mesh_x.ravel()[valid]
    final_img[fin_valid_y, fin_valid_x] = img[y_orig[valid], x_orig[valid]]

    return final_img

#this is not vectorized code for putting images together 
def apply_one_img(dest_img, apply_img, H):
    
    #get shapes 
    dest_h, dest_w = dest_img.shape[:2] 
    apply_h, apply_w = apply_img.shape[:2]  
    
    #get homography
    Hinv = np.linalg.inv(H)  
    Hinv = Hinv / Hinv[2,2]
    
    #iterate through the dest image and replace pixels if in frame 
    for y in range(dest_h):
        for x in range(dest_w):
            # do inverse transformation to get the source coordinates
            three_d_coords = np.matmul(Hinv, np.array([x, y, 1]))
            x_real, y_real, _ = three_d_coords / three_d_coords[2]
            
            #check the source coordinates are in the bounds of the source image
            if 0 <= x_real < apply_w and 0 <= y_real < apply_h:
                x_real, y_real = int(x_real), int(y_real)
                
                #assign the pixel value from the source image to the destination image
                dest_img[y, x] = apply_img[y_real, x_real]

    return dest_img


def find_corner_points(homography, width, height):
    
    # Define the corner points of the input image
    corners = np.array([
        [0, 0],        
        [width, 0],    
        [width, height], 
        [0, height]      
    ], dtype=np.float32)

    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    new_corners = np.matmul(homography, corners_homogeneous.T).T
    new_corners /= new_corners[:, 2:3] 
    
    w_min = np.floor(np.min(new_corners[:, 0])).astype(int)
    w = np.ceil(np.max(new_corners[:, 0])).astype(int)
    h_min = np.floor(np.min(new_corners[:, 1])).astype(int)
    h = np.ceil(np.max(new_corners[:, 1])).astype(int)
    
    print('each img')
    print(w, h, w_min, h_min)
    return w, h, w_min, h_min
    

def stich_images(homography_12, homography_23, homography_34, homography_45, img_list, file_name):
    
    # make new homographies 
    homography_13 = np.matmul(homography_23, homography_12)
    homography_23 = homography_23
    homography_3 = np.eye(3)
    homography_43 = np.linalg.inv(homography_34)
    homography_53 = np.linalg.inv(np.matmul(homography_45,homography_34))
    homography_list = [homography_13, homography_23, homography_3 ,homography_43, homography_53]
        
    #find pararama dimensions
    height_max = 0 
    width_max = 0 
    height_min = 0 
    width_min = 0 
    height = 0 
    width = 0
    
    for i in range(5):
        img_height, img_width = img_list[i].shape[:2]
        max_w, max_h, min_w, min_h = find_corner_points(homography_list[i],img_width, img_height)
        
        height_max = max(height_max,max_h) 
        width_max = max(width_max,max_w) 
        height_min = min(height_min,min_h) 
        width_min = min(width_min,min_w) 
        
    width = width_max - width_min
    height = height_max - height_min
    final_img = np.zeros((height, width,3), np.uint8)
    
    #tranlation matrix
    translation_homography = np.eye(3)
    translation_homography[0,2] = -height_min
    translation_homography[1,2] = -width_min
    
    for i in range(5):
        # new_h = np.matmul(translation_homography, homography_list[i])
        # final_img = apply_one_img(final_img, img_list[i], new_h)
        apply_one_img2(img_list[i], homography_list[i], final_img, width_min, height_min)
        
    #save the image 
    cv2.imwrite(file_name + '_pana.jpeg', final_img) 
    
    return 

def main():

    #extract all the images 
    img_1 = cv2.imread('1.jpg')
    img_2 = cv2.imread('2.jpg') 
    img_3 = cv2.imread('3.jpg')
    img_4 = cv2.imread('4.jpg') 
    img_5 = cv2.imread('5.jpg')
    img_list = [img_1, img_2, img_3, img_4, img_5]
    
    #apply sift to get the intest points between each adjacent image
    point_x_12, point_xprime_12 = SIFT(img_1, img_2, '1_2')
    point_x_23, point_xprime_23 = SIFT(img_2, img_3, '2_3')
    point_x_34, point_xprime_34 = SIFT(img_3, img_4, '3_4')
    point_x_45, point_xprime_45 = SIFT(img_4, img_5, '4_5')
    
    #apply RANSAC 
    inlier_x_12, inlier_x_prime_12, outlier_x_12, outlier_x_prime_12, \
    H_12 = RANSAC(point_x_12, point_xprime_12)
    inlier_x_23, inlier_x_prime_23, outlier_x_23, outlier_x_prime_23, \
        H_23 = RANSAC(point_x_23, point_xprime_23)
    inlier_x_34, inlier_x_prime_34, outlier_x_34, outlier_x_prime_34, \
        H_34 = RANSAC(point_x_34, point_xprime_34)
    inlier_x_45, inlier_x_prime_45, outlier_x_45, outlier_x_prime_45, \
        H_45 = RANSAC(point_x_45, point_xprime_45)

    #display the RANSAC correspondes 
    draw_inlier_image(img_1, img_2, inlier_x_12, inlier_x_prime_12, 
                      outlier_x_12, outlier_x_prime_12, '1_2')
    draw_inlier_image(img_2, img_3, inlier_x_23, inlier_x_prime_23, 
                      outlier_x_23, outlier_x_prime_23, '2_3')
    draw_inlier_image(img_3, img_4, inlier_x_34, inlier_x_prime_34, 
                      outlier_x_34, outlier_x_prime_34, '3_4')
    draw_inlier_image(img_4, img_5, inlier_x_45, inlier_x_prime_45, 
                      outlier_x_45, outlier_x_prime_45, '4_5')

    # use spicy lm optimization 
    homography_12 = least_squares(cost_function, H_12.flatten(), 
                                  args=(inlier_x_12, inlier_x_prime_12)).x.reshape(3,3)
    homography_23 = least_squares(cost_function, H_23.flatten(), 
                                  args=(inlier_x_23, inlier_x_prime_23)).x.reshape(3,3)
    homography_34 = least_squares(cost_function, H_34.flatten(), 
                                  args=(inlier_x_34, inlier_x_prime_34)).x.reshape(3,3)
    homography_45 = least_squares(cost_function, H_45.flatten(), 
                                  args=(inlier_x_45, inlier_x_prime_45)).x.reshape(3,3)
    
    #extract all the images again
    img_1 = cv2.imread('1.jpg')
    img_2 = cv2.imread('2.jpg') 
    img_3 = cv2.imread('3.jpg')
    img_4 = cv2.imread('4.jpg') 
    img_5 = cv2.imread('5.jpg')
    img_list = [img_1, img_2, img_3, img_4, img_5]
    
    #stich the images together 
    stich_images(homography_12, homography_23, homography_34, homography_45, 
                 img_list, 'fountain_LM')
    stich_images(H_12, H_23, H_34, H_45, img_list, 'fountain_nonLM')
    
    
    ########### Task 2 ####################
    
    #extract all the images 
    img_1 = cv2.imread('my_1.jpg')
    img_2 = cv2.imread('my_2.jpg') 
    img_3 = cv2.imread('my_3.jpg')
    img_4 = cv2.imread('my_4.jpg') 
    img_5 = cv2.imread('my_5.jpg')
    img_list = [img_1, img_2, img_3, img_4, img_5]
    
    
    
    #apply sift to get the intest points between each adjacent image
    point_x_12, point_xprime_12 = SIFT(img_1, img_2, '1_2_my')
    point_x_23, point_xprime_23 = SIFT(img_2, img_3, '2_3_my')
    point_x_34, point_xprime_34 = SIFT(img_3, img_4, '3_4_my')
    point_x_45, point_xprime_45 = SIFT(img_4, img_5, '4_5_my')
    
    
    
    #apply RANSAC 
    inlier_x_12, inlier_x_prime_12, outlier_x_12, outlier_x_prime_12, /
    H_12 = RANSAC(point_x_12, point_xprime_12)
    inlier_x_23, inlier_x_prime_23, outlier_x_23, outlier_x_prime_23, /
    H_23 = RANSAC(point_x_23, point_xprime_23)
    inlier_x_34, inlier_x_prime_34, outlier_x_34, outlier_x_prime_34, /
    H_34 = RANSAC(point_x_34, point_xprime_34)
    inlier_x_45, inlier_x_prime_45, outlier_x_45, outlier_x_prime_45, /
    H_45 = RANSAC(point_x_45, point_xprime_45)



    #display the RANSAC correspondes 
    draw_inlier_image(img_1, img_2, inlier_x_12, inlier_x_prime_12, 
                      outlier_x_12, outlier_x_prime_12, '1_2_my')
    draw_inlier_image(img_2, img_3, inlier_x_23, inlier_x_prime_23, 
                      outlier_x_23, outlier_x_prime_23, '2_3_my')
    draw_inlier_image(img_3, img_4, inlier_x_34, inlier_x_prime_34, 
                      outlier_x_34, outlier_x_prime_34, '3_4_my')
    draw_inlier_image(img_4, img_5, inlier_x_45, inlier_x_prime_45, 
                      outlier_x_45, outlier_x_prime_45, '4_5_my')



    # use spicy lm optimization 
    homography_12 = least_squares(cost_function, H_12.flatten(), 
                                  args=(inlier_x_12, inlier_x_prime_12)).x.reshape(3,3)
    homography_23 = least_squares(cost_function, H_23.flatten(), 
                                  args=(inlier_x_23, inlier_x_prime_23)).x.reshape(3,3)
    homography_34 = least_squares(cost_function, H_34.flatten(), 
                                  args=(inlier_x_34, inlier_x_prime_34)).x.reshape(3,3)
    homography_45 = least_squares(cost_function, H_45.flatten(), 
                                  args=(inlier_x_45, inlier_x_prime_45)).x.reshape(3,3)
    
 
    
    #extract all the images again
    img_1 = cv2.imread('my_1.jpg')
    img_2 = cv2.imread('my_2.jpg') 
    img_3 = cv2.imread('my_3.jpg')
    img_4 = cv2.imread('my_4.jpg') 
    img_5 = cv2.imread('my_5.jpg')
    img_list = [img_1, img_2, img_3, img_4, img_5]
    
    
    #stich the images together 
    stich_images(homography_12, homography_23, homography_34, homography_45, 
                 img_list, 'my_LM')
    stich_images(H_12, H_23, H_34, H_45, img_list, 'my_nonLM')

    
if __name__=="__main__":
    main()