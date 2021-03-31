import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_harris_corners(img):
# read image and convert to gray scale，then translate the type to float64
# img_gray = cv2.imread('testdata/ex.png', 0).astype('float64')

    ### step1，smooth the image by Gaussian kernel
    # gaussian blur，remove some noise
    blur = cv2.GaussianBlur(img, (3,3), 1.5)

    ### step2，Calculate Ix, Iy
    # set the kernel for function of filter2D 
    x_kernel = np.array((
        [[1., 0., -1.]] 
    ), dtype='float64')
    y_kernel = np.array((
        [[1.], [0.],[-1.]]), dtype='float64')

    # using filter2D to compute Ix and Iy 
    Ix = cv2.filter2D(blur, -1, kernel = x_kernel)
    Iy = cv2.filter2D(blur, -1, kernel = y_kernel)

    ### step3，Compute Ixx, Ixy, Iyy
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    ### step4，Compute Sxx, Sxy, Syy
    Sxx = cv2.GaussianBlur(Ixx, (3,3), 1)
    Syy = cv2.GaussianBlur(Iyy, (3,3), 1)
    Sxy = cv2.GaussianBlur(Ixy, (3,3), 1)

    ### step5，Compute the det and trace of matrix M
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy

    ### step6，Compute the response of the detector
    response = det / (trace+10**(-12))
    return response

### find the detected positions
def post_processing(response, threshold):
    # thresholding response
    response[response < threshold] = 0

    # record the position of candidate
    p_list = np.array(np.argwhere(response > threshold).tolist())

    window_size = 5
    # padding 
    padded_R = cv2.copyMakeBorder(response, window_size//2, window_size//2, window_size//2, window_size//2, cv2.BORDER_CONSTANT, value = 0)

    # find local maximum
    # if there is any neighbour larger than value of center，value of the position will be 0
    for pos in p_list:
        sign_value = np.sign(
            padded_R[pos[0]:pos[0] + window_size, pos[1]:pos[1]+window_size] - padded_R[pos[0] + window_size//2, pos[1] + window_size//2]
        )
        # 1 present neighbour's value larger than center's 
        if 1 in sign_value:
            response[pos[0],pos[1]] = 0
    local_max = np.argwhere(response > 0).tolist()
    local_max = sorted(local_max)
    return local_max

def adding_point_to_oriImg(img, local_max):
    detected_img = img.copy()
    for item in local_max:
        detected_img[item[0],item[1]] = [0, 0, 255]
    return detected_img

def main():
    img = cv2.imread('lina.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    response = detect_harris_corners(img_gray)
    corners_point = post_processing(response, 100)
    detected_img = adding_point_to_oriImg(img, corners_point)

    cv2.namedWindow('img', 0)
    cv2.imshow('img', detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





