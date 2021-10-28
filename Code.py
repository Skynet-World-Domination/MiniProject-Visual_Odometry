import numpy as np
import cv2
import random

stereo = cv2.StereoBM_create(numDisparities=16*7, blockSize=17)
fast = cv2.FastFeatureDetector_create()

CamMatrix = [[721.5377, 0.0, 609.5593], [ 0.0, 721.5377, 172.8540], [ 0.0, 0.0, 1.0]]


def compute_blur(img):
    return cv2.bilateralFilter(img, 3, 10, 10)

def compute_disparity(im1, im2):
    disparity = stereo.compute(im1, im2)
    return np.divide(disparity, 16.0)

def get_coordinate(kp, depth):
    matrix = CamMatrix
    d = depth
    u = int(kp.pt[0])
    v = int(kp.pt[1])
    
    x = depth*(u-matrix[0,2])/(matrix[0,0]*1000)
    y = depth*(v-matrix[1,2])/(matrix[1,1]*1000)
    z = depth/1000

    return [X,Y,Z]

def keypoint_extraction(img):
    #size of descriptor
    m = 7
    m = int(m/2) 
    keypoints = fast.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[0:100]
    descriptors = []
    for kpt in keypoints:
        kpt_x = int(kpt.pt[0])
        kpt_y = int(kpt.pt[1])
        descriptor = img[kpt_y-m:kpt_y+m+1, kpt_x-m:kpt_x+m+1]
        descriptor = descriptor.ravel()
        descriptor = np.delete(descriptor, 24)
        descriptors.append(descriptor)
    return keypoints, descriptors

def compute_S_matrix(des1, des2):
    S = np.zeros((len(des1), len(des2)))
    for i,d1 in enumerate(des1):
        for j, d2 in enumerate(des2):
            SAD = np.sum(np.abs(np.subtract(d1,d2,dtype=np.float)))
            S[i][j] = SAD
    return S

def match_features(S):
    #indx of minimum value for each fa
    fa_matches = np.zeros(len(S))
    #indx of minimum value for each fb
    fb_matches = np.zeros(len(S))
    matches = []
    for i in range(len(S)):
        fa_matches[i] = np.argmin(S[i])
        fb_matches[i] = np.argmin(S[:,i])

    for i,fb_bar in enumerate(fa_matches):
        fb_bar = int(fb_bar)
        if fb_matches[fb_bar] == i:
            matches.append([i,fb_bar])
    return matches
    

#Download images, they are already rectified
Ja_L = cv2.imread('2011_09_26/image_00/data/0000000000.png', 0)  # 0 flag returns a grayscale image
Ja_R = cv2.imread('2011_09_26/image_01/data/0000000000.png', 0)

Jb_L = cv2.imread('2011_09_26/image_00/data/0000000001.png', 0)
Jb_R = cv2.imread('2011_09_26/image_01/data/0000000001.png', 0)

'''
# Pre-filtering
Ja_L = compute_blur(Ja_L)
Ja_R = compute_blur(Ja_R)
Jb_L = compute_blur(Jb_L)
Jb_R = compute_blur(Jb_R)
'''

# Disparity images
Da = compute_disparity(Ja_L, Ja_R)
Db = compute_disparity(Jb_L, Jb_R)

# Detect keypoints of each image
Kpa_L, Dsa_L = keypoint_extraction(Ja_L)
Kpa_R, Dsa_R = keypoint_extraction(Ja_R)
Kpb_L, Dsb_L = keypoint_extraction(Jb_L)
Kpb_R, Dsb_R = keypoint_extraction(Jb_R)

S_L = compute_S_matrix(Dsa_L, Dsb_L)
matches_L_indexes = match_features(S_L)

S_R = compute_S_matrix(Dsa_R, Dsb_R)
matches_R_indexes = match_features(S_R)


# Construct the matches matrix full of features instead of feature's indexes
matches_R = []
matches_L = []
for match in matches_L_indexes:
    matches_L.append([Kpa_L[match[0]], Kpb_L[match[1]]])

for match in matches_R_indexes:
    matches_R.append([Kpa_R[match[0]], Kpb_R[match[1]]])


#Get real world coordinates of each features in the matches




'''
im_L = cv2.hconcat([Ja_L, Jb_L])

for match in matches_L_indexes:
    Point_a = Kpa_L[match[0]].pt
    Point_b = Kpb_L[match[1]].pt
    cv2.line(im_L, (int(Point_a[0]),int(Point_a[1])),
             (int(Point_b[0] + 1242),int(Point_b[1])),
             (random.randrange(255), random.randrange(255), random.randrange(255)))

im_R = cv2.hconcat([Ja_R, Jb_R])

for match in matches_R_indexes:
    Point_a = Kpa_R[match[0]].pt
    Point_b = Kpb_R[match[1]].pt
    cv2.line(im_R, (int(Point_a[0]),int(Point_a[1])),
             (int(Point_b[0] + 1242),int(Point_b[1])),
             (random.randrange(255), random.randrange(255), random.randrange(255)))

    
output = cv2.vconcat([im_R, im_L])
'''

cv2.imshow('ImT1_L', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

