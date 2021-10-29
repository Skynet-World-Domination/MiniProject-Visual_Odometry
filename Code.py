import numpy as np
import cv2
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt



stereo = cv2.StereoBM_create(numDisparities=192, blockSize=31)
fast = cv2.FastFeatureDetector_create()

CamMatrix = np.asarray([[721.5377, 0.0, 609.5593],
             [ 0.0, 721.5377, 172.8540],
             [ 0.0, 0.0, 1.0]])

fMetric = 0.004 # 
pixelSize = fMetric/CamMatrix[0,0] # 
fPixels = CamMatrix[0,0]
b = 0.54 # 

def compute_blur(img):
    return cv2.bilateralFilter(img, 3, 10, 10)

def compute_disparity(im1, im2):
    cv2.imwrite('disparity.png',stereo.compute(im1, im2))
    return stereo.compute(im1, im2)
    # disparity = stereo.compute(im1, im2)
    # return np.divide(disparity, 16.0)

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

def get_coordinate(kp, disparity_map):

    depth_map = (b*fPixels)/disparity_map 
    plt.figure("StereoBM disparity map")
    plt.imshow(depth_map, 'jet', vmin=0, vmax=20)

    matrix = CamMatrix
    u = int(kp.pt[1])
    v = int(kp.pt[0])
    depth = depth_map[u][v]
    
    X = depth*(u-matrix[0][2])/(matrix[0][0])
    Y = depth*(v-matrix[1][2])/(matrix[1][1])
    Z = depth
    return [X,Y,Z]

def compute_consistency(matches, Da, Db):
    treshold = 1
    W = np.zeros((len(matches),len(matches)))
    
    for i, match1 in enumerate(matches):
        kpa1 = match1[0]
        kpb1 = match1[1]
        Wa1 = get_coordinate(kpa1, Da)
        Wb1 = get_coordinate(kpb1, Db)
        for j, match2 in enumerate(matches):
            kpa2 = match2[0]
            kpb2 = match2[1]
            Wa2 = get_coordinate(kpa2, Da)
            Wb2 = get_coordinate(kpb2, Db)

            consistency = abs(distance.euclidean(Wa1, Wa2) - distance.euclidean(Wb1, Wb2))
            #print(i,j)
            #print(consistency)

            if consistency < treshold:
                W[i][j] = 1
    return W.astype(int)

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

depth = (b*fPixels)/Da 
plt.figure("StereoBM disparity map")
plt.imshow(depth, 'jet', vmin=0, vmax=20)

# Detect keypoints of each image
Kpa, Dsa = keypoint_extraction(Ja_L)
Kpb, Dsb = keypoint_extraction(Jb_L)

S = compute_S_matrix(Dsa, Dsb)
matchesIndexes = match_features(S)


# Construct the matches matrix full of features instead of feature's indexes
matches = []

for match in matchesIndexes:
    matches.append([Kpa[match[0]], Kpb[match[1]]])


# Compute consistency matrix :


#Get real world coordinates of each features in the matches

W = compute_consistency(matches, Da, Db)
np.savetxt('W.csv', W, delimiter=',')

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

