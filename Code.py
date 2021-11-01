import numpy as np
import cv2
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import cos, sin


window_size = 3
min_disp = 2
num_disp = 128
# Remember that the disparity must be divisible with 16

stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = -1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

fast = cv2.FastFeatureDetector_create()

CamMatrix = np.asarray([[721.5377, 0.0, 609.5593, 0],
             [ 0.0, 721.5377, 172.8540, 0],
             [ 0.0, 0.0, 1.0, 0]])


fMetric = 0.004 
# pixelSize = fMetric/CamMatrix[0,0] 
fPixels = CamMatrix[0,0]
b = 0.54 

def compute_blur(img):
    return cv2.bilateralFilter(img, 5, 20, 20)

def compute_disparity(im1, im2):
    disparity = stereo.compute(im1, im2).astype(np.float32)
    disparity = np.divide(disparity, 16.0)
    cv2.imwrite('disparity.png',disparity)
    return disparity

def keypoint_extraction(img, disparity):
    #size of descriptor
    m = 7
    m = int(m/2) 
    keypoints = fast.detect(img)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[0:300]

    # Remove kpts with unknow disparity (=15)
    list_of_bad_kp = []
    for i, kpt in enumerate(keypoints):
        kpt_x = int(kpt.pt[0])
        kpt_y = int(kpt.pt[1])
        if (disparity[kpt_y][kpt_x] == 15):
            list_of_bad_kp.append(i)
    keypoints = np.delete(keypoints,list_of_bad_kp)
    descriptors = []
    for i, kpt in enumerate(keypoints):
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
            SAD = np.sum(np.abs(np.subtract(d1,d2)))
            S[i][j] = SAD
    return S

def match_features(S):
    #index of minimum value for each fa
    fa_matches = np.zeros(len(S))
    #index of minimum value for each fb
    fb_matches = np.zeros(len(S[0]))
    matches = []
    for i in range(len(S)):
        fa_matches[i] = np.argmin(S[i])
    for i in range(len(S[0])):
        fb_matches[i] = np.argmin(S[:,i])

    for i,fb_bar in enumerate(fa_matches):
        fb_bar = int(fb_bar)
        if fb_matches[fb_bar] == i:
            matches.append([i,fb_bar])
    return matches

def get_coordinate(kp, disparity_map):
    depth_map = (b*fPixels)/disparity_map 
    matrix = CamMatrix
    u = int(kp.pt[0])
    v = int(kp.pt[1])
    depth = depth_map[v][u]
    
    X = depth*(u-matrix[0][2])/(matrix[0][0])
    Y = depth*(v-matrix[1][2])/(matrix[1][1])
    Z = depth
    return [X,Y,Z]

def compute_consistency(matches, Da, Db):
    treshold = 3
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

def compute_Q(W):
    matchesConsistencyScore = []
    # List of indexes of matches that we are going to keep
    Q = []
    for i in range(len(W)):
        matchesConsistencyScore.append(sum(W[i]))
    # clique initialisation
    Q.append(np.argmax(matchesConsistencyScore))

    candidates = np.arange(len(W))
    
    while True:
        # Find set of matches compatibles with the matches in the clique
        for q in Q:
            subCandidates = []
            for i in range(len(W)):
                if W[q][i]==1 and not i in Q and not i in subCandidates:
                    subCandidates.append(i)
            candidates = np.intersect1d(candidates,subCandidates)
        if len(candidates) == 0:
            break
        #score of candidates
        candidatesScore = [matchesConsistencyScore[i] for i in candidates]
        # find the index of the best candidate
        bestCandidateIndex = np.argmax(candidatesScore)
        # add this candidate to Q
        Q.append(candidates[bestCandidateIndex])
    
    return Q


def estimateMotion(params, imPtsa, imPtsb, worldsPtsa, worldsPtsb, CamMatrix):
    motionMatrix = np.eye(4)
    #https://mathworld.wolfram.com/EulerAngles.html
    theta = params[0]
    psi = params[1]
    phi = params[2]

    D = [[cos(phi), sin(phi), 0],
         [-sin(phi), cos(phi), 0],
         [0, 0, 1]]

    C = [[cos(theta), 0, sin(theta)],
         [0, 1, 0],
         [-sin(theta), 0, cos(theta)]]

    B = [ [1, 0, 0],
        [0, cos(psi), sin(psi)],
         [0, -sin(psi), cos(psi)]]

    motionMatrix[:3,:3] = np.matmul(D,np.matmul(C,B))
    
    motionMatrix[0,3] = params[3]
    motionMatrix[1,3] = params[4]
    motionMatrix[2,3] = params[5]
    Pdelt = np.matmul(CamMatrix,motionMatrix)
    Pdeltinv = np.matmul(CamMatrix, np.linalg.inv(motionMatrix))
    error1 = np.zeros((len(imPtsa),3))
    error2 = np.zeros((len(imPtsa),3))
    for i in range(len(imPtsa)):
        Pred1 = np.matmul(Pdelt, np.asarray(worldsPtsb[i]).reshape(-1,1))
        Pred1 /= Pred1[-1]
        Pred2 = np.matmul(Pdeltinv, np.asarray(worldsPtsa[i]).reshape(-1,1))
        Pred2 /= Pred2[-1]
        err1 = np.asarray(imPtsa[i]).reshape(-1,1) - Pred1
        err2 = np.asarray(imPtsb[i]).reshape(-1,1) - Pred2
        error1[i,:] = np.squeeze(err1)
        error2[i,:] = np.squeeze(err2)

        residual = np.vstack((error1,error2))
    #print(residual.flatten())
    return residual.flatten()

#Download images, they are already rectified
Ja_L = cv2.imread('2011_09_26/image_00/data/0000000075.png', 0)  # 0 flag returns a grayscale image
Ja_R = cv2.imread('2011_09_26/image_01/data/0000000075.png', 0)

Jb_L = cv2.imread('2011_09_26/image_00/data/0000000076.png', 0)
Jb_R = cv2.imread('2011_09_26/image_01/data/0000000076.png', 0)


# Pre-filtering
Ja_L = compute_blur(Ja_L)
Ja_R = compute_blur(Ja_R)
Jb_L = compute_blur(Jb_L)
Jb_R = compute_blur(Jb_R)


# Disparity images
Da = compute_disparity(Ja_L, Ja_R)
Db = compute_disparity(Jb_L, Jb_R)

##plt.figure("StereoBM disparity map")
##plt.imshow(Da, 'jet', vmin=0, vmax=20)
##plt.show()

depth = (b*fPixels)/Da 
cv2.imwrite('depth.png',depth)


# Detect keypoints of each image and get descriptors
Kpa, Dsa = keypoint_extraction(Ja_L, Da)
Kpb, Dsb = keypoint_extraction(Jb_L, Db)

# Get the score matrix S
S = compute_S_matrix(Dsa, Dsb)

#Match the features according to the score matrix
matchesIndexes = match_features(S)

# Construct the matches matrix full of features instead of feature's indexes
matches = []
for match in matchesIndexes:
    matches.append([Kpa[match[0]], Kpb[match[1]]])

# Compute consistency matrix :
W = compute_consistency(matches, Da, Db)
np.savetxt('W.csv', W, delimiter=',')

# Compute Q, set of matches in the inlier
Q = compute_Q(W)
# Q is the set of indexes of the matches that we are going to keep, let's exctract those matches
inlierMatches = [matches[i] for i in Q]

## = np.asarray(inlierMatches)[:,0]
## = np.asarray(inlierMatches)[:,1]

imPtsa = []
imPtsb = []
worldsPtsa = []
worldsPtsb = []

output = Ja_L
kpaa=np.asarray(inlierMatches)[:,0]
output = cv2.drawKeypoints(Ja_L, kpaa, output)
output2 = Jb_L
kpbb=np.asarray(inlierMatches)[:,1]
output2 = cv2.drawKeypoints(Jb_L, kpbb, output2)
cv2.imwrite('test.png', output)
cv2.imwrite('test2.png', output2)

for i in range(len(inlierMatches)):
##    imPtsa.append([inlierMatches[i][0].pt[0]-621, inlierMatches[i][0].pt[1]-187.5])
##    imPtsa[i].append(1)
##    print(imPtsa[i])
##    #1242 × 375
##    #621 x 187.5
##
    imPtsa.append([inlierMatches[i][0].pt[0], inlierMatches[i][0].pt[1]])
    imPtsa[i].append(1)
    imPtsb.append([inlierMatches[i][1].pt[0], inlierMatches[i][1].pt[1]])
    imPtsb[i].append(1)
    worldsPtsa.append(get_coordinate(inlierMatches[i][0], Da))
    worldsPtsa[i].append(1)
    worldsPtsb.append(get_coordinate(inlierMatches[i][1], Db))
    worldsPtsb[i].append(1)

##    imPtsa.append([matches[i][0].pt[0], matches[i][0].pt[1]])
##    imPtsa[i].append(1)
##    imPtsb.append([matches[i][1].pt[0], matches[i][1].pt[1]])
##    imPtsb[i].append(1)
##    worldsPtsa.append(get_coordinate(matches[i][0], Da))
##    worldsPtsa[i].append(1)
##    worldsPtsb.append(get_coordinate(matches[i][1], Db))
##    worldsPtsb[i].append(1)

x0 = np.array([0,0,0,0,0,1])
optRes = least_squares(estimateMotion, x0, method='lm',
                       args=(imPtsa, imPtsb, worldsPtsa, worldsPtsb, CamMatrix))

'''
keypoints_fail = matches[-2]
outputA = Ja_L
outputA = cv2.drawKeypoints(Ja_L, [keypoints_fail[0]], outputA, (255,0,0))
outputB = Jb_L
outputB=cv2.drawKeypoints(Jb_L, [keypoints_fail[1]], outputB, (0,255,0))

output = np.concatenate((outputA, outputB) , axis=0)   #cv2.vconcat(outputA, outputB)
cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()



im_L = cv2.hconcat([Ja_L, Jb_L])

for match in matchesIndexes:
    Point_a = Kpa[match[0]].pt
    Point_b = Kpb[match[1]].pt
    cv2.line(im_L, (int(Point_a[0]),int(Point_a[1])),
             (int(Point_b[0] + 1242),int(Point_b[1])),
             (random.randrange(255), random.randrange(255), random.randrange(255)))
    
cv2.imshow('output', im_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

