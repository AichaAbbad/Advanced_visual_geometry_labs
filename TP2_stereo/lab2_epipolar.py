import numpy as np
import cv2
from numpy import linalg as LA

def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    center = np.mean(points,axis = 0)  # TODO: find center of the set of points by computing mean of x & y
    dist = np.linalg.norm(points-center)  # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    s = np.sqrt(2)/ np.mean(dist) # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
img1 = cv2.imread('./img/chapel00.png', 0)  # queryImage
img2 = cv2.imread('./img/chapel01.png', 0)  # trainImage

# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
normalized_query_kpts = T_query@np.transpose(homogenize(query_kpts))  # TODO: apply T_query to query_kpts to normalize them
normalized_query_kpts = np.transpose(normalized_query_kpts)

T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
normalized_train_kpts = T_train@np.transpose(homogenize(train_kpts))  # TODO: apply T_train to train_kpts to normalize them
normalized_train_kpts = np.transpose(normalized_train_kpts)

# construct homogeneous linear equation to find fundamental matrix
xp1 = normalized_query_kpts[:,0]
yp1 = normalized_query_kpts[:,1]
x1 = normalized_train_kpts[:,0]
y1 = normalized_train_kpts[:,1]
one = normalized_train_kpts[:,2]

print(np.shape(xp1*x1))

A = np.array([[xp1*x1], [xp1*y1], [xp1] , [yp1*x1], [yp1*y1], [yp1],[x1],[y1],[one]]).squeeze() 
A = A.T # TODO: construct A according to Eq.(3) in lab subject
print(np.shape(A))
# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v

U,S,Vh = np.linalg.svd(A)
f = Vh[-1]

# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
U1,S1,Vh1 = np.linalg.svd(F)
print(np.diag(S1))

# hint: set the smallest singular value of F to 0
sigma,W= np.linalg.eig(np.diag(S1))
new_s = np.diag(sigma)
new_s[-1] = 0
# hint: reconstruct F from u, new_s, vh
F = U1@new_s@Vh1
assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

# TODO: de-normlaize F
F = np.transpose(T_query)@F@T_train

# hint: last line of Algorithme 1 in the lab subject
F_gt = np.loadtxt('chapel.00.01.F')
print("******")
print(F - F_gt)

