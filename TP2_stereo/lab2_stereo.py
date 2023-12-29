import numpy as np
import cv2
import open3d as o3d

K = np.array([
    [7.215377e2,        0,              6.095593e2],
    [0,                 7.215377e2,     1.728540e2],
    [0,                 0,              1]
])  # intrinsic matrix of camera

Bf = 3.875744e+02  # base line * focal length

# load images
left = cv2.imread('img/left.png', 0)
right = cv2.imread('img/right.png', 0)
left_color = cv2.imread('img/left_color.png')

# compute disparity
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(left, right)  # an image of the same size with "left" (or "right")
r,c = np.shape(disparity)
Z = np.zeros(np.shape(disparity))
pixels = []
# TODO: compute depth of every pixel whose disparity is positive
# hint: assume d is the disparity of pixel (u, v)
# hint: the depth Z of this pixel is Z = Bf / d
for i in range(r):
    for j in range(c):
        if disparity[i][j]>0:
            Z[i][j] = Bf/disparity[i][j]
            pixels.append([i,j,1])

pixels=np.array(pixels)
print(np.shape(Z))
# TODO: compute normalized coordinate of every pixel whose disparity is positive
# hint: the normalized coordinate of pixel [u, v, 1] is K^(-1) @ [u, v, 1]

n = np.transpose(np.linalg.inv(K)@np.transpose(pixels))        

# TODO: compute 3D coordinate of every pixel whose disparity is positive
# hint: 3D coordinate of pixel (u, v) is the product of Z and its normalized cooridnate

all_3d = n * Z[pixels[:,0],pixels[:,1]][:,np.newaxis]
# this is matrix storing 3D coordinate of every pixel whose disparity is positive
# the shape of all_3d is <num_pixels_positive_disparity, 3>
# each row of all_3d is a 3D coordinate [X, Y, Z]
# you need to change the value of all_3d with your computation of 3D coordinate of every pixel whose disparity is positive

# TODO: get color for 3D points
all_color = left_color[pixels[:,0],pixels[:,1]]  # this is matrix storing color of every pixel whose disparity is positive
# TODO: THE ORDER OF all_color IS THE SAME WITH all_3d
# the shape of all_color is <num_pixels_positive_disparity, 3>
# each row of all_color is [R, G, B] value

# normalize all_color
all_color = all_color.astype(np.float32) / 255.0

# Display pointcloud
cloud = o3d.geometry.PointCloud()  # create pointcloud object
cloud.points = o3d.utility.Vector3dVector(all_3d)
cloud.colors = o3d.utility.Vector3dVector(all_color)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])  # create frame object
o3d.visualization.draw_geometries([cloud, mesh_frame])
