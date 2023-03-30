# importing the required libraries
import cv2
import numpy as np

# reading image in grayscale
img = cv2.imread(r"D:\Courses\Python\book.jpeg", cv2.IMREAD_GRAYSCALE)

print(img.shape)
# initializing web cam
cap = cv2.VideoCapture(0)
# creating the SIFT algorithm
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp_image, desc_image =sift.detectAndCompute(img, None)

# initializing the dictionary
index_params = dict(algorithm = 0, trees = 4)
search_params = dict()

# by using Flann Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)
# reading the frame
_, frame = cap.read()

# converting the frame into grayscale
grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with SIFT
kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

# finding nearest match with KNN algorithm
matches= flann.knnMatch(desc_image, desc_grayframe, k=2)

# initialize list to keep track of only good points
good_points=[]
for m, n in matches:
    #append the points according
    #to distance of descriptors
    if(m.distance < 0.6*n.distance):
        good_points.append(m)
# maintaining list of index of descriptors
# in query descriptors
query_pts = np.float32([kp_image[m.queryIdx]
                .pt for m in good_points]).reshape(-1, 1, 2)

# maintaining list of index of descriptors
# in train descriptors
train_pts = np.float32([kp_grayframe[m.trainIdx]
                .pt for m in good_points]).reshape(-1, 1, 2)

# finding perspective transformation
# between two planes
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

# ravel function returns
# contiguous flattened array
matches_mask = mask.ravel().tolist()
# initializing height and width of the image
h, w = img.shape

# saving all points in pts
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

# applying perspective algorithm
dst = cv2.perspectiveTransform(pts, matrix)
# using drawing function for the frame
homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
# showing the final output
# with homography
img = cv2.resize(img, (561, 800))
cv2.imshow("input image", img)
cv2.imshow("Homography", homography)
cv2.waitKey(0)
cv2.destroyAllWindows()
