import cv2
import numpy as np
img1 = cv2.imread("Resources/Image1.jpg")
# img2 = cv2.imread("ImagesTrain/Kinect.jpg",0)
img2 = cv2.imread("Resources/Image2.jpg")
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
imgKp1 = cv2.drawKeypoints(img1,kp1,None)
imgKp2 = cv2.drawKeypoints(img2,kp2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [ ]
for m, n in matches:
  if m.distance < 0.75 * n.distance:
  good.append([m])
print(len(good))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("Features in Image 1",imgKp1)
cv2.imshow("Features in Image 2",imgKp2)
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Feature Matching", img3)
cv2.waitKey(0)

#Disparity Mapping
import cv2
from matplotlib import pyplot as plt
frameWidth = 480
frameHeight = 480
imgL = cv2.imread("Resources/Left.png",0)
imgR = cv2.imread("Resources/Right.png",0)
imgL = cv2.resize(imgL, (frameWidth, frameHeight))
imgR = cv2.resize(imgR, (frameWidth, frameHeight))
stereo = cv2.StereoBM_create(numDisparities = 32,blockSize = 5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,"gray")
plt.show()

#Point Cloud
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
  verts = verts.reshape(-1, 3)
  colors = colors.reshape(-1, 3)
  verts = np.hstack([verts, colors])
  with open(fn, &#39;wb&#39;) as f:
    f.write((ply_header % dict(vert_num=len(verts))).encode(&#39;utf-8&#39;))
    np.savetxt(f, verts, fmt=&#39;%f %f %f %d %d %d &#39;)

def main():
  print("loading images...")
  #imgL = cv.pyrDown(cv.imread(cv.samples.findFile("aloeL.jpg"))) #
  downscale images for faster processing
  #imgR = cv.pyrDown(cv.imread(cv.samples.findFile("aloeR.jpg")))
  frameWidth = 480
  frameHeight = 480
  imgL = cv.imread("Resources/LEFT1.png",0)
  imgR = cv.imread("Resources/RIGHT1.png",0)
  imgL = cv.resize(imgL, (frameWidth, frameHeight))
  imgR = cv.resize(imgR, (frameWidth, frameHeight))
  # disparity range is tuned for "aloe&#39; image pair
  window_size = 3
  min_disp = 16
  num_disp = 112-min_disp
  stereo = cv.StereoSGBM_create(minDisparity = min_disp,
  numDisparities = num_disp,
  blockSize = 16,
  P1 = 8*3*window_size**2,
  P2 = 32*3*window_size**2,
  disp12MaxDiff = 1,
  uniquenessRatio = 10,
  speckleWindowSize = 100,
  speckleRange = 32
  )
  print("computing disparity...")
  disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
  print("generating 3d point cloud...",)
  h, w = imgL.shape[:2]
  f = 0.8*w # guess for focal length
  Q = np.float32([[1, 0, 0, -0.5*w],
  [0,-1, 0, 0.5*h], # turn points 180 deg around x-axis,
  [0, 0, 0, -f], # so that y-axis looks up
  [0, 0, 1, 0]])
  points = cv.reprojectImageTo3D(disp, Q)
  colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
  mask = disp > disp.min()
  out_points = points[mask]
  out_colors = colors[mask]
  out_fn = 'out1.ply'
  write_ply(out_fn, out_points, out_colors)
  print('%s saved&' % out_fn)
  cv.imshow('Left', imgL)
  cv.imshow('Right', imgR)
  cv.imshow('disparity', (disp-min_disp)/num_disp)
  cv.waitKey()
  print('Done')

if __name__ == '__main__':
  print(__doc__)
  main()
  cv.destroyAllWindows()
