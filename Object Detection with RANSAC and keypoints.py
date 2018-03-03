import cv2
import numpy as np
import math
from random import randint


def read_matches(fname):
    fin = open(fname)
    f = fin.read()
    fin.close()
    src_points = []
    dst_points = []
    for line in f.strip().split("\n"):
        points = line.strip().split(",")
        p1 = points[0].strip().split(" ")
        p2 = points[1].strip().split(" ")
        src_points.append((float(p1[0]),float(p1[1])))
        dst_points.append((float(p2[0]),float(p2[1])))
    return src_points,dst_points

def euc_distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def average(points): #function of dlt
    x = 0
    y = 0
    for i in range(len(points)):
        x = x + points[i][0]
        y = y + points[i][1]
    average_x = x / len(points)
    average_y = y / len(points)
    return average_x,average_y

def move(points,avgx,avgy): #function of dlt
    new_points = []
    for i in range(len(points)):
        x = points[i][0] - avgx
        y = points[i][1] - avgy
        new_points.append((x,y))
    return new_points

def scale(points): #function of dlt
	total_distance = 0
	for i in range(len(points)):
		total_distance = total_distance + euc_distance(points[i],(0,0))
		scale = (2**(0.5))/((total_distance)/20)
	return scale

def construct_a(p1,p2,s1,s2,avg_src_x,avg_src_y,avg_dst_x,avg_dst_y): #function of dlt
	a = []
	t = np.dot(s1,[[1,0,-(avg_src_x)],[0,1,-(avg_src_y)],[0,0,1/s1]])
	t_prime = np.dot(s2,[[1,0,-(avg_dst_x)],[0,1,-(avg_dst_y)],[0,0,1/s2]])
	for i in range(len(p1)):
		x = np.dot([p1[i][0],p1[i][1],1],t)
		xp = np.dot([p2[i][0],p2[i][1],1],t_prime)
		temp = [[0,0,0,-(x[0]),-(x[1]),-1,(xp[1]*x[0]),(xp[1]*x[1]),xp[1]],[x[0],x[1],1,0,0,0,-(xp[0]*x[0]),-(xp[0]*x[1]),-(xp[0])]]
		a.append(temp[0])
		a.append(temp[1])
	return a,t,t_prime

def dlt(p1,p2): #main dlt function
	avg_srcx,avg_srcy = average(p1)
	avg_dstx,avg_dsty = average(p2)

	new_points_src = move(p1,avg_srcx,avg_srcy)
	new_points_dst = move(p2,avg_dstx,avg_dsty)

	scale_src = scale(new_points_src)
	scale_dst = scale(new_points_dst)

	a,t,t_prime = construct_a(new_points_src,new_points_dst,scale_src,scale_dst,avg_srcx,avg_srcy,avg_dstx,avg_dsty)
	a = np.array(a)
	w, u, vt = cv2.SVDecomp(a)

	h = vt[-1].reshape(3,3)

	temp = np.dot(h,t)
	inv_t_prime = np.linalg.inv(t_prime)
	unnormalize_h = np.dot(inv_t_prime, temp)
	unnormalize_h = np.array(unnormalize_h)
	return unnormalize_h

def compute_random(src_points,dst_points): #function of ransac
    src_random_corss = []
    dst_random_corss = []
    for i in range(4):
        rnd = randint(0,len(src_points)-1)
        src_random_corss.append(src_points[rnd])
        dst_random_corss.append(dst_points[rnd])
    return src_random_corss,dst_random_corss

def inlier_calculator(p1,p2,h,number_of_inliers,correspondences): #function of ransac
	for points in range(len(p1)):
		x_temp = np.dot(h,np.array([p1[points][0],p1[points][1],1]))
		x_new = np.array([x_temp[0]/x_temp[2],x_temp[1]/x_temp[2]])
		for k in range(len(p2)):
			if  euc_distance(x_new,p2[k]) <= 3:
				number_of_inliers = number_of_inliers + 1
				correspondences.append(points)
				break
	return correspondences,number_of_inliers

def ransac(p1,p2): #main ransac function
	global N_temp
	max_h = 0
	iterations = 0
	N = 10000
	correspondences = []
	while iterations < N:
		random_corss_src,random_corss_dst = compute_random(p1,p2)
		h_from_dlt = dlt(random_corss_src,random_corss_dst)
		inlier_ratio = 0
		inlier_ratio_temp = 0
		number_of_inliers = 0
		for points in range(len(p1)):
			x_temp = np.dot(h_from_dlt,np.array([p1[points][0],p1[points][1],1]))
			x_new = np.array([x_temp[0]/x_temp[2],x_temp[1]/x_temp[2]])
			for k in range(len(p2)):
				if  euc_distance(x_new,p2[k]) <= 3:
					number_of_inliers = number_of_inliers + 1
					correspondences.append(points)
					break
		inlier_ratio_temp = 100 * number_of_inliers / (len(p1))
		if inlier_ratio_temp > inlier_ratio:
			inlier_ratio = inlier_ratio_temp
			max_h = h_from_dlt
		print ("Number of inliers = ",number_of_inliers)
		probability = number_of_inliers / (len(p2))
		print ("Probability = ",probability)
		if probability != 0:
			N_temp = np.floor((-2)/np.log(1 - probability**4))
		if N_temp < N :
			N = N_temp
		iterations = iterations + 1
		print ("Iterations = ",iterations)
		print ("N = ",N)
		number_of_inliers = 0
	return max_h,correspondences


img1 = cv2.imread("img1.jpg",0)
img2 = cv2.imread("img2.jpg",0)
orb = cv2.ORB_create(nfeatures = 3000)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

kp_src = []
kp_dst = []
for match in matches:
	img1_idx = match.queryIdx
	img2_idx = match.trainIdx
	(x1,y1) = kp1[img1_idx].pt
	(x2,y2) = kp2[img2_idx].pt
	kp_src.append((x1,y1))
	kp_dst.append((x2,y2))
max_h,inliers = ransac(kp_src,kp_dst)
converge = 0
flag = True
while flag == True:
	len_of_first_inliers = len(inliers)
	inliers_new = []
	src_new = []
	dst_new = []
	for count in range(len(inliers)):
		if (inliers[count] < 258):
			src_new.append(kp_src[inliers[count]])
			dst_new.append(kp_dst[inliers[count]])
	reduced_homography = dlt(src_new,dst_new)
	reduced_homography,inliers = ransac(src_new,dst_new)
	if (len_of_first_inliers-(len(inliers)) < 5):
		converge = converge + 1
	else:
		converge = 0
	if converge == 3:
		flag = False
print ("Homography:\n",reduced_homography)
corner_1 = np.dot(reduced_homography,np.array([175,576,1]))
corner_1_cart = np.array([int(corner_1[0]/corner_1[2]),int(corner_1[1]/corner_1[2])])
corner_2 = np.dot(reduced_homography,np.array([155,470,1]))
corner_2_cart = np.array([int(corner_2[0]/corner_2[2]),int(corner_2[1]/corner_2[2])])
print ("corner1  :\n",corner_1_cart)
print ("corner2  :\n",corner_2_cart)
cv2.rectangle(img2, (corner_1_cart[0],corner_1_cart[1]), (corner_2_cart[0],corner_2_cart[1]), (0, 0, 255), 2)
cv2.imwrite("object detection.png", img2)

""" 
COMMENT1
ORB is basically consist of FAST keypoint detector and BRIEF descriptor with modifications. ORB works on binary descriptors. For keypoint 
detection, FAST does not compute the orientation. However, ORB computes orientation with its own way of calculation. 
For keypoint description, ORB use BRIEF descriptors, BRIEF is not good at computing rotation. ORB does “steer” BRIEF according
to the orientation of keypoints. ORB uses image pyramids for scale invariance and intensity centroid for rotation invariance. ORB uses Harris
corner detection. ORB is an alternative to SIFT.
The SIFT algorithm uses a form of ‘blob’ detection. Unlike feature detectors which use directed image gradients(Harris Corner). 
These kind of detectors use the laplacian of an image, which contains no directional information. This allows detection of scale in addition
to position within the image.
If we compare them, computing Hamming distance is fast compared to Euclidean distance. However, SIFT is more scale invariant. According to
my researches, SIFT is patented. This means you need to pay in order to use it. ORB is free. ORB is faster than SIFT.

COMMENT2
The nature of ORB use BFMatcher. BFMatcher means brute force matcher. For finding correspondences, ORB use Hamming distance. SIFT use 
Euclidean distance.

COMMENT3
That distance threshold determines if it is inlier or outlier. We need to find the rigth balance. Low threshold rejects good data,
high threshold accepts outliers. If we set distance smaller than 3, we can find more accurate inliers. However, this makes us reject
small differences like 2.1. So, it is all about our sample image.

COMMENT4
We do normalization in order to convert a wide range of data to a defined interval that is useful for us. Also, for more accurate estimation
in existance of noise and faster solution.
"""