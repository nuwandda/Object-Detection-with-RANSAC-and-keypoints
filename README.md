# Object-Detection-with-RANSAC-and-keypoints
In this project, I wrote an object detection algorithm with RANSAC


First, we read the given inputs. Then, we need to choose a keypoint algorithm in order to find keypoint detectors and descriptors. With RANSAC, we can find homography and inliers. We got a number that inliers converge. Running RANSAC after and after, we reduce the inliers. Then, it draws a rectangle around the book in the image. This implementation finds the location of book. However, it can not draw rectangle around it. It will be corrected in next days.
