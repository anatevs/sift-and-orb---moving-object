# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:07:19 2021

@author: User
"""


import cv2
# import numpy as np
# mesure the time duration of completing process
import time

# from my_standart_modules import showing_img
im_show_sc = 3
im_show_matches = 0 #if 0 there will be shown all matches


algorithm_type = 'orb'
fn_n = 'night__'
isshow_kp = 0

fn0 = fn_n + '-pattern.jpg'

img0 = cv2.cvtColor(cv2.imread(fn0), cv2.COLOR_BGR2GRAY)


fn = 'night_20201028_KSU1_014_tele_plane.mp4'
cap = cv2.VideoCapture(fn)

if (cap.isOpened() == False):
    print('error in opening file')

while (cap.isOpened()):
    isrtrn, frame = cap.read()
    
    if isrtrn == True:
        
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if algorithm_type == 'orb':
            tic = time.process_time()
            
            # ORB creating
            orb = cv2.ORB_create()
            
            # calculating keypoints and descriptors of sample image
            kp0, des0 = orb.detectAndCompute(img0, None)
            
            # calculating keypoints and descriptors of current image
            kp1, des1 = orb.detectAndCompute(img1, None)
            
            # create BFM matcher object
            bf_orb = cv2.BFMatcher()
            # BFM matching for orb:
            matches = bf_orb.match(des0, des1)
            
            #  sorting matches in order of it's distances
            matches = sorted(matches, key = lambda x: x.distance)
            
            # picking only matches with distances near minimum distance
            matches_good = []
            thr_good = (1 - 0.6)*(matches[len(matches)-1].distance -  matches[0].distance) + matches[0].distance
            for m in matches:
                if m.distance <= thr_good:
                    matches_good.append(m)            
            
            toc = time.process_time()
            
            if im_show_matches == 0:
                im_show_matches = len(matches_good)
            
            # picture with images and lines between good matches
            img_match = cv2.drawMatches(img0, kp0, img1, kp1, matches_good[ : im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        elif algorithm_type == 'sift':
            tic = time.process_time()
            # SIFT creating and matching
            sift = cv2.SIFT_create()
            
            kp0, des0 = sift.detectAndCompute(img0, None)
            kp1, des1 = sift.detectAndCompute(img1, None)
            
            # BFM matcher
            bf_sift = cv2.BFMatcher()
            # BFM matching for sift:
            matches = bf_sift.knnMatch(des0, des1, k=2)
            
            # Apply ratio test
            matches_good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    matches_good.append([m])
            
            toc = time.process_time()
            
            if im_show_matches == 0:
                im_show_matches = len(matches_good)
            
            # picture with images and lines between good matches
            img_match = cv2.drawMatchesKnn(img0, kp0, img1, kp1, matches_good[ : im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            
            
            idxs_im1 = [matches_good[i].trainIdx for i in range(0, len(matches_good))]
            xy_im1 = [kp1[i].pt for i in idxs_im1]
            
            
            
            
            
            
        # print(f'eof db part-file processing {toc-tic} s')
    
    # # make visible found keypoints in images
    
    # if isshow_kp == 1:
    #     img0 = cv2.drawKeypoints(img0, kp0, img0)
    #     img1 = cv2.drawKeypoints(img1, kp1, img1)
    
    # # showing images with/without keypoints on them
    # showing_img(img0, im_show_sc, cmap='gray')
    # showing_img(img1, im_show_sc, cmap='gray')
    # # showing images with good matches
    # showing_img(img_match, im_show_sc)

        
        
        cv2.imshow(fn, img_match)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()



