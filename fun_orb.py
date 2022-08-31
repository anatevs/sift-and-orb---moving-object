# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:33:43 2021

@author: User
"""

"""version of fun_orb_v1_1
   Trying to use a smoothing in an "edge" region (that we are using to correctly
   working of ORB)

"""

def orb_watching(frame, imgs0, kps0, dess0, xy_imgs0, delt_xy, edge_width, test_mode=0):
    
    # frame - current frame
    # imgs0 - list of sample images of objects (with edge_width)
    # xy_imgs0 - list of coordinates of sample images from a previous frame for every object
    # delt_xy - list of sizes of ROI for every object
    # edge_width - width of a frame added around the object for a normal work of ORB algorithm

    import cv2
    import numpy as np
    
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #into gray scale
    
    if test_mode == 1:
        im_show_matches = 0 #if 0 there will be shown all matches
        isshow_kp = 0
    
    
    
    rng_smpls = range(0, len(imgs0))
    
    out_coord = [[] for i in rng_smpls]
    out_img_sample = [[] for i in rng_smpls]
    out_kp = [[] for i in rng_smpls]
    out_des = [[] for i in rng_smpls]
    out_delt_xy = [[] for i in rng_smpls]
    img_match_frame = []
    for i_obj, img0 in enumerate(imgs0):
        
        # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #into gray scale
        # !!!!!!!!!!!!!!!!!! add cheking of the borders of frame for img1 with delta!!!!
        
        # we will use an abs from delt_xy because in this version we using delt_xy as a region around the object (not as a directed shift)
        d_xy_abs = np.abs(delt_xy[i_obj])
        xy_im1_roi = [xy_imgs0[i_obj][0]-d_xy_abs[0], xy_imgs0[i_obj][1]-d_xy_abs[1], 
                      xy_imgs0[i_obj][2]+d_xy_abs[0], xy_imgs0[i_obj][3]+d_xy_abs[1]]
        
        # a version with delt_xy as a directed shift
        # xy_im1_roi = [xy_imgs0[i_obj][0]+delt_xy[i_obj][0], xy_imgs0[i_obj][1]+delt_xy[i_obj][1], 
        #               xy_imgs0[i_obj][2]+delt_xy[i_obj][0], xy_imgs0[i_obj][3]+delt_xy[i_obj][1]]
        
        img1 = frame[xy_im1_roi[1] : xy_im1_roi[3], xy_im1_roi[0] : xy_im1_roi[2]] #pattern to search matches
        
        # if test_mode == 1:
        #     cv2.imshow('im0', img0)
        #     cv2.waitKey()
        #     cv2.imshow('im1', img1)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        
        
        # ORB creating
        orb = cv2.ORB_create()
        
        # calculating keypoints and descriptors of sample image
        kp0 = kps0[i_obj]
        des0 = dess0[i_obj]
        
        
        # creating a partly smooth sample image
        # at first making smooth sample
        n_kern = 5
        img1_smooth = cv2.blur(img1, (n_kern, n_kern))
        # size of img1
        size_im1 = np.shape(img1)
        # entering a part with an object in a smoothed image
        img1_smooth[edge_width : size_im1[0]-edge_width, edge_width : size_im1[1]-edge_width] = (
            img1[edge_width : size_im1[0]-edge_width, edge_width : size_im1[1]-edge_width])
        
        img1_orig = img1 #just for checking and comparing
        img1 = img1_smooth
        # calculating keypoints and descriptors of current image
        kp1, des1 = orb.detectAndCompute(img1, None)
        
        
        if isshow_kp == 1:
            from my_standart_modules import show_gr
            img0_kp = cv2.drawKeypoints(img0, kp0, img0)
            img1_kp = cv2.drawKeypoints(img1, kp1, img1)
            show_gr(img0_kp, im_sc=3)
            show_gr(img1_kp, im_sc=3)
        
        out_img_sample[i_obj] = img1
        out_kp[i_obj] = kp1
        out_des[i_obj] = des1
        
        if (kp0 != []) & (kp1 != []): #some keypoints were found in a sample and in a current frame fragment
            # create BFM matcher object
            bf_orb = cv2.BFMatcher()
            # BFM matching for orb:
            matches = bf_orb.match(des0, des1)
            
            if matches:        
                #  sorting matches in order of it's distances
                matches = sorted(matches, key = lambda x: x.distance)
                
                # picking only matches with distances near minimum distance
                matches_good = []
                thr_good = (1 - 0.6)*(matches[len(matches)-1].distance -  matches[0].distance) + matches[0].distance
                
                for m in matches:
                    if m.distance <= thr_good:
                        matches_good.append(m)            
                
                if matches_good:
                    # indexes of good matches in the kp1 (i.e. train image keypoints)
                    idxs_im1 = [matches_good[i].trainIdx for i in range(0, len(matches_good))]
                    # coordinates of good matched kp1
                    xy_im1 = np.array([kp1[i].pt for i in idxs_im1])
                    
                    # coordinates of object with region around it
                    xy_edge = (np.array([(min(xy_im1[:, 0])) - edge_width, min(xy_im1[:, 1]) - edge_width, max(xy_im1[:, 0])+edge_width, max(xy_im1[:, 1])+edge_width]) + xy_im1_roi[0:2]*2).astype(int) #coordinates of the found object with region (edge_width thikness) around it
                    
                    
                    # !!!!!!!! this block is not very good; maybe it'll be better just detect an object coordinates via labeling operations
                    #incriesing an edge_width as keypoints not always detected in a edge of object
                    edge_width_plus = edge_width + 5
                    xy_edge = (np.array([(min(xy_im1[:, 0])) - edge_width_plus, min(xy_im1[:, 1]) - edge_width_plus, max(xy_im1[:, 0])+edge_width_plus, max(xy_im1[:, 1])+edge_width_plus]) + xy_im1_roi[0:2]*2).astype(int)
                    # !!!!!!!! end of not good block
                    
                    
                    if (xy_edge[0] >= 0) * (xy_edge[1]>=0) * ((frame.shape[1] - xy_edge[2])>=0) * ((frame.shape[0] - xy_edge[3]) >= 0): #if the sample with region around it is in the frame borders
                        out_coord[i_obj] = xy_edge #coordinates of the found object with region (edge_width thikness) around it
                        out_delt_xy[i_obj] = [(xy_edge[0] - xy_imgs0[i_obj][0])*3, (xy_edge[1] - xy_imgs0[i_obj][1])*3]
            
                                
            if test_mode == 1:
                if im_show_matches == 0:
                    im_show_matches = len(matches_good)
                                
                # making keypoints in a frame coordinate system
                # calculating kp1 one more time because I didn't find a way to copy it as a full duplicate (copy.deepcopy doesn't work with keypoints datatype)
                kp1_frame = orb.detect(img1, None)
                # changing a coordinates parameter of keypoints
                for i_pt in range(0, len(kp1_frame)):
                    kp1_frame[i_pt].pt = (kp1[i_pt].pt[0] + xy_im1_roi[0], kp1[i_pt].pt[1] + xy_im1_roi[1])
                
                img_match_frame = cv2.drawMatches(img0, kp0, frame, kp1_frame, matches_good[:im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                
        return out_img_sample, out_kp, out_des, out_coord, out_delt_xy, img_match_frame
