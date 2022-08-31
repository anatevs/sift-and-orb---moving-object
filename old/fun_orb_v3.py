# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:51:08 2021

@author: User
"""

"""trying to add around the object a pixrels with mean background signals 
(instead of using background itself) as in v2. But here we use another way to 
detect an object coordinates (not a matched points as it was in v2)"""


def orb_watching(frame, imgs0, xy_imgs0, xy_objs, delt_xy, edge_width, test_mode=0):
    
    # frame - current frame
    # imgs0 - list of sample images of objects (with edge_width)
    # xy_imgs0 - list of coordinates of sample images from a previous frame for every object
    # xy_objs - list of coordinated of objects (exactly objects i.e. without any frame) from a previous frame
    # delt_xy - list of sizes of ROI for every object
    # edge_width - width of a frame added around the object for a normal work of ORB algorithm

    import cv2
    import numpy as np
    import skimage.measure as ski_msr
    
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #into gray scale
    
    if test_mode == 1:
        im_show_matches = 0 #if 0 there will be shown all matches
        isshow_kp = 1
        from my_standart_modules import show_gr
    
    
    
    rng_smpls = range(0, len(imgs0))
    
    out_coord_edge = [[] for i in rng_smpls]
    out_coord_obj = [[] for i in rng_smpls]
    out_img_sample = [[] for i in rng_smpls]
    out_delt_xy = [[] for i in rng_smpls]
    img_match_frame = []
    for i_obj, img0 in enumerate(imgs0):
        
        # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #into gray scale
        # !!!!!!!!!!!!!!!!!! add cheking of the borders of frame for img1 with delta!!!!
        
        # next we will use an abs from delt_xy because in this version we are using delt_xy as a region around the object (not as a directed shift)
        d_xy_abs = np.abs(delt_xy[i_obj])
        # where we suppouse to find the object in a current frame
        xy_im1_roi = [xy_imgs0[i_obj][0]-d_xy_abs[0], xy_imgs0[i_obj][1]-d_xy_abs[1], 
                      xy_imgs0[i_obj][2]+d_xy_abs[0], xy_imgs0[i_obj][3]+d_xy_abs[1]]
        
        # a version with delt_xy as a directed shift
        # xy_im1_roi = [xy_imgs0[i_obj][0]+delt_xy[i_obj][0], xy_imgs0[i_obj][1]+delt_xy[i_obj][1], 
        #               xy_imgs0[i_obj][2]+delt_xy[i_obj][0], xy_imgs0[i_obj][3]+delt_xy[i_obj][1]]
        
        # img1 = frame[xy_im1_roi[1] : xy_im1_roi[3], xy_im1_roi[0] : xy_im1_roi[2]] #pattern to serch matches
        
        obj_in_smpl_xy = xy_objs[i_obj] - np.array([xy_im1_roi[0], xy_im1_roi[1], xy_im1_roi[0], xy_im1_roi[1]])
        
        img1_obj = frame[xy_objs[i_obj][1]:xy_objs[i_obj][3], xy_objs[i_obj][0]:xy_objs[i_obj][2]]
        img1_roi = frame[xy_im1_roi[1] : xy_im1_roi[3], xy_im1_roi[0] : xy_im1_roi[2]]
        img1_smooth = cv2.blur(img1_roi, (n_kern, n_kern))
        
        img1 = img1_smooth
        img1[obj_in_smpl_xy[1]:obj_in_smpl_xy[3], obj_in_smpl_xy[0]:obj_in_smpl_xy[2]] = img1_obj
        
        
        # if test_mode == 1:
        #     cv2.imshow('im0', img0)
        #     cv2.waitKey()
        #     cv2.imshow('im1', img1)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        
        
        # ORB creating
        orb = cv2.ORB_create()
        
        # calculating keypoints and descriptors of sample image
        kp0, des0 = orb.detectAndCompute(img0, None)
        
        # calculating keypoints and descriptors of current image
        kp1, des1 = orb.detectAndCompute(img1, None)
        
        if isshow_kp == 1:
            img0_kp = cv2.drawKeypoints(img0, kp0, img0)
            img1_kp = cv2.drawKeypoints(img1, kp1, img1)
            show_gr(img0_kp, im_sc=5)
            show_gr(img1_kp, im_sc=5)
        
        
        if (kp0 != []) & (kp1 != []): #there was found some keypoints in sample and in a current frame fragment
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
                    # idxs_im1 = [matches_good[i].trainIdx for i in range(0, len(matches_good))]
                    # coordinates of good matched kp1
                    # xy_im1 = np.array([kp1[i].pt for i in idxs_im1])
                    
                    # coordinates of object
                    # xy_obj_i = (np.array([(min(xy_im1[:, 0])), min(xy_im1[:, 1]), max(xy_im1[:, 0]), max(xy_im1[:, 1])]) + xy_im1_roi[0:2]*2).astype(int)
                    
                    
                    
                    
                    
                    
                    
                    lbl_data = ski_msr.label(img1 < 0.7*np.max(img1_obj), connectivity=2)
                    obj_reg_idxs = np.argwhere(lbl_data)
                    # adding xy_roi to make coordinates in a global coord.sys.
                    xy_obj_i = (np.array([(min(obj_reg_idxs[:, 1])), min(obj_reg_idxs[:, 0]), max(obj_reg_idxs[:, 1]) + 1, max(obj_reg_idxs[:, 0]) + 1]) + xy_im1_roi[0:2]*2).astype(int)
                    
                    
        #             img_lbl, num_lbl = ski.measure.label(diff_thr_frame[n_kern : len(diff_thr_frame) - n_kern, n_kern : len(diff_thr_frame[0]) - n_kern], return_num=True, connectivity=connect_bw)
        
        # geom_cents[i] = np.zeros((num_lbl, 2))
        # # count_det = 0
        
        # if i == 1: #1st frame
        #     prev_cents = prev_cents_in
        # elif i >= 2: #from 2nd frame
        #     prev_cents = geom_cents[i-1]
        
        
        # if np.sum(prev_cents) != 0: #in a previous frame were found objects
        #     for i_obj in range(0, num_lbl):                
        #         pixels_obj = np.argwhere(img_lbl)
        #         geom_cents[i][i_obj, :] = [np.mean(pixels_obj[:, 0]), np.mean(pixels_obj[:, 1])]

                    
                    
                    
                    
                    # coordinates of object with region around it
                    # xy_edge = (np.array([(min(xy_im1[:, 0])) - edge_width, min(xy_im1[:, 1]) - edge_width, max(xy_im1[:, 0])+edge_width, max(xy_im1[:, 1])+edge_width]) + xy_im1_roi[0:2]*2).astype(int) #coordinates of the found object with region (edge_width thikness) around it
                    xy_edge = (np.array([-edge_width, -edge_width, edge_width, edge_width]) + xy_obj_i).astype(int)
                    
                    if (xy_edge[0] >= 0) * (xy_edge[1]>=0) * ((frame.shape[1] - xy_edge[2])>=0) * ((frame.shape[0] - xy_edge[3]) >= 0): #if the sample with region around it is in the frame borders
                        out_img_sample[i_obj] = frame[xy_edge[1] : xy_edge[3], xy_edge[0] : xy_edge[2]] #sample of the found object
                        out_coord_edge[i_obj] = xy_edge #coordinates of the found object with region (edge_width thikness) around it
                                                
                        out_delt_xy[i_obj] = [(xy_edge[0] - xy_imgs0[i_obj][0])*3, (xy_edge[1] - xy_imgs0[i_obj][1])*3]
                        out_coord_obj[i_obj] = xy_obj_i + out_delt_xy[i_obj][0:2]*2
            
            
                    # if im_show_matches == 0:
                    #     im_show_matches = len(matches_good)
                    
                    # picture with images and lines between good matches
                    # img_match = cv2.drawMatches(img0, kp0, img1, kp1, matches_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
            if test_mode == 1:
                if im_show_matches == 0:
                    im_show_matches = len(matches_good)
                
                # picture with images and lines between good matches
                # img_match = cv2.drawMatches(img0, kp0, img1, kp1, matches_good[:im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # making keypoints in a frame coordinate system
                kp1_frame = kp1
                for i_pt in range(0, len(kp1_frame)):
                    kp1_frame[i_pt].pt = (kp1[i_pt].pt[0] + xy_im1_roi[0], kp1[i_pt].pt[1] + xy_im1_roi[1])
                
                img_match_frame = cv2.drawMatches(img0, kp0, frame, kp1_frame, matches_good[:im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # cv2.imshow('res', img_match)
                # cv2.imshow('res', img_match_frame)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            
        return out_img_sample, out_coord_edge, out_coord_obj, out_delt_xy, img_match_frame

        # else:
        #     return out_img_sample, out_coord, out_delt_xy
        
            
        

import cv2
import numpy as np

# video
fn_vid = 'video_Planer1.mp4'

cap = cv2.VideoCapture(fn_vid)
if not cap.isOpened():
    print('cant open file')

# obj_xy = [ [280, 245, 330, 290] ]
obj_xy = [ np.array([280+16, 245+15, 330-18, 290-15]) ]
edge_w = 30
smpl_xy = [ obj_xy[0] + np.array([-1, -1, 1, 1])*edge_w ]
obj_in_smpl_xy = obj_xy[0] - np.array([smpl_xy[0][0], smpl_xy[0][1], smpl_xy[0][0], smpl_xy[0][1]])
n_kern = 6



delt_xy = [ [2, 0] ]

obj_xy[0] = obj_xy[0] + delt_xy[0]*2

test_mode = 1

count = 0

while cap.isOpened():
    isrtrn, fr_curr = cap.read()
    fr_curr = cv2.cvtColor(fr_curr, cv2.COLOR_BGR2GRAY)
    
    if isrtrn:
        # block for picking up some exactly frame and some part of it
        # fr_to_count = 2
        # if count == fr_to_count: #2nd frame
        #     from my_standart_modules import showing_img
        #     showing_img(frame, im_sc = 3)
        #     showing_img(frame[250:295, 280:330, :], im_sc = 3)
        # elif count > fr_to_count:
        #     break
        
        # this block with "count" needed when we don't have prev frame info, i.e. we are in a test mode with video file
        count = count + 1
        if count == 1:
            im_in = fr_curr
            smpl_obj = im_in[obj_xy[0][1]:obj_xy[0][3], obj_xy[0][0]:obj_xy[0][2]]
            smpl_src = im_in[smpl_xy[0][1]:smpl_xy[0][3], smpl_xy[0][0]:smpl_xy[0][2]]
            smpl_smooth = cv2.blur(smpl_src, (n_kern, n_kern))
            
            smpl1 = smpl_smooth
            smpl1[obj_in_smpl_xy[1]:obj_in_smpl_xy[3], obj_in_smpl_xy[0]:obj_in_smpl_xy[2]] = smpl_obj
            smpls = [smpl1]
        # else:
        else:
            # smpls, smpl_xy, delt_xy, img_match = orb_watching(fr_curr, smpls, smpl_xy, delt_xy, edge_w, test_mode=test_mode)
            out_img_smpl, out_edge_xy, out_obj_xy, out_delt_xy, out_img_match = orb_watching(fr_curr, smpls, smpl_xy, obj_xy, delt_xy, edge_w, test_mode=test_mode)
                       
            if out_delt_xy[0]: #there was found some matched objects in a current frame
                smpls, smpl_xy, obj_xy, delt_xy, img_match = out_img_smpl, out_edge_xy, out_obj_xy, out_delt_xy, out_img_match
                
                
                # smpls1, smpl_xy, delt_xy, img_match = out_img_smpl, out_xy, out_delt_xy, out_img_match
                
                
            else: #there was no any matched objects in the current frame
                print(count)
                break
                # continue #going to the next frame with unchanged samples and delta x,y
            # cv2.imshow('res', out_img_match)
            # if cv2.waitKey(40) & 0xFF == ord('q'):
            #     break
        
    else:
        break

# cv2.destroyAllWindows()




# fn0 = 'frame_077_723.jpg'
# fn1 = 'frame_077_724.jpg'
# fn2 = 'frame_077_725.jpg'

# fns = [fn0, fn1, fn2]

# smpl_xy = [ [180, 180, 287, 280] ]
# im_in = cv2.imread(fn1)
# smpls = [ im_in[smpl_xy[0][1]:smpl_xy[0][3], smpl_xy[0][0]:smpl_xy[0][2]] ]
# delt_xy = [ [40, 20] ]
# edge_w = 40
# fr_curr = cv2.imread(fn2)

# smpl_xy = [ [160, 160, 277, 270] ]
# im_in = cv2.imread(fn0)
# smpls = [ im_in[smpl_xy[0][1]:smpl_xy[0][3], smpl_xy[0][0]:smpl_xy[0][2]] ]
# delt_xy = [ [5, 25] ]
# edge_w = 40
# fr_curr = cv2.imread(fn1)

# three frames in "frame_077_72..." files
# im_in = cv2.cvtColor(cv2.imread(fns[0]), cv2.COLOR_BGR2GRAY)
# obj_xy = [ [200, 200, 237, 230] ]
# edge_w = 40
# smpl_xy = [ [obj_xy[0][0] - edge_w, obj_xy[0][1] - edge_w, obj_xy[0][2] + edge_w, obj_xy[0][3] + edge_w] ]
# delt_xy = [ [5, 25] ]
# smpls = [ im_in[smpl_xy[0][1]:smpl_xy[0][3], smpl_xy[0][0]:smpl_xy[0][2]] ]
# test_mode = 1

# for fn_curr in fns[1:len(fns)]:
#     fr_curr = cv2.cvtColor(cv2.imread(fn_curr), cv2.COLOR_BGR2GRAY)
#     smpls, smpl_xy, delt_xy, img_match = orb_watching(fr_curr, smpls, smpl_xy, delt_xy, edge_w, test_mode=test_mode)
    
#     cv2.imshow('res', img_match)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()

