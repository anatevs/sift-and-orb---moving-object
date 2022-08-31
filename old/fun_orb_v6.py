# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 18:19:15 2022

@author: User
"""



"""version of fun_orb_v5
   Using a keypoints and a knowladge about a size of the object from 
   a previous frame to detect object's coordinates
   
   This method requires to pass a kp from a previous frame not only descriptors
   (we passed kp in a previous versions only for visualisation of algorithm work)
   
   
   The main idea is that there are no so large changes in an object sizes in a
   two neighbor frames. So we can say that in a current frame the size of 
   object is the same and rect region around object has the same position   
   relativly to the some of the equal kp.
   And than we also check all equal kp for them position: if kp is outside the
   determind rect region than the rect region will be increased to cover all
   these outside keypoints.

"""

def orb_watching(frame, imgs0, kps0, dess0, xy_imgs0, delt_xy, edge_width, test_mode=0):
    
    # frame - current frame
    # imgs0 - list of sample images of objects (with edge_width)
    # xy_imgs0 - list of coordinates of sample images from a previous frame for every object
    # delt_xy - list of sizes of ROI for every object
    # edge_width - width of a frame added around the object for a normal work of ORB algorithm

    import cv2
    import numpy as np
    import skimage.measure as ski_msr
    
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #into gray scale
    
    if test_mode == 1:
        im_show_matches = 0 #if 0 there will be shown all matches
        isshow_kp = 0
        from my_standart_modules import show_gr
    
    
    
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
        
        # next we will use an abs from delt_xy because in this version we using delt_xy as a region around the object (not as a directed shift)
        d_xy_abs = np.abs(delt_xy[i_obj])
        xy_im1_roi = [xy_imgs0[i_obj][0] - edge_width - d_xy_abs[0], xy_imgs0[i_obj][1] - edge_width - d_xy_abs[1], 
                      xy_imgs0[i_obj][2] + edge_width + d_xy_abs[0], xy_imgs0[i_obj][3] + edge_width + d_xy_abs[1]]
        
        # a version with delt_xy as a directed shift
        # xy_im1_roi = [xy_imgs0[i_obj][0]+delt_xy[i_obj][0], xy_imgs0[i_obj][1]+delt_xy[i_obj][1], 
        #               xy_imgs0[i_obj][2]+delt_xy[i_obj][0], xy_imgs0[i_obj][3]+delt_xy[i_obj][1]]
        
        img1 = frame[xy_im1_roi[1] : xy_im1_roi[3], xy_im1_roi[0] : xy_im1_roi[2]] #pattern to search matches
        
        # ORB creating
        orb = cv2.ORB_create()
        
        # calculating keypoints and descriptors of sample image
        # kp0, des0 = orb.detectAndCompute(img0, None)
        kp0 = kps0[i_obj]
        des0 = dess0[i_obj]
        
        
        # creating a partly smooth sample image
        # at first making smooth sample
        n_kern = 5 #!!!! internal number
        img1_smooth = cv2.blur(img1, (n_kern, n_kern))
        # size of img1
        size_im1 = np.shape(img1)
        # adding a part with an object in a smoothed image
        img1_smooth[edge_width : size_im1[0]-edge_width, edge_width : size_im1[1]-edge_width] = (
            img1[edge_width : size_im1[0]-edge_width, edge_width : size_im1[1]-edge_width])
        
        img1_orig = img1
        img1 = img1_smooth
        # calculating keypoints and descriptors of current image
        kp1, des1 = orb.detectAndCompute(img1, None)
        
        
        if isshow_kp == 1:
            img0_kp = cv2.drawKeypoints(img0, kp0, img0)
            img1_kp = cv2.drawKeypoints(img1, kp1, img1)
            show_gr(img0_kp, im_sc=3)
            show_gr(img1_kp, im_sc=3)
        
        out_img_sample[i_obj] = img1
        out_kp[i_obj] = kp1
        out_des[i_obj] = des1
        
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
                thr_good = (1 - 0.6)*(matches[len(matches)-1].distance -  matches[0].distance) + matches[0].distance #!!!!! magical number (threshold)
                
                for m in matches:
                    if m.distance <= thr_good:
                        matches_good.append(m)            
                
                if matches_good:
                    # # indexes of good matches in the kp1 (i.e. train image keypoints)
                    # idxs_im1 = [matches_good[i].trainIdx for i in range(0, len(matches_good))]
                    # # coordinates of good matched kp1
                    # xy_im1 = np.array([kp1[i].pt for i in idxs_im1])
                    
                    # # coordinates of object with region around it
                    # xy_edge = (np.array([(min(xy_im1[:, 0])) - edge_width, min(xy_im1[:, 1]) - edge_width, max(xy_im1[:, 0])+edge_width, max(xy_im1[:, 1])+edge_width]) + xy_im1_roi[0:2]*2).astype(int) #coordinates of the found object with region (edge_width thikness) around it
                    
                    
                    # selecting some kp, I guess 1st one, with a closest distance, will be ok
                    
                    
                    
                    lbl_data = ski_msr.label(img1 > 0.7*np.max(img1), connectivity=2)
                    show_gr(img1, im_sc = 3)
                    show_gr(img1 > 0.7*np.max(img1), im_sc = 3)
                    obj_reg_idxs = np.argwhere(lbl_data)
                    # adding xy_roi to make coordinates in a global coord.sys.
                    xy_obj_i = (np.array([(min(obj_reg_idxs[:, 1])), min(obj_reg_idxs[:, 0]), max(obj_reg_idxs[:, 1]) + 1, max(obj_reg_idxs[:, 0]) + 1]) + xy_im1_roi[0:2]*2).astype(int)
                    
                    
                    
                    
                    
                    # coordinates of object with region around it
                    # xy_edge = (np.array([(min(xy_im1[:, 0])) - edge_width, min(xy_im1[:, 1]) - edge_width, max(xy_im1[:, 0])+edge_width, max(xy_im1[:, 1])+edge_width]) + xy_im1_roi[0:2]*2).astype(int) #coordinates of the found object with region (edge_width thikness) around it
                    xy_edge = (np.array([-edge_width, -edge_width, edge_width, edge_width]) + xy_obj_i).astype(int)
                    
                    
                    if (xy_edge[0] >= 0) * (xy_edge[1]>=0) * ((frame.shape[1] - xy_edge[2])>=0) * ((frame.shape[0] - xy_edge[3]) >= 0): #if the sample with region around it is in the frame borders
                        # out_img_sample[i_obj] = frame[xy_edge[1] : xy_edge[3], xy_edge[0] : xy_edge[2]] #sample of the found object
                        # out_img_sample[i_obj] = img1
                        
                        #!!!!!! here we will change out_coord to xy_obj_i !!! done
                        out_coord[i_obj] = xy_obj_i #coordinates of rect around the found object
                        
                        #!!!!!! here we will need to detect delt from difference of xy_obj_i (not xy_edge) !!! done
                        out_delt_xy[i_obj] = [(xy_obj_i[0] - xy_imgs0[i_obj][0])*3, (xy_obj_i[1] - xy_imgs0[i_obj][1])*3]
            
                    
            # if im_show_matches == 0:
                    #     im_show_matches = len(matches_good)
                    
                    # picture with images and lines between good matches
                    # img_match = cv2.drawMatches(img0, kp0, img1, kp1, matches_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    
            if test_mode == 1:
                if im_show_matches == 0:
                    im_show_matches = len(matches_good)
                
                # drawing the rect around object in a current frame
                frame = cv2.rectangle(frame, (tuple(xy_obj_i[0:2] + [-1]*2)), (tuple(xy_obj_i[2:4] + [1]*2)), color=255)
                
                # picture with images and lines between good matches
                # img_match = cv2.drawMatches(img0, kp0, img1, kp1, matches_good[:im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # making keypoints in a frame coordinate system
                # 1st just calculating kp1 one more time because I didn't find a way to copy it as a full duplicate (copy.deepcopy doesn't work with keypoints datatype)
                kp1_frame = orb.detect(img1, None)
                # changing a coordinates parameter of keypoints
                for i_pt in range(0, len(kp1_frame)):
                    kp1_frame[i_pt].pt = (kp1[i_pt].pt[0] + xy_im1_roi[0], kp1[i_pt].pt[1] + xy_im1_roi[1])
                
                # creating a matching image with a previous and current frames
                img_match_frame = cv2.drawMatches(img0, kp0, frame, kp1_frame, matches_good[:im_show_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # cv2.imshow('res', img_match)
                # cv2.imshow('res', img_match_frame)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                
        return out_img_sample, out_kp, out_des, out_coord, out_delt_xy, img_match_frame

        # else:
        #     return out_img_sample, out_coord, out_delt_xy
        
            
        

import cv2


# video
# fn_vid = 'video_Planer1.mp4'

# obj_xy = [ [280, 245, 330, 290] ]
# edge_w = 60
# smpl_xy = [ [obj_xy[0][0] - edge_w, obj_xy[0][1] - edge_w, obj_xy[0][2] + edge_w, obj_xy[0][3] + edge_w] ]
# delt_xy = [ [2, 0] ]

fn_vid = 'video_Planer7.mp4'

obj_xy = [ [325, 290, 355, 302] ]
margin_w = 2 #a margin for the rect arount object
obj_xy1 = [[obj_xy[0][0] - margin_w, obj_xy[0][1] - margin_w, obj_xy[0][2] + margin_w, obj_xy[0][3] + margin_w]]
edge_w = 50
smpl_xy = [ [obj_xy[0][0] - edge_w, obj_xy[0][1] - edge_w, obj_xy[0][2] + edge_w, obj_xy[0][3] + edge_w] ]
delt_xy = [ [0, 0] ]

test_mode = 1

cap = cv2.VideoCapture(fn_vid)
if not cap.isOpened():
    print('cant open file')

count = 0
while cap.isOpened():
    isrtrn, fr_curr = cap.read()
    
    if isrtrn:
        # block for picking up some exactly frame and some part of it
        # fr_to_count = 2
        # if count <= fr_to_count: #2nd frame
        #     from my_standart_modules import show_gr
        #     # show_gr(fr_curr, im_sc = 5)
        #     show_gr(fr_curr[290:302, 325:355, :], im_sc = 3)
        # elif count > fr_to_count:
        #     break
        
        # this block with "count" needed when we don't have prev frame info, i.e. we are in a test mode with video file
        count = count + 1
        if count == 1:
            im_in = fr_curr
            smpls = [ im_in[smpl_xy[0][1]:smpl_xy[0][3], smpl_xy[0][0]:smpl_xy[0][2]] ]
            
            # ORB creating
            orb0 = cv2.ORB_create()
            # calculating keypoints and descriptors of sample image
            kp0, des0 = orb0.detectAndCompute(smpls[0], None)
            kps0 = [kp0]
            dess0 = [des0]
        else:
        # elif (count > 218) & (count < 226):
            fr_curr = cv2.cvtColor(fr_curr, cv2.COLOR_BGR2GRAY)
            # smpls, smpl_xy, delt_xy, img_match = orb_watching(fr_curr, smpls, smpl_xy, delt_xy, edge_w, test_mode=test_mode)
            out_img_smpl, out_kp, out_des, out_xy, out_delt_xy, out_img_match = orb_watching(fr_curr, smpls, kps0, dess0, obj_xy1, delt_xy, edge_w, test_mode=test_mode)
            if out_delt_xy[0]: #there was found some matched objects in a current frame
                smpls, kps0, dess0, obj_xy1, delt_xy, img_match = out_img_smpl, out_kp, out_des, out_xy, out_delt_xy, out_img_match
                
                
                # smpl_xy, delt_xy, img_match = out_xy, out_delt_xy, out_img_match
                # if (count % 6) == 0:
                #     smpls = out_img_smpl
                
                
            else: #there was no any matched objects in the current frame
                print(count)
                break
                # continue #going to the next frame with unchanged samples and delta x,y
            cv2.imshow('res', out_img_match)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
        
    else:
        break

cv2.destroyAllWindows()