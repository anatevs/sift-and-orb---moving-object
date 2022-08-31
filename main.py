# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:33:43 2021

@author: User
"""

"""
Description:
Using of my "watching_orb"  function to tracking object in a sky
"""

"""
With a sort of GUI to set a coordinates of an object in a first frame by using 
a mouse pointer
!! but we still have to set a delt_xy - shift of object to next frame (in 
current test videos a speed of objects is not so big and we will set delt_xy
as [0, 0])

"""


import cv2
from fun_orb import orb_watching


from os import path
fld_rel = '\\videos-to-test\\'
fld = path.dirname(__file__)

fn_vid = 'video.mp4'

edge_w = 50
delt_xy = [ [0, 0] ]

test_mode = 1

cap = cv2.VideoCapture(fld + fld_rel + fn_vid)
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
            
            # picking up a sample in a 1st frame
            r = cv2.selectROI(im_in, showCrosshair=False)
            smpl_xy = [ [r[0] - edge_w, r[1] - edge_w, r[0]+r[2] + edge_w, r[1]+r[3] + edge_w] ]
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
            out_img_smpl, out_kp, out_des, out_xy, out_delt_xy, out_img_match = orb_watching(fr_curr, smpls, kps0, dess0, smpl_xy, delt_xy, edge_w, test_mode=test_mode)
            if out_delt_xy[0]: #there was found some matched objects in a current frame
                smpls, kps0, dess0, smpl_xy, delt_xy, img_match = out_img_smpl, out_kp, out_des, out_xy, out_delt_xy, out_img_match
                
                
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
