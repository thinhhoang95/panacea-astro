from panacea_flow import PanaceaFlow
import numpy as np
import cv2 as cv
import sys
import math

'''
The multistate variant of the optical flow calculator, with long-lasting feature
visibility for cross-keyframe constraining filter
'''

class PanaceaLKFlowMSS(PanaceaFlow):
    lk_params = dict( winSize  = (8, 8),
                  maxLevel = 4,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.05,
                        minDistance = 3,
                        blockSize = 5 )
    ftracks = [] # contains the multi-keyframe points

    show_cv_debug = False

    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        self.track_len = 2 # maximum number of points of one track
        self.detect_interval = 1 # feature points are generated every 1 frame
        self.tracks = []
        
        self.maximum_gap_allowed_in_mss = 33 # maximum gap allowed for missing keyframes in the feature tracking track (consult this variable in action below)

        self.frame_idx = 0
        self.fx = 3.04E-3
        self.fy = 3.04E-3
        self.px_scale_x = 5.875E-6
        self.px_scale_y = 5.875E-6
        # self.fx = focal_length_x * self.px_scale_x
        # self.fy = focal_length_y * self.px_scale_y
        self.cx = 320 * self.px_scale_x
        self.cy = 240 * self.px_scale_y

        # IMPORTING RIG CALIBRATION PARAMETERS
        #self.Rito = np.load('rito.pca.npy')
        #self.Rbc = np.load('rbc.pca.npy')
        #self.Ritpt = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # z upward to z downward

        # theta = [26.0982*math.pi/180, 84.3910*math.pi/180] # 5.025 first residue
        # theta = [-26.0982*math.pi/180, 84.3910*math.pi/180] # 6.894 first residue
        theta = [0*math.pi/180, 84.3910*math.pi/180] # 3.318 first residue


        self.Ritpt = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        self.Ritptz = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        self.Rito = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
        self.Rbc = np.array([[math.cos(theta[1]), -math.sin(theta[1]), 0],[math.sin(theta[1]), math.cos(theta[1]), 0],[0,0,1]])

        self.minimum_marginalized_track_len = 4


    def set_frame(self, img1, img2):
        super().__init__(img1, img2)

    def set_mss_frame(self, img1):
        super().mss_init(img1)

    def calculate(self): # calculate the optical flow
        valid_track = False
        vis = cv.cvtColor(self.img2.copy(),cv.COLOR_GRAY2RGB) # draw the optical tracks on this vis object
        if len(self.tracks) > 0:
                img0, img1 = self.img1, self.img2
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # print('Shape of p0', np.shape(p0))
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 2, (0, 255, 255), -1)
                    valid_track = True
                    # cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # print('Size of new tracks: ', np.shape(self.tracks))
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 255))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(self.img1)
            mask[:] = 255
            # for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            #     cv.circle(mask, (x, y), 5, 0, -1)
            p = cv.goodFeaturesToTrack(self.img1, mask = mask, **self.feature_params)
            # print('Frame ', self.frame_idx, ' detect ', np.shape(p), ' good points to track')
            # print('Size of track before append points: ', np.shape(self.tracks))
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
                # print('Size of track after append points: ', np.shape(self.tracks))

        self.frame_idx += 1
        if self.show_cv_debug:
            cv.imshow('lk_track', vis)
            ch = cv.waitKey(200)
            if ch == 27: # user presses esc
                sys.exit('User manually quitted the optical flow display')
        if valid_track:
            return self.tracks
        else:
            return []
    
    def first_frame_track_generate(self, img1_path):
        self.img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        mask = np.zeros_like(self.img1)
        mask[:] = 255
        p = cv.goodFeaturesToTrack(self.img1, mask = mask, **self.feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])

    def revisualize_of(self, tracks, x_k, x_kp, camera_ray_length_k, camera_ray_length_kp, Ric_k, Ric_kp):
        vis = cv.cvtColor(self.img2.copy(),cv.COLOR_GRAY2RGB)
        # Draw the tracks on the frame
        for track in tracks:
            if (len(track)==1):
                continue # skip all single tracks
            track0 = (track[0][0] * self.px_scale_x, track[0][1] * self.px_scale_y)
            # Backproject the point from track[0]
            rhs_lm_pos = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            # landmark_position = rhs_lm_pos + np.concatenate((x_k[0:2], -1.67), axis=None)
            landmark_position = rhs_lm_pos + x_k[0:3]
            # Project the point to form track[1]
            landmark_in_camera = Ric_kp @ (landmark_position - x_kp[0:3])
            landmark_image = (self.fx * landmark_in_camera[0:2]/landmark_in_camera[2] + np.array([self.cx, self.cy])) / self.px_scale_x
            cv.polylines(vis, np.int32([[track[0]] + [(landmark_image[0],landmark_image[1])]]), False, (0, 255, 0))
        
        if self.show_cv_debug:
            cv.imshow('lk_track', vis)
            ch = cv.waitKey(1500)
            if ch == 27: # user presses esc
                sys.exit('User manually quitted the optical flow display')

    '''
    Detect ORB keypoints in the frame
    Output: tracks, tracks_marginalize
    '''
    def detect_and_match(self, state_no_1, state_no_2):
        orb = cv.ORB_create(500)
        # sift = cv.xfeatures2d.SIFT_create()
        # Detect keypoints and calculate descriptors
        kp1, des1 = orb.detectAndCompute(self.img1_mss, None)
        kp2, des2 = orb.detectAndCompute(self.img2, None)
        # BF matcher
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        matchesMask = [[0,0] for i in range(len(matches))]
        # FLANN parameters
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        # FLANN match features
        # flann = cv.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(des1, des2, k = 2)
        tracks_modified = []
        tracks_marginalize = []
        for i,(m,n) in enumerate(matches):
            match_completed = False
            if m.distance < 0.7*n.distance:
                # Seems like a good match, check each track for the last keypoint
                matchesMask[i]=[1,0]
                key1 = kp1[m.queryIdx].pt # tuple of (x,y)
                key2 = kp2[m.trainIdx].pt # tuple of (x,y)
                for track in self.ftracks:
                    pair = track[-1][:-1] # tuple of (x,y): latest keypoint in the track
                    last_state_no = track[-1][-1]
                    if pair == key1 and last_state_no != state_no_2: # found a track: append new point into the track!
                        track.append(key2 + (state_no_2,))
                        tracks_modified.append(track)
                        match_completed = True
                        # proceed onto next match
                if not match_completed: # if there are no track to match, create a new one
                    trk = [key1 + (state_no_1,), key2 + (state_no_2,)]
                    self.ftracks.append(trk)
                    tracks_modified.append(trk)
                    match_completed = True
        
        oldest_state_index_ftracks = 9999
        oldest_state_index_marginalize = 9999
        longest_frame_count = 0
        longest_flow_components = 0

        original_ftracks = self.ftracks.copy()
        for track in original_ftracks:
            modified_track_found = False
            recover_from_gap = False
            for mtrack in tracks_modified:
                same_track = all(item in track for item in mtrack)
                if same_track: # the considering track is maintained in ftracks and recently modified
                    modified_track_found = True
                    if len(track)>=2 and state_no_2 - track[-2][-1] >= 20: # and state_no_1>0 (use this for image dataset not starting from pointer 0): # 10-11 IMU samples per camera image!
                        # Track recently recovered from gap, mark as marginalization
                        recover_from_gap = True
                    longest_flow_components = max(longest_flow_components, len(track))
            
            kf_gap_in_track = state_no_2 - track[-1][-1] # e.g. 150, 161, 180 -> missing 170, gap = 180-161 = 19

            if not modified_track_found and len(track)>=self.minimum_marginalized_track_len and kf_gap_in_track >= self.maximum_gap_allowed_in_mss: # find old tracks that are not updated in the preceeding code block (not recently appended into ftracks - as they hold only 2 keyframes and might be appended in the future), holds at least 4 keyframes and the gap (current to last keyframe) is more than 33
                tracks_marginalize.append(track.copy()) # too old tracks that might not get appended in the future with strong likelihood
                oldest_state_index_marginalize = min(oldest_state_index_marginalize, track[0][2])
                self.ftracks.remove(track)
            elif modified_track_found and recover_from_gap: # tracks that recently recover from gap are destined to marginalize
                tracks_marginalize.append(track.copy())
                oldest_state_index_marginalize = min(oldest_state_index_marginalize, track[0][2])
                self.ftracks.remove(track)
            elif not modified_track_found and len(track)<self.minimum_marginalized_track_len and kf_gap_in_track >= self.maximum_gap_allowed_in_mss: # old tracks that do not carry information for more than 4 keyframes and the gap is more than 33 -> fuck 'em and forget about 'em
                self.ftracks.remove(track) # too old tracks that might not get appended in the future with very little information (holds less than 4 keyframes)
            # Any track whose gap is less than 33 is retained for further possible growth

        # Re-evaluate the frame count after altering the ftracks
        for track in self.ftracks:
            frame_count = len(track)
            oldest_state_index_ftracks = min(track[0][2], oldest_state_index_ftracks)
            if frame_count > longest_frame_count:
                longest_frame_count = frame_count
                
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
        img3 = cv.drawMatchesKnn(self.img1,kp1,self.img2,kp2,matches,None,**draw_params)
        if self.show_cv_debug:
            cv.imshow('Matching', img3)

        # print('Tracks ({:d})'.format(len(self.ftracks)))
        # print(self.ftracks[0:min(len(self.ftracks),10)])
        # print('Tracks to marginalize ({:d})'.format(len(tracks_marginalize)))
        # print(tracks_marginalize[0:min(len(tracks_marginalize),10)])

            cv.waitKey(200)

        return self.ftracks, tracks_marginalize, oldest_state_index_ftracks, oldest_state_index_marginalize, longest_flow_components, longest_frame_count