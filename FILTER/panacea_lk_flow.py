from panacea_flow import PanaceaFlow
import numpy as np
import cv2 as cv
import sys
import math

class PanaceaLKFlow(PanaceaFlow):
    lk_params = dict( winSize  = (8, 8),
                  maxLevel = 4,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.05,
                        minDistance = 3,
                        blockSize = 5 )
    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        self.track_len = 2 # maximum number of points of one track
        self.detect_interval = 1 # feature points are generated every 1 frame
        self.tracks = []
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
        theta = [0*math.pi/180, -84.3910*math.pi/180] # 3.318 first residue


        self.Ritpt = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        self.Ritptz = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        self.Rito = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
        self.Rbc = np.array([[math.cos(theta[1]), -math.sin(theta[1]), 0],[math.sin(theta[1]), math.cos(theta[1]), 0],[0,0,1]])



    def set_frame(self, img1, img2):
        super().__init__(img1, img2)

    def calculate(self):
        valid_track = False
        vis = self.img2.copy() # draw the optical tracks on this vis object
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
                    cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    valid_track = True
                    # cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # print('Size of new tracks: ', np.shape(self.tracks))
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
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
        cv.imshow('lk_track', vis)
        ch = cv.waitKey(200)
        if ch == 27: # user presses esc
            sys.exit('User manually quitted the optical flow display')
        if valid_track:
            return self.tracks
        else:
            return None
    
    def first_frame_track_generate(self, img1_path):
        self.img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        mask = np.zeros_like(self.img1)
        mask[:] = 255
        p = cv.goodFeaturesToTrack(self.img1, mask = mask, **self.feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])

    def revisualize_of(self, tracks, x_k, x_kp, camera_ray_length_k, camera_ray_length_kp, Ric_k, Ric_kp):
        vis = self.img2.copy()
        # Draw the tracks on the frame
        for track in tracks:
            if (len(track)==1):
                continue # skip all single tracks
            track0 = (track[0][0] * self.px_scale_x, track[0][1] * self.px_scale_y)
            # Backproject the point from track[0]
            rhs_lm_pos = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            landmark_position = rhs_lm_pos + np.concatenate((x_k[0:2], -1.67), axis=None)
            # Project the point to form track[1]
            landmark_in_camera = Ric_kp @ (landmark_position - np.concatenate((x_kp[0:2],-1.67), axis=None))
            landmark_image = (self.fx * landmark_in_camera[0:2]/landmark_in_camera[2] + np.array([self.cx, self.cy])) / self.px_scale_x
            cv.polylines(vis, np.int32([[track[0]] + [(landmark_image[0],landmark_image[1])]]), False, (0, 255, 0))
        cv.imshow('lk_track', vis)
        ch = cv.waitKey(200)
        if ch == 27: # user presses esc
            sys.exit('User manually quitted the optical flow display')
