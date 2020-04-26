from panacea_flow import PanaceaFlow
import numpy as np
import cv2 as cv

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

    def set_frame(self, img1, img2):
        super().__init__(img1, img2)

    def calculate(self):
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
                    # cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # print('Size of new tracks: ', np.shape(self.tracks))
                # cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
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
        return self.tracks