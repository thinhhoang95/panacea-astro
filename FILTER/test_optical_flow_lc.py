from panacea_flow import PanaceaFlow
import numpy as np
import cv2 as cv
import glob
import time
import os

# from common import anorm2, draw_str

lk_params = dict( winSize  = (8, 8),
                  maxLevel = 4,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.05,
                       minDistance = 3,
                       blockSize = 5 )

class App:
    def __init__(self):
        self.track_len = 10 # maximum number of points of one track
        self.detect_interval = 1
        self.tracks = []
        self.frame_idx = 0
        
    def run(self):
        # image_files = glob.glob('images/*0002*') + glob.glob('images/*0004*')
        image_files = sorted(glob.glob('images/*.jpg'))
        for image_file in image_files:
            # _ret, frame = self.cam.read()
            frame_gray = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
            vis = frame_gray.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                print('Shape of p0', np.shape(p0))
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
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
                self.tracks = new_tracks
                print('Size of new tracks: ', np.shape(self.tracks))
                cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                print('Frame ', self.frame_idx, ' detect ', np.shape(p), ' good points to track')
                print('Size of track before append points: ', np.shape(self.tracks))
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                    print('Size of track after append points: ', np.shape(self.tracks))

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
            ch = cv.waitKey(5000)
            if ch == 27:
                break

def main():
    import sys
    App().run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()