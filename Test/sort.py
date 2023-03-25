import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class Sort:
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1

        # update trackers
        for t in self.trackers:
            t.predict()

        matched, unmatched_dets, unmatched_trks = self.match(dets)

        # update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # remove trackers that have been active for too long
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -= 1

        # get active tracks
        ret = []
        for trk in self.trackers:
            if trk.is_confirmed() and trk.time_since_update <= 0:
                ret.append(trk.to_tlbr())
        return np.array(ret)

    def match(self, dets):
        if len(self.trackers) == 0:
            return [], [], list(range(len(dets)))

        iou_matrix = np.zeros((len(dets), len(self.trackers)), dtype=np.float32)

        for d, det in enumerate(dets):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = box_iou(det, trk.get_state())

        matched_indices = linear_sum_assignment(-iou_matrix)

        matched_indices = np.asarray(matched_indices)

        matched_indices = np.transpose(matched_indices)

        unmatched_dets = []
        for d, det in enumerate(dets):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)

        unmatched_trks = []
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < 0.5:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)


class KalmanBoxTracker:
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # initial covariance matrix
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.track_id = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def is_confirmed(self):
        """
        Returns the True if the track is confirmed.
        """
        return self.hits >= self.min_hits

    def is_deleted(self):
        """
        Returns True if the track is dead.
        """
        return self.time_since_update > self.max_age

    def to_tlwh(self):
        """
        Returns the bounding box in format (top left x, top left y, width, height),
        normalized [0, 1].
        """
        bbox = self.get_state()
        return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])

    def to_tlbr(self):
        """
        Returns the bounding box in format (top left x, top left y, bottom right x, bottom right y),
        normalized [0, 1].
        """
        bbox = self.to_tlwh()
        return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    area1 = w1 * h1
    area2 = w2 * h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is not None:
        return [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2, score]
    else:
        return [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2]