import math
import cv2
import dlib
import numpy as np


def estimate_roll_angle(eye_l, eye_r):
    # print("rotate face...")

    eye_direction = (eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])
    # print("eye direction: " + str(eye_direction[0]) + ", " + str(eye_direction[0]))

    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # print("rotate: " + str(rotation))

    rotation = math.degrees(rotation)
    # print("rotate: " + str(rotation))
    return rotation


def rotate_face(image, center, rotation):
    mat = cv2.getRotationMatrix2D(tuple(np.array(np.array(center))), rotation, 1.0)
    # print(mat)
    # print(image.shape)

    h = image.shape[0]
    w = image.shape[1]
    sz = (w, h)

    # print("size: " + str(sz))
    image_rot = cv2.warpAffine(image, mat, sz)
    # print(image_rot.shape)

    return image_rot


def create_bounding_box(landmarks, image, distance):
    margin = 0

    # create bounding box
    p27_x = landmarks[27, 0]
    p27_y = landmarks[27, 1]
    # print("landmark 27: " + str(p27_x) + ", " + str(p27_y))

    p8_x = landmarks[8, 0]
    p8_y = landmarks[8, 1] + margin
    p15_x = landmarks[15, 0] + margin
    p15_y = landmarks[15, 1]
    p1_x = landmarks[1, 0] - margin
    p1_y = landmarks[1, 1]
    P_x = p27_x
    P_y = p27_y - distance - margin

    if P_y < 0:
        P_y = 0
    if p8_y > image.shape[0]:
        p8_y = image.shape[0]
    if p1_x < 0:
        p1_x = 0
    if p15_x > image.shape[1]:
        p15_x = image.shape[1]

    # print("point top: " + str(P_x) + ", " + str(P_y))
    # print("point bottom: " + str(p8_x) + ", " + str(p8_y))
    # print("point right: " + str(p15_x) + ", " + str(p15_y))
    # print("point left: " + str(p1_x) + ", " + str(p1_y))

    p_t = P_y
    p_b = p8_y
    p_l = p1_x
    p_r = p15_x

    return p_t, p_b, p_l, p_r


def calculate_distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_eye_points(landmarks):
    # Find the center of the left eye by averaging the points around the eye.
    cnt = 0
    l_x = 0
    l_y = 0
    for i in range(36, 41, 1):
        # print(str(landmarks[i,0]) + ", "+ str(landmarks[i,1]))
        l_x = l_x + landmarks[i, 0]
        l_y = l_y + landmarks[i, 1]
        # l += landmarks(i)
        cnt = cnt + 1
    l_x = l_x / cnt
    l_y = l_y / cnt
    l_p = [l_x, l_y]
    # print("left eye: " + str(l_p[0]) + ", " + str(l_p[1]))

    # Find the center of the right eye by averaging the points around the eye
    cnt = 0
    r_x = 0
    r_y = 0
    for i in range(42, 47, 1):
        # print(str(landmarks[i,0]) + ", "+ str(landmarks[i,1]))
        r_x = r_x + landmarks[i, 0]
        r_y = r_y + landmarks[i, 1]
        # l += landmarks(i)
        cnt = cnt + 1
    r_x = r_x / cnt
    r_y = r_y / cnt
    r_p = [r_x, r_y]
    # print("right eye: " + str(r_p[0]) + ", " + str(r_p[1]))

    return l_p, r_p


class FaceDetection:

    def __init__(self, predictor_path, rows=112, cols=112):
        self.rows = rows
        self.cols = cols
        self.shape = None
        self.dets = 0
        self.rects = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def run(self, image):
        # get landmarks
        landmarks = self.get_landmarks(image)

        # get eye points
        eye_left, eye_right = calculate_eye_points(landmarks)

        # rotate image
        rotation = (-1) * estimate_roll_angle(eye_left, eye_right)
        image_rot = rotate_face(image, eye_left, rotation)

        # calculate distance between the centers of the eyes
        distance = calculate_distance(eye_left, eye_right)

        # get new landmarks
        landmarks_new = self.get_landmarks(image_rot)

        # create bounding box
        p_t, p_b, p_l, p_r = create_bounding_box(landmarks_new, image, distance)

        # crop image
        image_crop = image_rot[int(p_t):int(p_b), int(p_l):int(p_r)]

        # resize image
        image_resized = cv2.resize(image_crop, (self.rows, self.cols))

        return image_resized

    def get_landmarks(self, img):
        self.shape = self.predictor(img, self.rects[0]).parts()
        landmarks = np.matrix([[p.x, p.y] for p in self.shape])
        return landmarks

    def find_face(self, img):
        ret = True
        self.rects = self.detector(img, 1)
        if len(self.rects) == 0:
            # print("ERROR: no faces found!")
            ret = False
        return ret

    def show_detected_face(self, img):
        print("show detected face...")

        win = dlib.image_window()

        win.clear_overlay()
        win.set_image(img)

        # for k, d in enumerate(self.rects):
        # Draw the face landmarks on the screen.
        # win.add_overlay(landmarks)

        # draw bounding box
        win.add_overlay(self.rects)
        dlib.hit_enter_to_continue()


if __name__ == "__main__":
    from PIL import Image
    dector = FaceDetection(
        "/home/rongfan/05-personality_traits/DeepPersonality/pre_trained_weights/shape_predictor_68_face_landmarks.dat")
    img = Image.open(
        "/home/rongfan/05-personality_traits/DeepPersonality/datasets/image_data/test_data/--Ymqszjv54.000/frame_1.jpg")
    img = np.array(img)
    dector.find_face(img)
    dector.show_detected_face(img)
    face = dector.run(img)
    Image.fromarray(face).show()

