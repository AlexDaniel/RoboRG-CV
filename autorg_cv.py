#!/usr/bin/env python3
import numpy as np
import cv2

import argparse
parser = argparse.ArgumentParser(description='Opens a video stream and'
                                 + ' outputs pan/tilt/zoom errors to stdout.')
parser.add_argument('device', type=str,
                    help='Path to process (path to a video file or /dev/videoX)')
parser.add_argument('--config', '-c', type=open, required=True,
                    help='Path to the json file with color goals')

parser.add_argument('--nth', type=int, default=1,
                    help='Process only nth frame (1 means every frame will be processed)')
parser.add_argument('--resize', '-r', type=float, default=0.5,
                    help='Coefficient for the resize operation before processing (1 means original size)')

parser.add_argument('--device-framerate', type=int, default=30,
                    help='Desired framerate of the device')
parser.add_argument('--device-width',     type=int, default=1280,
                    help='Desired width setting for the device')
parser.add_argument('--device-height',    type=int, default=720,
                    help='Desired width setting for the device')

args = parser.parse_args()

import json
config = json.load(args.config)

font = cv2.FONT_HERSHEY_PLAIN

color_contours = (  0, 255,   0)
color          = (  0, 255, 255)
color_target   = (  0,   0, 255)
color_bottom   = (180, 180, 180)

edge_dilate = 1
erode_dilate_iterations = 15
floor_shrink_erode = 6
red_line_threshold = 40

zoom_target = 0.85

floor_l_goal   = config['floor_l_goal']
floor_a_goal   = config['floor_a_goal']
floor_b_goal   = config['floor_b_goal']
floor_l_window = config['floor_l_window']
floor_a_window = config['floor_a_window']
floor_b_window = config['floor_b_window']

line_l_goal    = config['line_l_goal']
line_a_goal    = config['line_a_goal']
line_b_goal    = config['line_b_goal']
line_l_window  = config['line_l_window']
line_a_window  = config['line_a_window']
line_b_window  = config['line_b_window']

resize = args.resize
nth_frame = args.nth


cap = cv2.VideoCapture(args.device)
if cap.get(cv2.CAP_PROP_MODE) != 0:
    # ↑ somehow it means that we're working on a device
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.device_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.device_height)
    cap.set(cv2.CAP_PROP_FPS,          args.device_framerate)


import collections
views = None  # collections.OrderedDict
current_view = None

floor_lower = np.array([0, 0, 0])
floor_upper = np.array([0, 0, 0])
line_lower  = np.array([0, 0, 0])
line_upper  = np.array([0, 0, 0])


def set_ranges():
    global floor_lower, floor_upper
    global  line_lower,  line_upper

    floor_lower = np.array([floor_l_goal - floor_l_window,
                            floor_a_goal - floor_a_window,
                            floor_b_goal - floor_b_window])
    floor_upper = np.array([floor_l_goal + floor_l_window,
                            floor_a_goal + floor_a_window,
                            floor_b_goal + floor_b_window])
    line_lower  = np.array([ line_l_goal -  line_l_window,
                             line_a_goal -  line_a_window,
                             line_b_goal -  line_b_window])
    line_upper  = np.array([ line_l_goal +  line_l_window,
                             line_a_goal +  line_a_window,
                             line_b_goal +  line_b_window])


def sample_teh_floor(image):
    floor_size = 30
    height, width, channels = image.shape
    cropped = image[height - floor_size:height - 1, 0:width]

    min_color_per_row = np.amin(cropped, axis=0)
    min_color = np.amin(min_color_per_row, axis=0)

    max_color_per_row = np.amax(cropped, axis=0)
    max_color = np.amax(max_color_per_row, axis=0)

    avg_color_per_row = np.average(cropped, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    global floor_l_goal, floor_a_goal, floor_b_goal
    floor_l_goal, floor_a_goal, floor_b_goal = np.round(avg_color).astype(int)

    print("minimum:", min_color)
    print("maximum:", max_color)
    print("average:", avg_color)
    set_ranges()
    to_settings()


def tail():
    key = next(reversed(views))
    return views[key]


settings_window = 'Settings'
settings_selfchange = 10


def from_settings(val):
    if settings_selfchange:  # don't retrigger on changes
        return
    global floor_l_goal,   floor_a_goal,   floor_b_goal
    global floor_l_window, floor_a_window, floor_b_window
    floor_l_goal   = cv2.getTrackbarPos('floor_l_goal',   settings_window)
    floor_a_goal   = cv2.getTrackbarPos('floor_a_goal',   settings_window)
    floor_b_goal   = cv2.getTrackbarPos('floor_b_goal',   settings_window)
    floor_l_window = cv2.getTrackbarPos('floor_l_window', settings_window)
    floor_a_window = cv2.getTrackbarPos('floor_a_window', settings_window)
    floor_b_window = cv2.getTrackbarPos('floor_b_window', settings_window)
    set_ranges()


def to_settings():
    global settings_selfchange
    settings_selfchange = 10
    cv2.setTrackbarPos('floor_l_goal',   settings_window, floor_l_goal)
    cv2.setTrackbarPos('floor_a_goal',   settings_window, floor_a_goal)
    cv2.setTrackbarPos('floor_b_goal',   settings_window, floor_b_goal)
    cv2.setTrackbarPos('floor_l_window', settings_window, floor_l_window)
    cv2.setTrackbarPos('floor_a_window', settings_window, floor_a_window)
    cv2.setTrackbarPos('floor_b_window', settings_window, floor_b_window)


def init_settings():
    cv2.namedWindow(settings_window)
    cv2.createTrackbar('floor_l_goal',   settings_window,   0, 255, from_settings)
    cv2.createTrackbar('floor_a_goal',   settings_window,   0, 255, from_settings)
    cv2.createTrackbar('floor_b_goal',   settings_window,   0, 255, from_settings)
    cv2.createTrackbar('floor_l_window', settings_window,   0, 255, from_settings)
    cv2.createTrackbar('floor_a_window', settings_window,   0, 255, from_settings)
    cv2.createTrackbar('floor_b_window', settings_window,   0, 255, from_settings)



set_ranges()
init_settings()
to_settings()

cnt = 0
while(cap.isOpened()):
    if settings_selfchange > 0:
        settings_selfchange -= 1
    ret, frame = cap.read()
    cnt += 1
    if cnt % nth_frame != 0:
        continue  # skip some frames

    views = collections.OrderedDict()
    views['original']  = frame
    views['resized']   = cv2.resize(tail(), (0, 0), fx=resize, fy=resize)
    height, width, _ = tail().shape
    final = tail().copy()  # this is our final image
    views['lab']       = cv2.cvtColor(tail(), cv2.COLOR_BGR2LAB)
    views['bilateral'] = cv2.bilateralFilter(tail(), 7, 75, 75)

    views['floor_color']  = cv2.inRange(tail(), floor_lower, floor_upper)
    kernel = np.ones((3, 3), np.uint8)
    views['floor_eroded']  = cv2.erode( tail(), kernel, iterations=erode_dilate_iterations)
    views['floor_dilated'] = cv2.dilate(tail(), kernel, iterations=erode_dilate_iterations)

    # definitely not top
    views['floor_eroded_hack'] = tail().copy()
    cv2.rectangle(tail(), (0, 0), (width, int(height / 2.3)), (0), -1)

    floor_moments = cv2.moments(tail(), True)
    floor_x = None
    floor_y = None
    if floor_moments['m00'] != 0:
        floor_x = int(floor_moments['m10'] / floor_moments['m00'])
        floor_y = int(floor_moments['m01'] / floor_moments['m00'])
    else:
        floor_x = int(width  / 2)
        floor_y = int(height / 2)

    _, contours, hierarchy = cv2.findContours(tail(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    if len(contours) > 0:
        cont = np.vstack(contours[i] for i in range(len(contours)))
        hull.append(cv2.convexHull(cont, False))

    dil = cv2.dilate(views['floor_color'], kernel, iterations=edge_dilate)
    views['mask_interesting'] = cv2.bitwise_not(dil)

    views['mask_floor'] = np.zeros_like(tail())
    # views['mask_floor_dilated'] = cv2.dilate(mask_floor, kernel, iterations=edge_dilate)
    cv2.drawContours(tail(), hull, 0, 255, -1)

    views['mask_floor_inside'] = tail().copy()  # drawn onto later

    views['line_color'] = cv2.inRange(views['lab'], line_lower, line_upper)
    views['line_color_floor'] = cv2.bitwise_and(tail(), views['mask_floor'])
    # line_kernel = np.array((0, 1, 0,
    #                         1, 1, 1,
    #                         0, 1, 0), dtype=np.uint8)
    # views['line_color_eroded'] = cv2.erode(tail(), line_kernel, iterations=1)

    # red_lines = cv2.HoughLinesP(tail(), 3, np.pi / 60, 50, None, 60, 10)
    # red_lines = cv2.HoughLinesP(tail(), rho = 1, theta = 1*np.pi/180,
    #                             threshold = 1, minLineLength = 60, maxLineGap = 5)
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines = lsd.detect(tail())[0]
    red_lines = []
    false_lines = []
    cut_lines = []
    if detected_lines is not None:
        for i in range(0, len(detected_lines)):
            line = detected_lines[i][0]
            import math
            length = math.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
            if length > red_line_threshold:
                red_lines.append(line)
            else:
                false_lines.append(line)


    for i in range(0, len(red_lines)):
        try:
            foo = red_lines[i]
            xdiff = foo[2] - foo[0]
            ydiff = foo[3] - foo[1]
            scale = max(width  / xdiff, height / ydiff)
            new1 = (int(foo[0] - xdiff * scale), int(foo[1] - ydiff * scale))
            new2 = (int(foo[2] + xdiff * scale), int(foo[3] + ydiff * scale))
            retval, pt1, pt2 = cv2.clipLine((0, 0, width, height), new1, new2)
            cut_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
        except:
            pass


    for i in range(0, len(cut_lines)):
        view = views['mask_floor_inside']
        line = cut_lines[i]
        hack = 10
        far1_x = line[0] + (line[0] - floor_x) * hack
        far1_y = line[1] + (line[1] - floor_y) * hack
        far2_x = line[2] + (line[2] - floor_x) * hack
        far2_y = line[3] + (line[3] - floor_y) * hack
        cutout = np.array([
            (line[0], line[1]),
            (line[2], line[3]),
            (far2_x, far2_y),
            (far1_x, far1_y),
        ])

        cv2.fillConvexPoly(view, cutout, 0)

    views['mask_floor_inside_shrinked'] = cv2.erode(views['mask_floor_inside'], kernel, iterations=floor_shrink_erode)
    views['mask_target'] = cv2.bitwise_and(tail(), views['mask_interesting'])

    m = cv2.moments(tail(), True)
    _, target_contours, _ = cv2.findContours(tail(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # views['red_lines'] = lsd.drawSegments(views['resized'], red_lines)

    views['final'] = final

    # draw contours & hull
    cv2.drawContours(tail(), contours, -1, color_contours, 1, 8, hierarchy)  # draw contours
    cv2.drawContours(tail(), hull, -1, color, 3, 8)  # draw floor hull
    #cv2.drawContours(tail(), target_contours, -1, color_bottom, 3, 8)  # draw floor hull

    # draw target
    target_x = None
    target_y = None
    bottom = None
    if m['m00'] != 0:
        # pan
        target_x = int(m['m10'] / m['m00'])
        target_y = int(m['m01'] / m['m00'])
        cv2.circle(tail(), (target_x, target_y), 10, color_target, 1)
        cv2.line(tail(), (target_x - 13, target_y), (target_x + 13, target_y),
                 color_target, 1)
        cv2.line(tail(), (target_x, target_y - 13), (target_x, target_y + 13),
                 color_target, 1)

        # zoom
        bottom = target_contours[0][0][0] # init to something
        for contour in target_contours:
            for point in contour:
                p = point[0]
                if p[1] > bottom[1]:
                    bottom = p
        cv2.line(tail(), (bottom[0] - 10, bottom[1]), (bottom[0] + 10, bottom[1]),
                 color_bottom, 2)
        cv2.line(tail(), (0, bottom[1]), (width, bottom[1]),
                 color_bottom, 1)
        cv2.line(tail(), (0, height - bottom[1]), (width, height - bottom[1]),
                 color_bottom, 1)
        cv2.line(tail(), (0, int(height * zoom_target)), (int(width / 50), int(height * zoom_target)),
                 color_target, 1)

    # draw floor center
    if floor_x is not None:
        cv2.circle(tail(), (floor_x, floor_y), 2, color, 1)


    # draw lines
    for i in range(0, len(cut_lines)):
        line = cut_lines[i]
        cv2.line(tail(), (line[0], line[1]), (line[2], line[3]),
                 (  0, 255,   0), 1, cv2.LINE_AA)
    for i in range(0, len(red_lines)):
        line = red_lines[i]
        cv2.line(tail(), (line[0], line[1]), (line[2], line[3]),
                 (255,   0,   0), 1, cv2.LINE_AA)
    for i in range(0, len(false_lines)):
        line = false_lines[i]
        cv2.line(tail(), (line[0], line[1]), (line[2], line[3]),
                 (255,   0, 255), 1, cv2.LINE_AA)

    # send current error
    if m['m00'] != 0:
        half_width = width / 2
        pan_error = (target_x - half_width) / half_width
        print('output Z ' + str(round(pan_error, 4)))
        zoom_error = bottom[1] / height - zoom_target
        zoom_error *= -1
        print('output Y ' + str(round(zoom_error, 4)))

    import sys
    sys.stdout.flush()


    if current_view is None:
        current_view = len(views) - 1

    name, data = list(views.items())[current_view]
    cv2.putText(data, str(current_view) + ' - ' + name,
                (10, 20), font, 1, (160, 160, 160), 1, cv2.FILLED)
    cv2.imshow('view', data)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        sample_teh_floor(views['lab'])
    if key == ord(','):
        current_view -= 1
        current_view %= len(views)
    if key == ord('.'):
        current_view += 1
        current_view %= len(views)
    if key == ord('q'):
        break

print('Z 0')
cap.release()
cv2.destroyAllWindows()
