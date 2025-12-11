import cv2
import numpy as np

# Parameters for Shi-Tomasi corner detection (good features to track)
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Open video
cap = cv2.VideoCapture('/Users/queenlatifa/Desktop/CSC231/girl_walking.mp4')

ret, first_frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()

# Let user select ROI (object to track)
roi = cv2.selectROI("Select Object to Track", first_frame, False, False)
cv2.destroyWindow("Select Object to Track")

x, y, w, h = roi
if w == 0 or h == 0:
    print("No ROI selected, exiting.")
    exit()

# Convert to grayscale
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track inside the ROI
mask = np.zeros_like(old_gray)
mask[y:y+h, x:x+w] = 255
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

if p0 is None:
    print("No features found in the selected ROI.")
    exit()

# Create mask image for drawing tracks
mask_draw = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st.ravel() == 1]
        good_old = p0[st.ravel() == 1]

        # Filter points inside the ROI rectangle
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        inside_points = []
        inside_old_points = []

        for new_pt, old_pt in zip(good_new, good_old):
            nx, ny = new_pt.ravel()
            if x_min <= nx <= x_max and y_min <= ny <= y_max:
                inside_points.append(new_pt)
                inside_old_points.append(old_pt)

        if len(inside_points) == 0:
            # No points left inside ROI, clear lines
            mask_draw = np.zeros_like(frame)
            cv2.imshow('Tracking', frame)
            # Optionally, break here to stop tracking
            # break
        else:
            inside_points = np.array(inside_points).reshape(-1, 1, 2)
            inside_old_points = np.array(inside_old_points).reshape(-1, 1, 2)

            # Draw tracking lines and points
            for new, old in zip(inside_points, inside_old_points):
                a, b = new.ravel()
                c, d = old.ravel()
                mask_draw = cv2.line(mask_draw, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            img = cv2.add(frame, mask_draw)
            cv2.imshow('Tracking', img)

            # Update previous frame and points
            old_gray = frame_gray.copy()
            p0 = inside_points
    else:
        # No points detected or tracked, clear lines
        mask_draw = np.zeros_like(frame)
        cv2.imshow('Tracking', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()