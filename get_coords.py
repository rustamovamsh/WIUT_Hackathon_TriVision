import cv2
import numpy as np

img = cv2.imread('reference_frame.jpg')
if img is None:
    print("Error: Could not load reference_frame.jpg")
    exit()

orig_h, orig_w = img.shape[:2]

# Set display width to 1280 and compute exact proportional height
DISP_W = 1280
scale = DISP_W / float(orig_w)
DISP_H = int(orig_h * scale)

# Resize image strictly for display
display_img = cv2.resize(img, (DISP_W, DISP_H))


points_orig = []
points_disp = []


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coordinates back to original high-res video coordinates
        orig_x = int(x / scale)
        orig_y = int(y / scale)

        points_orig.append([orig_x, orig_y])
        points_disp.append((x, y))

        print(f"Click #{len(points_orig)} -> Original Video Coord: [{orig_x}, {orig_y}]")

        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        if len(points_disp) > 1:
            cv2.line(display_img, points_disp[-2], points_disp[-1], (0, 255, 0), 2)
        cv2.imshow("Click Corners", display_img)


cv2.namedWindow("Click Corners")
cv2.imshow("Click Corners", display_img)
cv2.setMouseCallback("Click Corners", mouse_click)

print("Click the 4 corners of the INTERSECTION area (median break/turning area).")
print("Press ANY key when done.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n--- COPY AND PASTE THIS INTO zones.py ---")
print(f"INTERSECTION_ZONE = np.array({points_orig}, dtype=np.int32)")