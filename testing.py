import cv2
import matplotlib.pyplot as plt
import zones

cap = cv2.VideoCapture("samples/C3902.MP4")
ok, frame = cap.read()
cap.release()

overlay = frame.copy()
cv2.polylines(overlay, [zones.ROAD_ZONE], True, (0, 255, 0), 3)
for cw in zones.ALL_CROSSWALKS:
    cv2.polylines(overlay, [cw], True, (255, 0, 0), 3)
for m in zones.ALL_MEDIANS:
    cv2.polylines(overlay, [m], True, (0, 165, 255), 3)
cv2.polylines(overlay, [zones.LANE_TOWARDS], True, (255, 255, 0), 3)
cv2.polylines(overlay, [zones.LANE_AWAY], True, (255, 0, 255), 3)
cv2.polylines(overlay, [zones.INTERSECTION_ZONE], True, (0, 0, 255), 3)
for line in zones.SOLID_LINES:
    cv2.line(overlay, tuple(line[0]), tuple(line[1]), (0, 255, 255), 4)
cv2.line(overlay, tuple(zones.STOP_LINES[0]), tuple(zones.STOP_LINES[1]), (255, 255, 255), 4)

plt.figure(figsize=(16, 9))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.show()
