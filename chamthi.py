import cv2
import numpy as np

# Đường dẫn ảnh
image_path = "baithi3.jpg"

# Kích thước A4 chuẩn ở DPI 300
A4_WIDTH_PX = 2480  # 210mm * 300dpi / 25.4
A4_HEIGHT_PX = 3508  # 297mm * 300dpi / 25.4

# Đọc ảnh
image = cv2.imread(image_path)
orig = image.copy()

# Chuyển sang grayscale và làm mờ
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Phát hiện cạnh
edges = cv2.Canny(gray, 50, 200)


contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

output = image.copy()
cv2.drawContours(output, contours, -1, (127, 127, 255), 2)

# Tìm contour có 4 điểm (tờ giấy)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc_cnt = approx
        break

# Hàm sắp xếp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# Biến đổi phối cảnh
pts = doc_cnt.reshape(4, 2)
rect = order_points(pts)

dst = np.array([
    [0, 0],
    [A4_WIDTH_PX - 1, 0],
    [A4_WIDTH_PX - 1, A4_HEIGHT_PX - 1],
    [0, A4_HEIGHT_PX - 1]], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (A4_WIDTH_PX, A4_HEIGHT_PX))

# Tăng sáng
alpha = 1.1  # Hệ số tăng sáng (1.0 là không thay đổi)
beta = 3    # Hệ số tăng độ sáng (0 là không thay đổi)
warped = cv2.addWeighted(warped, alpha, np.zeros(warped.shape, warped.dtype), 0, beta)

# Ngưỡng để chuyển vùng sáng thành trắng
threshold_value_B = 135
bright_regions_mask = warped > threshold_value_B
# Đặt các vùng sáng thành màu trắng (255)
warped[bright_regions_mask] = 255

# Tăng contrast
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8,8))
lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l2 = clahe.apply(l)
lab = cv2.merge((l2,a,b))
warped = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# # Làm sắc nét
# #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# #warped = cv2.filter2D(warped, -1, kernel)

# Chuyển đổi sang thang độ xám
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

dilated_black = cv2.bitwise_not(warped)
kernel = np.ones((3, 3), np.uint8)
dilated_black = cv2.dilate(dilated_black, kernel, iterations=1)
warped = cv2.bitwise_not(dilated_black)

# Ngưỡng để chuyển vùng sáng thành đen
threshold_value_B = 150
bright_regions_mask = warped < threshold_value_B
# Đặt các vùng sáng thành màu đen (0)
warped[bright_regions_mask] = 0

# Lưu kết quả
# cv2.imwrite("de_thi_A4_scaled_noBlur.jpg", warped)
cv2.imwrite("de_thi_A4_scaled_preprocessing.jpg", warped)

##############################################################################

# Bước 1: Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread('de_thi_A4_scaled_preprocessing.jpg')
#image = warped
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bước 2: Làm mờ và phân ngưỡng nhị phân đảo
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY_INV)

# Bước 3: Tìm contour
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Bước 4: Lọc các hình vuông màu đen (marker)
markers = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > 5500 and area < 8000:  # Có 4 cạnh và đủ lớn
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.8 < aspect_ratio < 1.2:  # gần hình vuông
            markers.append((x, y, w, h))
            # Vẽ hình chữ nhật bao quanh ô vuông màu đen
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

# Bước 5: Kiểm tra số lượng marker
num_markers = len(markers)

if num_markers == 4:
    print("Đã phát hiện đúng 4 marker.")
else:
    print(f"Phát hiện {num_markers} marker. Cần đúng 4 marker để tiếp tục.")

# Sắp xếp lại 4 marker thành: [top-left, top-right, bottom-right, bottom-left]
def sort_markers(markers):
    centers = [(x + w//2, y + h//2) for (x, y, w, h) in markers]
    centers = np.array(centers)

    # Tổng tọa độ nhỏ nhất -> góc trên trái
    # Tổng tọa độ lớn nhất -> góc dưới phải
    s = centers.sum(axis=1)
    diff = np.diff(centers, axis=1)

    top_left = centers[np.argmin(s)]
    bottom_right = centers[np.argmax(s)]
    top_right = centers[np.argmin(diff)]
    bottom_left = centers[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

# Sắp xếp và chuẩn hóa
sorted_pts = sort_markers(markers)

# Chiều rộng và chiều cao chuẩn sau khi cắt
width = 1000
height = 1400

# Ma trận biến đổi để crop lại ảnh chính cho thẳng thớm
dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype='float32')

thresh = cv2.bitwise_not(thresh)
matrix = cv2.getPerspectiveTransform(sorted_pts, dst_pts)
warped = cv2.warpPerspective(thresh, matrix, (width, height))

# Tọa độ 3 vùng: (x, y, w, h)
# Vùng mã số báo danh (ví dụ góc phải giữa)
sbd_crop = warped[40:340, 580:770]

# Vùng mã đề (góc phải trên)
ma_de_crop = warped[40:340, 822:930]

# Vùng đáp án
answer_crop_1_17 = warped[702:1355, 180:343]
answer_crop_18_34 = warped[702:1355, 418:581]
answer_crop_35_40 = warped[702:1355, 656:819]

# Hiển thị và lưu
#cv2.imwrite("sbd.jpg", sbd_crop)

#cv2.imwrite("ma_de.jpg", ma_de_crop)

#cv2.imwrite("answer_1_17.jpg", answer_crop_1_17)
#cv2.imwrite("answer_18_34.jpg", answer_crop_18_34)
#cv2.imwrite("answer_35_40.jpg", answer_crop_35_40)

import matplotlib.pyplot as plt

def draw_grid(image, rows, cols, color=(0, 255, 0), thickness=1):
    img = image.copy()
    h, w = img.shape[:2]
    dy, dx = h / rows, w / cols

    # Vẽ các đường ngang
    for i in range(1, rows):
        y = int(i * dy)
        cv2.line(img, (0, y), (w, y), color, thickness)

    # Vẽ các đường dọc
    for j in range(1, cols):
        x = int(j * dx)
        cv2.line(img, (x, 0), (x, h), color, thickness)

    return img

# Áp dụng lưới
sbd_grid = draw_grid(sbd_crop, 10, 6)
ma_de_grid = draw_grid(ma_de_crop, 10, 3)
ans_1_17_grid = draw_grid(answer_crop_1_17, 17, 4)
ans_18_34_grid = draw_grid(answer_crop_18_34, 17, 4)
ans_35_40_grid = draw_grid(answer_crop_35_40, 17, 4)

# Hiển thị bằng matplotlib để dễ quan sát
def show_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


import json
import matplotlib.pyplot as plt

all_values = []
all_labels = []

def detect_filled_bubbles(image, rows, cols, threshold=190):
    """Trả về ma trận nhị phân (1 nếu ô được tô đen) và lưu giá trị mean để thống kê"""
    global all_values, all_labels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    dy, dx = h // rows, w // cols

    result = []
    for r in range(rows):
        row_result = []
        for c in range(cols):
            cell = gray[r * dy:(r + 1) * dy, c * dx:(c + 1) * dx]
            # Create circular mask
            mask = np.zeros_like(cell, dtype=np.uint8)
            cy, cx = dy // 2, dx // 2
            radius = int(min(dy, dx) * 0.4)  # or tune to match your bubble size
            cv2.circle(mask, (cx, cy), radius, 255, -1)  # white-filled circle

            # Compute masked average
            mean_val = cv2.mean(cell, mask=mask)[0]  # returns a tuple, take [0]
            row_result.append(1 if mean_val < threshold else 0)
            #print(f"Cell[{r},{c}] = {mean_val}")

            # Thêm vào biểu đồ
            all_values.append(mean_val)
            all_labels.append(1 if mean_val < threshold else 0)
        result.append(row_result)
    return result

# ----------------------
# XỬ LÝ VÙNG SBD (10 dòng x 6 cột)
sbd_result_raw = detect_filled_bubbles(sbd_crop, 10, 6)
sbd_result = {
    str(col + 1): {str(row): sbd_result_raw[row][col] for row in range(10)}
    for col in range(6)
}

# ----------------------
# XỬ LÝ VÙNG MÃ ĐỀ (10 dòng x 3 cột)
ma_de_result_raw = detect_filled_bubbles(ma_de_crop, 10, 3)
ma_de_result = {
    str(col + 1): {str(row): ma_de_result_raw[row][col] for row in range(10)}
    for col in range(3)
}

# ----------------------
# XỬ LÝ ĐÁP ÁN (17 dòng x 4 cột mỗi phần)
answers_result = {}

# Tạo danh sách đáp án từ 1 → 40
label_map = ['a', 'b', 'c', 'd']

def add_answer_block(crop_img, start_index):
    block = detect_filled_bubbles(crop_img, 17, 4)
    for i, row in enumerate(block):
        answers_result[str(start_index + i)] = {
            label_map[j]: row[j] for j in range(4)
        }

add_answer_block(answer_crop_1_17, 1)
add_answer_block(answer_crop_18_34, 18)
add_answer_block(answer_crop_35_40, 35)

# ----------------------
# GỘP THÀNH DỮ LIỆU JSON
result_json = {
    "sbd": sbd_result,
    "ma_de": ma_de_result,
    "answers": answers_result
}

# ----------------------
# LƯU FILE KẾT QUẢ
with open("result3.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, indent=4)

print("✅ Hoàn tất. Kết quả đã lưu vào 'result.json'")

# ----------------------
# VẼ BIỂU ĐỒ mean_val các ô
plt.figure(figsize=(12, 6))
plt.scatter(range(len(all_values)), all_values,
            c=['red' if l == 1 else 'green' for l in all_labels])
plt.axhline(190, color='blue', linestyle='--', label='Ngưỡng (threshold = 190)')
plt.title("Biểu đồ thống kê giá trị trung bình mức xám (mean pixel) của các ô")
plt.xlabel("Chỉ số ô (index)")
plt.ylabel("Giá trị mean pixel")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()