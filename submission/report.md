# Báo Cáo Thực Hành Lab 18 — 2D Perception
**Học viên:** Dương Văn Thanh  
**Mã học viên:** 2A202602368  

Link notebook đã chạy: https://github.com/JJayzdev/Track04-Day18-DuongVanThanh-2A202602368-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

---

## 1. Báo Cáo Thí Nghiệm Bonus 4C — So sánh hai model trên val gốc và val lật gương

### Bảng kết quả thực nghiệm (trên T4 GPU, 40 epoch):

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương |
|---|:---:|:---:|
| **flip_idx giải phẫu** | **0.457** | **0.439** |
| **flip_idx đồng nhất** | **0.417** | **0.298** |

### Phân tích hiện tượng và Metric che lỗi:
1. **Hiện tượng quan sát**:
   - Trên **tập val gốc**: Cả hai mô hình đều đạt Pose mAP50-95 xấp xỉ nhau (0.457 so với 0.417). Nếu chỉ nhìn vào metric mAP trên tập val gốc, người xây dựng mô hình sẽ lầm tưởng rằng `flip_idx` đồng nhất hoạt động bình thường và không có bất kỳ lỗi nào.
   - Trên **tập val lật gương (`tiger-pose-mirror`)**: Mô hình dùng `flip_idx giải phẫu` vẫn giữ vững phong độ ổn định ở mức **0.439**, trong khi mô hình dùng `flip_idx đồng nhất` bị **tụt giảm mạnh xuống chỉ còn 0.298** (giảm gần 30% hiệu năng do các keypoint trái/phải bị đảo lộn).

2. **Metric nào đã che lỗi?**
   - Metric **Pose mAP trên tập validation gốc** đã che giấu hoàn toàn lỗi sai nhãn.
   - **Lý do**: Trong tập dữ liệu `tiger-pose` gốc, 100% số lượng hổ ở cả tập train (210 ảnh) và tập val (53 ảnh) đều quay sang phải (`facing right = 210/53, left = 0/0`). Khi augmentation lật ảnh ngang (`fliplr = 0.5`), ảnh con hổ bị lật sang quay trái. 
   - Với `flip_idx` đồng nhất `[0, 1, ..., 11]`, chân phía camera của con hổ quay trái bị ép gán nhãn là "chân phải". Tuy nhiên, vì tập validation gốc chỉ có hổ quay phải nên mô hình chưa bao giờ bị chấm điểm trên bất kỳ con hổ quay trái nào. Do đó metric mAP trên val gốc không hề phát hiện được lỗi sai lệch giải phẫu này.

3. **Bài học thiết kế tập validation**:
   - Tập validation không chỉ cần chia ngẫu nhiên từ tập train mà phải kiểm soát tính cân bằng phân phối (distributional balance) đối với các biến thể đối xứng và hướng quay.
   - Một tập val thiên lệch (100% quay về một hướng) sẽ tạo ra "ảo giác độ chính xác cao" (false sense of safety), dẫn đến mô hình bị hỏng nặng khi gặp dữ liệu thực tế ngoài production.

---

## 2. Báo Cáo Bài Tập Về Nhà — Đo Pipeline và Đánh Giá Export ONNX trên CPU

### Cấu hình đo lường:
- Model: `yolo26n.pt` được export sang ONNX (`model.export(format="onnx")`).
- Môi trường đo: Intel CPU / Colab VM CPU, chạy 30 lần lặp sau warm-up trên ảnh kích thước 640×640.

### Kết quả đo Latency (ms):

| Cấu hình Head | Confidence Threshold | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Tổng Latency (ms) | Số Box phát hiện |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **One-to-many + NMS** | `conf = 0.25` | 1.8 | 32.5 | 2.4 | 36.7 | 5 |
| **One-to-many + NMS** | `conf = 0.001` | 1.8 | 32.7 | 18.6 | 53.1 | 51 |
| **One-to-one (NMS-free)** | `conf = 0.25` | 1.8 | 32.4 | 0.3 | 34.5 | 5 |
| **One-to-one (NMS-free)** | `conf = 0.001` | 1.8 | 32.6 | 0.4 | 34.8 | 5 |

### Nhận xét & Đánh giá:
1. **Nghẽn cổ chai của NMS truyền thống**:
   - Khi hạ ngưỡng confidence xuống `0.001` (tương tự như khi phát hiện vật thể nhỏ, cảnh phức tạp hoặc trong điều kiện ánh sáng yếu), head one-to-many sinh ra hơn 1.000 ứng viên tiềm năng.
   - Thuật toán greedy NMS chạy tuần tự trên CPU phải tính toán ma trận IoU khổng lồ và sắp xếp điểm số liên tục, làm thời gian `postprocess` tăng gấp gần **8 lần** (từ 2.4 ms lên 18.6 ms), chiếm tới 35% tổng thời gian pipeline.
2. **Sự vượt trội của Head One-to-One (NMS-free)**:
   - Head one-to-one loại bỏ hoàn toàn bước NMS. Thời gian hậu xử lý giữ nguyên ở mức **< 0.5 ms** bất kể mức confidence hay số lượng box ứng viên.
   - Tổng thời gian pipeline của kiến trúc NMS-free gần như phẳng tuyệt đối (~34.8 ms), mang lại độ trễ dự đoán cực kỳ ổn định (deterministic latency), đặc biệt tối ưu cho các thiết bị nhúng biên (Edge AI, CPU, NPU) trong camera giám sát cổng nhà máy.
