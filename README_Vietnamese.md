# Quy Trình Tái Tạo 3D Tùy Biến

Đây là một dự án Python dạng mô-đun để chuyển một tập ảnh thành mô hình tái tạo 3D. Dự án được chia thành hai giai đoạn:

1. **SfM (Structure-from-Motion)** xây dựng mô hình cảnh thưa và ước lượng tư thế camera.
2. **MVS (Multi-View Stereo)** dùng mô hình thưa đó để tạo ra đám mây điểm dày.

Thiết kế của mã nguồn cho phép bạn chạy pipeline mặc định từ đầu đến cuối, hoặc thay thế hầu như mọi bước bằng các phương pháp khác.

## Tổng Quan

Giai đoạn SfM tải ảnh, trích xuất và ghép đặc trưng, xác minh hình học, ước lượng tham số nội camera, khởi tạo tái tạo, đăng ký các camera còn lại, tam giác hóa điểm, tối ưu bundle adjustment, loại bỏ ngoại lai và xuất mô hình sparse.

Giai đoạn MVS nhận tái tạo sparse và chuyển nó thành kết quả dense bằng cách ước lượng bản đồ độ sâu, refine chúng, hợp nhất giữa nhiều góc nhìn, tạo điểm, lọc và gộp đám mây, ước lượng pháp tuyến và xuất đầu ra dense cuối cùng.

Tóm tắt:

`ảnh -> tái tạo sparse SfM -> ước lượng độ sâu MVS -> đám mây điểm dày đặc (dense)`

## Cấu Trúc Kho Mã

```text
SFM/
  pipeline.py                 # Pipeline SfM
  step1_image_loader.py       # Load và tiền xử lý ảnh
  step2_feature_extraction.py # Phát hiện và mô tả keypoint
  step3_feature_matching.py   # Ghép descriptor giữa các ảnh
  step4_geometric_verification.py
  step5_camera_intrinsics.py  # Ước lượng tham số nội camera
  step6_initial_reconstruction.py
  step7_camera_registration.py
  step8_triangulation.py
  step9_bundle_adjustment.py
  step10_outlier_filtering.py
  step11_export.py            # Xuất sparse reconstruction

MVS/
  mvs_pipeline.py             # Pipeline MVS
  mvs_step0_input_loader.py   # Đọc đầu ra SfM vào pipeline MVS
  mvs_step1_depth_estimation.py
  mvs_step2_depth_refinement.py
  mvs_step3_depth_fusion.py
  mvs_step4_point_cloud.py
  mvs_step8_export.py         # Xuất dense reconstruction
```

## Cách Hoạt Động

Quy trình thông thường là:

1. Chạy pipeline SfM trên một tập dữ liệu.
2. Lưu hoặc truyền trực tiếp sparse reconstruction mà SfM tạo ra.
3. Đưa kết quả đó vào pipeline MVS.
4. Xuất dense reconstruction.

## Ví Dụ Nhanh

```python
from SFM.pipeline import SfMPipeline
from MVS.mvs_pipeline import MVSPipeline

sfm = SfMPipeline(image_dir="path/to/images", output_dir="output/")
sparse_reconstruction = sfm.run()

mvs = MVSPipeline(
    sfm_source=sparse_reconstruction,
    output_dir="mvs_output/",
)
dense_reconstruction = mvs.run()
```

Mỗi bước trong pipeline có thể được tùy biến bằng cách truyền một đối tượng khác vào constructor.

## Yêu Cầu

Dự án được viết bằng Python và sử dụng các thư viện phổ biến cho thị giác máy tính và tính toán khoa học như:

- `numpy`
- `scipy`
- `opencv-python`

Một số tính năng tùy chọn cũng có thể dùng:

- `torch` cho tăng tốc GPU
- `laspy` cho xuất LAS
- `pye57` cho xuất E57

## Tóm Tắt

Dự án này là một bộ công cụ tái tạo 3D dạng mô-đun. Nó được xây dựng để thử nghiệm, nên các thiết lập mặc định là điểm bắt đầu phù hợp nhưng vẫn cho phép thay thế bộ trích xuất đặc trưng, bộ ghép, bộ ước lượng, bộ tinh chỉnh, phương pháp hợp nhất và bộ xuất dữ liệu.