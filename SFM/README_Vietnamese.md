# Pipeline SfM Incremental

Pipeline Structure-from-Motion mô-đun, dạng plug-and-play, viết bằng Python.
Mỗi bước đều cung cấp một base class rõ ràng để bạn có thể thay thế bằng thuật toán khác mà không phải động vào phần còn lại của code.

---

## Cấu Trúc Thư Mục

```text
sfm/
├── sfm_types.py                   # Cấu trúc dữ liệu dùng chung (Reconstruction, Camera, …)
├── step1_image_loader.py          # Tải ảnh và tiền xử lý
├── step2_feature_extraction.py    # Phát hiện keypoint + mô tả
├── step3_feature_matching.py      # Ghép descriptor + chọn cặp
├── step4_geometric_verification.py# Lọc inlier dựa trên RANSAC
├── step5_camera_intrinsics.py     # Ước lượng tham số nội camera
├── step6_initial_reconstruction.py# Cặp seed + khởi tạo hai ảnh
├── step7_camera_registration.py   # Đăng ký camera PnP (next-best-view)
├── step8_triangulation.py         # Tam giác hóa điểm 3-D mới
├── step9_bundle_adjustment.py     # Tối ưu đồng thời pose + điểm
├── step10_outlier_filtering.py    # Làm sạch đám mây điểm
├── step11_export.py               # Xuất PLY / COLMAP / NVM
└── pipeline.py                    # Bộ điều phối — nối toàn bộ bước lại với nhau
```

---

## Ví Dụ Chạy Pipeline

```python
from pipeline import SfMPipeline

pipeline = SfMPipeline(image_dir="my_images/", output_dir="output/")
reconstruction = pipeline.run()
```

---

## Thay Đổi Phương Pháp

Mỗi bước đều có một base class và nhiều triển khai cụ thể.
Truyền bất kỳ triển khai nào vào `SfMPipeline(...)` dưới dạng tham số keyword.

### Bộ Trích Xuất Đặc Trưng (Bước 2)
| Class | Descriptor | Ghi chú |
|---|---|---|
| `SIFTExtractor` | float-128 | Chất lượng tốt nhất, mặc định |
| `ORBExtractor` | binary-256 | Nhanh nhất |
| `AKAZEExtractor` | binary | Cân bằng tốt |
| `SuperPointExtractor` | float-256 | Deep learning, cần tự cung cấp weights |

```python
from step2_feature_extraction import ORBExtractor
pipeline = SfMPipeline("imgs/", feature_extractor=ORBExtractor(n_features=5000))
```

### Bộ Chọn Cặp Ảnh (Bước 3)
| Class | Ghi chú |
|---|---|
| `ExhaustivePairSelector` | Tất cả các cặp, mặc định |
| `SequentialPairSelector(window=5)` | Ảnh theo thứ tự / video |
| `VocabTreePairSelector` | Có khả năng mở rộng, cần mô hình truy hồi |

### Bộ Ghép (Bước 3)
| Class | Ghi chú |
|---|---|
| `BFMatcher` | Brute-force, mặc định |
| `FLANNMatcher` | Nhanh hơn với descriptor float |
| `SuperGlueMatcher` | Deep learning, cần mô hình riêng |

### Bộ Xác Minh Hình Học (Bước 4)
| Class | Ghi chú |
|---|---|
| `FundamentalRANSAC` | Mặc định, không cần nội tham số |
| `HomographyRANSAC` | Cảnh phẳng |
| `EssentialRANSAC(K)` | Cần tham số nội, chính xác nhất |
| `MAGSACVerifier` | MAGSAC++, cần opencv-contrib |

### Bộ Ước Lượng Nội Tham Số (Bước 5)
| Class | Ghi chú |
|---|---|
| `EXIFIntrinsicsEstimator` | Dùng tiêu cự từ EXIF, mặc định |
| `CalibrationFileEstimator(path)` | Nạp file JSON camera đã hiệu chỉnh |
| `PrincipalPointEstimator` | Heuristic, không cần EXIF |
| `FixedIntrinsicsEstimator(K)` | K đã biết |

### Bộ Tối Ưu Bundle Adjustment (Bước 9)
| Class | Ghi chú |
|---|---|
| `SciPyBundleAdjuster` | LM thuần Python, mặc định |
| `LocalBundleAdjuster(window=5)` | Nhanh hơn cho cảnh lớn |
| `StubBundleAdjuster` | Tắt BA (debug) |

### Bộ Xuất (Bước 11)
| Class | Đầu ra |
|---|---|
| `PLYExporter` | `sparse_cloud.ply` + `cameras.ply` |
| `COLMAPExporter` | `sparse/{cameras,images,points3D}.txt` |
| `NVMExporter` | `reconstruction.nvm` |

---

## Ví Dụ Đầy Đủ

```python
from pipeline import SfMPipeline
from step1_image_loader       import ResizePreprocessor
from step2_feature_extraction import SIFTExtractor
from step3_feature_matching   import SequentialPairSelector, FLANNMatcher
from step4_geometric_verification import EssentialRANSAC
from step5_camera_intrinsics  import CalibrationFileEstimator
from step9_bundle_adjustment  import LocalBundleAdjuster
from step11_export            import COLMAPExporter
import numpy as np

pipeline = SfMPipeline(
    image_dir  = "my_images/",
    output_dir = "output/",
    preprocessor       = ResizePreprocessor(max_side=1600),
    feature_extractor  = SIFTExtractor(n_features=10000),
    pair_selector      = SequentialPairSelector(window=5),
    matcher            = FLANNMatcher(ratio_thresh=0.75),
    intrinsics_estimator = CalibrationFileEstimator("calib.json"),
    bundle_adjuster    = LocalBundleAdjuster(window=5),
    ba_every_n_images  = 3,
    exporter           = COLMAPExporter(),
)
reconstruction = pipeline.run()
```

---

## Luồng Dữ Liệu

```text
images/
  └─ Bước 1 → images[], exif_list[]
       └─ Bước 2 → Reconstruction.keypoints
            └─ Bước 3 → Reconstruction.image_matches
                 └─ Bước 4 → Reconstruction.verified_matches
                      └─ Bước 5 → Reconstruction.cameras
                           └─ Bước 6 → registered_images[0,1] + initial points3d
                                └─ vòng lặp:
                                     Bước 7 → registered_images[new]
                                     Bước 8 → points3d (tăng dần)
                                     Bước 9 → tối ưu tại chỗ
                                     Bước 10 → loại bỏ outlier
                                └─ Bước 11 → file PLY / COLMAP / NVM
```

---

## Dependencies

```text
opencv-python >= 4.5
numpy
scipy          # cho SciPyBundleAdjuster
Pillow         # trích xuất EXIF (tùy chọn nhưng nên có)
```

Cài đặt:
```bash
pip install opencv-python numpy scipy Pillow
```