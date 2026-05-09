# Pipeline Tái Tạo Dense MVS

Pipeline Multi-View Stereo dạng mô-đun. Nó nhận đầu ra SfM (camera + đám mây sparse)
và tạo ra một đám mây điểm dense có màu. Mỗi bước đều có nhiều cách triển khai có thể thay thế, bao gồm biến thể CPU và GPU ở những nơi phù hợp.

---

## Cấu Trúc Thư Mục

```text
mvs/
├── mvs_types.py                   # Cấu trúc dữ liệu dùng chung + phát hiện GPU
├── mvs_step0_input_loader.py      # Tải đầu ra SfM (COLMAP / NVM / trực tiếp)
├── mvs_step1_depth_estimation.py  # Bản đồ độ sâu theo từng ảnh
├── mvs_step2_depth_refinement.py  # Bộ lọc bilateral / guided / vá các lỗ trống 
├── mvs_step3_depth_fusion.py      # Tính sự nhất quán hình học đa góc nhìn
├── mvs_step4_point_cloud.py       # Tạo + lọc + gộp + pháp tuyến
├── mvs_step8_export.py            # Xuất PLY / PCD / LAS / E57
└── mvs_pipeline.py                # Bộ điều phối
```

---

## Ví Dụ Cách Chạy Pipeline

```python
from mvs_pipeline import MVSPipeline

# Từ thư mục sparse của COLMAP
pipeline = MVSPipeline(sfm_source="colmap_sparse/", output_dir="output/")
recon = pipeline.run()

# Trực tiếp từ kết quả của pipeline SfM
from mvs_pipeline import MVSPipeline
from mvs_step0_input_loader import SfMPipelineLoader

pipeline = MVSPipeline(
    sfm_source=my_sfm_reconstruction,     # đối tượng sfm_types.Reconstruction
    output_dir="output/",
    sfm_loader=SfMPipelineLoader(image_dir="my_images/"),
)
recon = pipeline.run()
```

---

## Các Phương Pháp Có Thể Hoán Đổi Theo Từng Bước

### Bước 0 — SfM Loader
| Class | Nguồn |
|---|---|
| `COLMAPLoader` | Thư mục sparse dạng text của COLMAP |
| `SfMPipelineLoader` | Đối tượng `sfm_types.Reconstruction` của chúng ta |
| `NVMLoader` | File VisualSFM `.nvm` |
| `ManualLoader` | Mảng numpy thô |

### Bước 1 — Ước Lượng Độ Sâu
| Class | Backend | Ghi chú |
|---|---|---|
| `PlaneSweepCPU` | CPU NumPy | Mặc định cho CPU, ổn định |
| `PatchMatchCPU` | CPU NumPy | Chất lượng tốt hơn, chậm hơn |
| `SGMDepthEstimator` | CPU OpenCV | Tốt nhất cho stereo pairs |
| `PlaneSweepGPU` | PyTorch CUDA/MPS | Nhanh hơn 10–50 lần |
| `PatchMatchGPU` | PyTorch CUDA/MPS | Chất lượng tốt + GPU |
| `MVSNetEstimator` | Mô hình DL | Có thể cắm cho MVSNet/IterMVS |

```python
from mvs_step1_depth_estimation import PatchMatchGPU
pipeline = MVSPipeline("sparse/", depth_estimator=PatchMatchGPU(n_iters=5, image_scale=0.5))
```

### Bước 2 — Tinh Chỉnh Độ Sâu
| Class | Ghi chú |
|---|---|
| `BilateralRefiner` | Làm mượt giữ biên (CPU) |
| `ConfidenceThresholdRefiner` | Loại bỏ pixel có độ tin cậy thấp |
| `HoleFillRefiner` | Nội suy các lỗ nhỏ |
| `MedianRefiner` | Loại nhiễu muối tiêu |
| `GuidedFilterRefiner` | Bộ lọc độ sâu có hướng dẫn theo màu (CPU) |
| `BilateralRefinerGPU` | Bilateral trên GPU (PyTorch) |
| `JointBilateralRefinerGPU` | Bilateral có hướng dẫn theo màu trên GPU |
| `ChainRefiner([...])` | Áp dụng nhiều bước liên tiếp |

### Bước 3 — Hợp Nhất Độ Sâu
| Class | Ghi chú |
|---|---|
| `GeometricConsistencyFusion` | Kiểu COLMAP, CPU |
| `GeometricConsistencyFusionGPU` | Gom theo batch trên GPU (PyTorch) |
| `TSDFFusion` | Hợp nhất TSDF thể tích (CPU) |
| `VotingFusion` | Bình chọn độ sâu theo pixel (CPU) |

### Bước 4 — Tạo Điểm
| Class | Ghi chú |
|---|---|
| `BackProjectionGenerator` | Back-projection trên CPU |
| `BackProjectionGeneratorGPU` | Back-projection trên GPU (PyTorch) |

### Bước 5 — Lọc Điểm
| Class | Ghi chú |
|---|---|
| `StatisticalOutlierFilter` | Lọc theo khoảng cách KNN |
| `RadiusOutlierFilter` | Loại điểm cô lập |
| `ConfidenceFilter` | Loại các điểm dưới ngưỡng tin cậy |
| `ChainPointFilter([...])` | Áp dụng nhiều bộ lọc |

### Bước 6 — Gộp Đám Mây
| Class | Ghi chú |
|---|---|
| `VoxelGridMerger(voxel_size)` | Giảm mẫu về mật độ đồng đều |
| `SimpleMerger` | Nối tất cả các view |

### Bước 7 — Ước Lượng Pháp Tuyến
| Class | Ghi chú |
|---|---|
| `PCLNormalEstimator` | PCA qua KNN (CPU) |
| `PCLNormalEstimatorGPU` | SVD theo batch (PyTorch) |
| `DepthGradientNormalEstimator` | Pháp tuyến có sẵn dựa trên gradient độ sâu |

### Bước 8 — Xuất
| Class | Định dạng |
|---|---|
| `PLYExporter` | PLY nhị phân/ASCII với RGB + pháp tuyến |
| `PCDExporter` | Định dạng PCL PCD |
| `LASExporter` | LAS của ASPRS (cần laspy) |
| `E57Exporter` | E57 của ASTM (cần pye57) |
| `MultiExporter([...])` | Xuất nhiều định dạng cùng lúc |

---

## Hỗ Trợ GPU

Pipeline tự động phát hiện phần cứng có sẵn tại thời điểm import:
```
CUDA (NVIDIA) → torch.device("cuda")   [nhanh nhất]
MPS  (Apple)  → torch.device("mps")    [nhanh trên máy M-series]
CPU fallback  → torch.device("cpu")    [luôn hoạt động]
```

Cài PyTorch để hỗ trợ GPU:
```bash
# CUDA (NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon) — PyTorch chuẩn đã bao gồm MPS:
pip install torch torchvision

# Chỉ CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Ví Dụ Đầy Đủ

```python
from mvs_pipeline import MVSPipeline
from mvs_step1_depth_estimation import PatchMatchGPU
from mvs_step2_depth_refinement import ChainRefiner, ConfidenceThresholdRefiner, JointBilateralRefinerGPU, HoleFillRefiner
from mvs_step3_depth_fusion     import GeometricConsistencyFusionGPU
from mvs_step4_point_cloud      import VoxelGridMerger, PCLNormalEstimatorGPU
from mvs_step8_export           import MultiExporter, PLYExporter, PCDExporter

pipeline = MVSPipeline(
    sfm_source      = "colmap_sparse/",
    output_dir      = "dense_output/",
    depth_estimator = PatchMatchGPU(n_iters=5, image_scale=0.5, n_src_views=6),
    depth_refiner   = ChainRefiner([
        ConfidenceThresholdRefiner(min_confidence=0.3),
        JointBilateralRefinerGPU(kernel_size=11),
        HoleFillRefiner(max_hole_size=200),
    ]),
    depth_fusion    = GeometricConsistencyFusionGPU(
        min_consistent_views=3,
        pixel_thresh=2.0,
        rel_depth_thresh=0.04,
    ),
    cloud_merger    = VoxelGridMerger(voxel_size=0.003),
    normal_estimator = PCLNormalEstimatorGPU(k=25),
    exporter        = MultiExporter([PLYExporter(binary=True), PCDExporter()]),
)
recon = pipeline.run()
```

---

## Dependencies

```text
# Bắt buộc
opencv-python >= 4.5
numpy
scipy

# Tùy chọn (tăng tốc GPU)
torch >= 2.0

# Tùy chọn (định dạng xuất)
laspy      # Xuất LAS
pye57      # Xuất E57
```