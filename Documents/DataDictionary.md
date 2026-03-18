# Data Dictionary — Ames Housing Dataset

Nguồn: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  
Tác giả gốc: Dean De Cock

---

## Files

| File | Mô tả |
|---|---|
| `train.csv` | Training set — 1,460 rows, có cột `SalePrice` |
| `test.csv` | Test set — 1,459 rows, không có `SalePrice` |
| `data_description.txt` | Mô tả đầy đủ từng cột và các giá trị hợp lệ |
| `sample_submission.csv` | Mẫu submission từ linear regression đơn giản |

---

## Target Variable

| Cột | Mô tả |
|---|---|
| `SalePrice` | Giá bán bất động sản (USD) — biến mục tiêu cần dự đoán |

---

## Các cột dữ liệu

### Thông tin lô đất

| Cột | Mô tả |
|---|---|
| `MSSubClass` | Hạng tòa nhà |
| `MSZoning` | Phân loại quy hoạch tổng thể |
| `LotFrontage` | Số feet đường tiếp giáp lô đất |
| `LotArea` | Diện tích lô đất (sq ft) |
| `Street` | Loại đường tiếp cận |
| `Alley` | Loại đường hẻm tiếp cận |
| `LotShape` | Hình dạng chung của lô đất |
| `LandContour` | Độ bằng phẳng của lô đất |
| `Utilities` | Loại tiện ích có sẵn |
| `LotConfig` | Cấu hình lô đất |
| `LandSlope` | Độ dốc của lô đất |
| `Neighborhood` | Vị trí trong địa giới thành phố Ames |
| `Condition1` | Gần đường chính hoặc đường sắt |
| `Condition2` | Gần đường chính hoặc đường sắt (nếu có thêm) |

### Thông tin tòa nhà

| Cột | Mô tả |
|---|---|
| `BldgType` | Loại nhà ở |
| `HouseStyle` | Kiểu nhà |
| `OverallQual` | Chất lượng vật liệu và hoàn thiện tổng thể |
| `OverallCond` | Đánh giá tình trạng tổng thể |
| `YearBuilt` | Năm xây dựng gốc |
| `YearRemodAdd` | Năm cải tạo |
| `RoofStyle` | Kiểu mái |
| `RoofMatl` | Vật liệu mái |
| `Exterior1st` | Lớp phủ bên ngoài nhà |
| `Exterior2nd` | Lớp phủ bên ngoài nhà (nếu có thêm vật liệu) |
| `MasVnrType` | Loại ốp đá |
| `MasVnrArea` | Diện tích ốp đá (sq ft) |
| `ExterQual` | Chất lượng vật liệu bên ngoài |
| `ExterCond` | Tình trạng hiện tại của vật liệu bên ngoài |
| `Foundation` | Loại móng |

### Tầng hầm

| Cột | Mô tả |
|---|---|
| `BsmtQual` | Chiều cao tầng hầm |
| `BsmtCond` | Tình trạng chung của tầng hầm |
| `BsmtExposure` | Tường tầng hầm tiếp xúc ra ngoài |
| `BsmtFinType1` | Chất lượng khu vực hoàn thiện tầng hầm |
| `BsmtFinSF1` | Diện tích hoàn thiện loại 1 (sq ft) |
| `BsmtFinType2` | Chất lượng khu vực hoàn thiện thứ 2 |
| `BsmtFinSF2` | Diện tích hoàn thiện loại 2 (sq ft) |
| `BsmtUnfSF` | Diện tích tầng hầm chưa hoàn thiện (sq ft) |
| `TotalBsmtSF` | Tổng diện tích tầng hầm (sq ft) |

### Hệ thống tiện ích

| Cột | Mô tả |
|---|---|
| `Heating` | Loại hệ thống sưởi |
| `HeatingQC` | Chất lượng và tình trạng hệ thống sưởi |
| `CentralAir` | Điều hòa trung tâm |
| `Electrical` | Hệ thống điện |

### Diện tích sàn & phòng

| Cột | Mô tả |
|---|---|
| `1stFlrSF` | Diện tích tầng 1 (sq ft) |
| `2ndFlrSF` | Diện tích tầng 2 (sq ft) |
| `LowQualFinSF` | Diện tích hoàn thiện chất lượng thấp (sq ft) |
| `GrLivArea` | Diện tích sinh hoạt trên mặt đất (sq ft) |
| `BsmtFullBath` | Số phòng tắm đầy đủ ở tầng hầm |
| `BsmtHalfBath` | Số phòng tắm nhỏ ở tầng hầm |
| `FullBath` | Số phòng tắm đầy đủ trên mặt đất |
| `HalfBath` | Số phòng tắm nhỏ trên mặt đất |
| `BedroomAbvGr` | Số phòng ngủ trên tầng hầm |
| `KitchenAbvGr` | Số bếp |
| `KitchenQual` | Chất lượng bếp |
| `TotRmsAbvGrd` | Tổng số phòng trên mặt đất (không tính phòng tắm) |
| `Functional` | Đánh giá chức năng ngôi nhà |
| `Fireplaces` | Số lò sưởi |
| `FireplaceQu` | Chất lượng lò sưởi |

### Garage

| Cột | Mô tả |
|---|---|
| `GarageType` | Vị trí garage |
| `GarageYrBlt` | Năm xây garage |
| `GarageFinish` | Hoàn thiện nội thất garage |
| `GarageCars` | Sức chứa garage (số xe) |
| `GarageArea` | Diện tích garage (sq ft) |
| `GarageQual` | Chất lượng garage |
| `GarageCond` | Tình trạng garage |
| `PavedDrive` | Đường lái xe có lát đường không |

### Tiện ích ngoài trời

| Cột | Mô tả |
|---|---|
| `WoodDeckSF` | Diện tích sàn gỗ ngoài trời (sq ft) |
| `OpenPorchSF` | Diện tích hiên mở (sq ft) |
| `EnclosedPorch` | Diện tích hiên có mái che (sq ft) |
| `3SsnPorch` | Diện tích hiên 3 mùa (sq ft) |
| `ScreenPorch` | Diện tích hiên lưới (sq ft) |
| `PoolArea` | Diện tích hồ bơi (sq ft) |
| `PoolQC` | Chất lượng hồ bơi |
| `Fence` | Chất lượng hàng rào |
| `MiscFeature` | Tiện ích khác không thuộc các danh mục trên |
| `MiscVal` | Giá trị tiện ích khác (USD) |

### Thông tin bán hàng

| Cột | Mô tả |
|---|---|
| `MoSold` | Tháng bán |
| `YrSold` | Năm bán |
| `SaleType` | Loại giao dịch |
| `SaleCondition` | Tình trạng giao dịch |

---

## Cột được drop trong pipeline

Các cột sau bị loại do >50% missing hoặc không mang thông tin:

| Cột | Lý do |
|---|---|
| `PoolQC` | ~99% missing |
| `MiscFeature` | ~96% missing |
| `Alley` | ~93% missing |
| `Fence` | ~80% missing |
| `Id` | Chỉ là index, không có thông tin dự đoán |