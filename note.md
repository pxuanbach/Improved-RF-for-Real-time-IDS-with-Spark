## Đánh giá mô hình RF trong môi trường Spark

### 1. Tổng quan về cách tiếp cận

Mô hình hiện tại sử dụng Spark để huấn luyện Random Forest (RF) với quy trình như sau:

1. Mỗi node chạy thuật toán **ReliefF Feature Selection** trên một phần dữ liệu (subset) để chọn đặc trưng quan trọng.
2. Tổng hợp danh sách đặc trưng từ các node để có một tập hợp đặc trưng toàn cục.
3. Huấn luyện RF trên toàn bộ tập dữ liệu đã giảm chiều.

### 2. RF có được training trên từng node worker không?

Có hai trường hợp chính:

#### Trường hợp 1: RF được huấn luyện trên từng worker (khả thi nhưng không phổ biến)

- Mỗi worker huấn luyện một mô hình RF riêng biệt trên subset dữ liệu của nó.
- Cần một phương pháp để **tổng hợp nhiều mô hình RF** từ các worker thành một mô hình RF toàn cục.
- Điều này không phổ biến trong Spark MLlib vì RF dựa trên **bagging** và việc hợp nhất các mô hình RF là không đơn giản.

#### Trường hợp 2: RF chỉ huấn luyện trên driver (cách phổ biến trong Spark MLlib)

- Apache Spark MLlib triển khai RF bằng cách **huấn luyện trên driver node**, trong khi dữ liệu được phân phối trên các worker để tối ưu tính toán.
- Các worker chỉ hỗ trợ **phân phối dữ liệu và tính toán song song**, chứ không huấn luyện RF độc lập trên từng phần dữ liệu.
- Spark sẽ chia dữ liệu thành các RDD (Resilient Distributed Dataset), sau đó driver sẽ điều phối việc huấn luyện bằng cách sử dụng dữ liệu từ các worker.
- Đây là cách triển khai phổ biến nhất của Random Forest trong Spark MLlib, vì RF dựa trên bagging (bootstrap aggregating) và việc tổng hợp nhiều cây từ các worker là không cần thiết.

### 3. Cách kiểm tra RF có chạy trên từng worker hay không

Có thể kiểm tra bằng cách:

- **Xem log của Spark Driver**: Nếu RF được huấn luyện trên từng worker, log sẽ hiển thị nhiều tiến trình RF trên các worker.
- **Kiểm tra số lượng mô hình RF được tạo**: Nếu chỉ có một mô hình RF duy nhất, thì RF đang chạy trên driver.
- **Dùng `explainParams()` trong Spark ML**:
  ```python
  from pyspark.ml.classification import RandomForestClassifier
  rf = RandomForestClassifier()
  print(rf.explainParams())
  ```
  Điều này giúp xác định cách RF được triển khai trong Spark.

### 4. Cách triển khai RF trên từng worker

Nếu muốn RF chạy trên từng worker, có thể thử:

- **Huấn luyện nhiều mô hình RF riêng biệt trên từng worker**, sau đó dùng phương pháp **model averaging hoặc stacking** để hợp nhất kết quả.
- **Sử dụng PySpark RDD thay vì Spark MLlib**, rồi triển khai RF theo từng phần dữ liệu phân tán.

### 5. Cách triển khai đang dùng

- Từng worker chạy ReliefF trên subset dữ liệu của nó để chọn ra đặc trưng quan trọng.
- Danh sách đặc trưng từ các worker được tổng hợp để có danh sách toàn cục.
- Random Forest huấn luyện trên tập dữ liệu đã giảm chiều trên toàn bộ Spark cluster.

### 6. Gợi ý tối ưu hóa

- **Kiểm tra tính đồng nhất của danh sách đặc trưng toàn cục** để tránh chênh lệch giữa các worker.
- **Đánh giá hiệu suất RF sau khi giảm chiều** để đảm bảo việc chọn đặc trưng không ảnh hưởng tiêu cực đến mô hình.
- **Thử nghiệm các phương pháp thay thế ReliefF**, như Boruta hoặc Mutual Information, để chọn đặc trưng hiệu quả hơn trong dữ liệu mạng.

#

# Dưới đây là các bước logic

### **1. Logic trong file tiền xử lý (Preprocessing)**

#### **1.1. Đọc dữ liệu**

- Đọc 8 file CSV từ S3 (CICIDS2017), gộp thành DataFrame.
- `repartition(16)` và `cache()` để tối ưu xử lý.

#### **1.2. Tiền xử lý dữ liệu**

- **Đổi tên và làm sạch** (`preprocess_data`):
  - Đổi `' Label'` thành `'Label'`.
  - Loại giá trị không hợp lệ (`Heartbleed`, `Web Attack � Sql Injection`, `Infiltration`).
  - Chuẩn hóa nhãn: `Web Attack � Brute Force` → `Brute Force`, `Web Attack � XSS` → `XSS`.
  - Tạo cột `Attack`: 0 (BENIGN), 1 (tấn công).
  - Tạo cột `Label_Category` ánh xạ từ `Label` sang nhóm tấn công.
- **Loại cột không phù hợp** (`remove_unwanted_columns`):
  - Loại: `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `Flow Bytes/s`, `Flow Packets/s`, `Protocol`, `Destination Port`.
- **Lọc cột số**:
  - Chọn cột kiểu `double`, `integer`, `float`, loại bỏ `Label`, `Label_Category`, `Attack`.
- **Xử lý NaN/Infinity** (`handle_nan_infinity`):
  - Thay Infinity bằng `None`, NaN bằng trung vị cột.
- **Tạo label index** (`create_label_index`):
  - Ánh xạ `Label_Category` thành `label` (số) bằng `StringIndexer`.
- **Tạo vector đặc trưng** (`create_feature_vector`):
  - Gộp cột số thành vector `features` bằng `VectorAssembler`.
- **Chuẩn hóa dữ liệu** (`normalize_features`):
  - Chuẩn hóa `features` về [0, 1] bằng `MinMaxScaler`.

#### **1.3. Chọn đặc trưng bằng ReliefF**

- **Chia dữ liệu**:
  - Chia thành các split (mỗi split tối đa 15,000 bản ghi).
- **Chạy ReliefF**:
  - Áp dụng `ReliefFSelector` (threshold=0.3, numNeighbors=10, sampleSize=8) trên từng split.
- **Kết hợp đặc trưng**:
  - Gộp đặc trưng từ các split, tính trọng số trung bình, chọn `global_top_features`.

#### **1.4. Giảm chiều dữ liệu**

- Chỉ giữ cột trong `global_top_features` + `Label`, `Label_Category`, `Attack`.
- Tạo vector đặc trưng từ `global_top_features`.

#### **1.5. Lưu dữ liệu**

- Lưu dữ liệu giảm chiều vào S3 (`reduced_data.parquet`).
- Lưu metadata (`global_top_features`, `label_to_name`) vào S3.

---

### **2. Logic trong file huấn luyện (Training)**

#### **2.1. Load dữ liệu và metadata**

- Đọc dữ liệu giảm chiều từ S3 (`reduced_data.parquet`).
- Load `global_top_features` và `label_to_name` từ S3.

#### **2.2. Cân bằng dữ liệu - SMOTE**

Xử lý mất cân bằng dữ liệu bằng cách tạo thêm các mẫu tổng hợp cho các lớp thiểu số (minority classes), đảm bảo các lớp có số lượng mẫu cân bằng hơn.

- Chuyển dữ liệu về Pandas.
- Áp dụng SMOTE (random_state=42) để tạo mẫu tổng hợp cho lớp thiểu số.
- Chuyển dữ liệu đã resample về Spark DataFrame.

#### **2.3. Chọn lọc đặc trưng bổ sung bằng RFSelector**

Giảm số lượng đặc trưng từ 25 (đã được chọn bởi bước giảm chiều trước đó) xuống 18 để tối ưu hóa không gian đặc trưng trước khi huấn luyện mô hình Random Forest.

- Khởi tạo `RFSelector`:
  - Sử dụng `RFSelector(spark, n_features=18, n_trees=100, max_depth=20)` để chọn 18 đặc trưng quan trọng nhất từ 25 đặc trưng hiện có.
  - Tham số `n_trees=100` đảm bảo kết quả chọn đặc trưng ổn định, và `max_depth=20` cho phép các cây quyết định khai thác đủ thông tin để đánh giá tầm quan trọng của đặc trưng.
- Áp dụng `RFSelector` trên Spark DataFrame đã được cân bằng bởi SMOTE.
- Lưu danh sách 18 đặc trưng được chọn vào S3 (`selected_features_18.parquet`).
- Cập nhật Spark DataFrame để chỉ chứa 18 đặc trưng được chọn.

#### **2.4. Chia dữ liệu**

- Chia train/test: 80% train, 20% test (seed=42).

#### **2.5. Huấn luyện Random Forest**

- Huấn luyện `RandomForestClassifier` trên dữ liệu với 18 đặc trưng:
  - `numTrees=200`, `maxDepth=42`, `minInstancesPerNode=2`, `featureSubsetStrategy="sqrt"`, `impurity="gini"`, `seed=42`.
- Đo thời gian huấn luyện.
- Lưu mô hình vào S3 (`random_forest_model`).

#### **2.6. Dự đoán và đánh giá**

- Dự đoán trên tập test.
- Tính: F1-score, precision (macro), recall (macro), accuracy.

---

### Giải thích lý do thêm bước này

- **Vị trí của bước 2.3**: Bước chọn lọc đặc trưng bổ sung được đặt sau **Cân bằng dữ liệu - SMOTE** để đảm bảo rằng dữ liệu đã được cân bằng trước khi áp dụng `RFSelector`. Điều này giúp RFSelector đánh giá tầm quan trọng của đặc trưng một cách công bằng, không bị thiên lệch bởi các lớp thiểu số có ít mẫu.
- **Trước bước chia dữ liệu**: Việc giảm đặc trưng được thực hiện trước khi chia dữ liệu thành tập train/test để đảm bảo rằng cả hai tập đều sử dụng cùng một tập hợp 18 đặc trưng, tránh rò rỉ thông tin (data leakage).
- **Tham số của RFSelector**:
  - `n_features=18`: Được chọn dựa trên phân tích trước đó, vì 18 đặc trưng đã được chứng minh là một con số hợp lý trong một số nghiên cứu NIDS (ví dụ: Web ID: 9, Journal of Big Data 2020).
  - `n_trees=100`: Tăng từ 50 (như bạn đề xuất trước đó) lên 100 để đảm bảo kết quả chọn đặc trưng ổn định hơn.
  - `max_depth=20`: Tăng từ 10 lên 20 để cho phép các cây quyết định khai thác thông tin sâu hơn, cải thiện chất lượng đánh giá tầm quan trọng của đặc trưng.
- **Lưu danh sách đặc trưng**: Lưu danh sách 18 đặc trưng được chọn vào S3 để có thể tái sử dụng hoặc kiểm tra sau này.

### Lợi ích của việc thêm bước này

- **Tối ưu hóa không gian đặc trưng**: Giảm từ 25 xuống 18 đặc trưng có thể giúp loại bỏ các đặc trưng dư thừa hoặc kém hiệu quả, cải thiện tốc độ huấn luyện và giảm nguy cơ overfitting.
- **Tăng hiệu quả tính toán**: Với ít đặc trưng hơn, mô hình Random Forest cuối cùng sẽ huấn luyện nhanh hơn, phù hợp với mục tiêu tối ưu hóa hiệu suất tính toán của bài báo (giảm 25% thời gian huấn luyện, trang 1835).
- **Cải thiện hiệu suất phân loại**: Nếu các đặc trưng được chọn bởi RFSelector thực sự quan trọng hơn, điều này có thể cải thiện F1-score, đặc biệt trên các lớp thiểu số (như Infiltration, vốn có F1 thấp trong bài báo, F1=0.796).

### **3. Logic áp dụng cho dữ liệu mới (Inference)**

#### **3.1. Load mô hình và metadata**

- Load mô hình Random Forest từ S3.
- Load `global_top_features` và `label_to_name` từ S3.

#### **3.2. Đọc và tiền xử lý dữ liệu mới**

- Đọc file CSV mới.
- Áp dụng các bước tiền xử lý giống file preprocessing:
  - Đổi tên cột, loại giá trị không hợp lệ, ánh xạ `Label_Category`.
  - Loại cột không phù hợp.
  - Lọc cột số, xử lý NaN/Infinity, tạo label index, tạo vector đặc trưng, chuẩn hóa.

#### **3.3. Chuẩn bị dữ liệu**

- Chỉ giữ cột trong `global_top_features`.
- Tạo vector đặc trưng.

#### **3.4. Dự đoán**

- Dùng mô hình để dự đoán.
- Ánh xạ nhãn dự đoán về nhãn gốc bằng `label_to_name`.

#### **3.5. Đánh giá**

- Tính F1-score, precision, recall, accuracy.

---

##

## Bộ dữ liệu **CICIDS2017** và **Spark**

Với việc đang sử dụng bộ dữ liệu **CICIDS2017**, tôi sẽ phân tích kỹ hơn các giải pháp đã được spark hỗ trợ (Logistic Regression, Gradient-Boosted Trees, SVM, Naive Bayes, Multilayer Perceptron) để xác định cái nào phù hợp nhất khi kết hợp với hướng đi hiện tại của bạn (**ReliefF - Random Forest**) và đặc điểm của CICIDS2017. Tôi sẽ xem xét các yếu tố như: tính chất dữ liệu, hiệu suất trên Spark, khả năng xử lý dữ liệu không cân bằng, và độ phức tạp triển khai.

---

### Đặc điểm chính của CICIDS2017 liên quan đến mô hình

1. **Dữ liệu lớn**: Hơn 2.8 triệu bản ghi, yêu cầu các mô hình phải tận dụng được khả năng phân tán của Spark.
2. **Không cân bằng**: Lớp "Benign" chiếm đa số (~83%), trong khi các lớp tấn công (như DDoS, Brute Force) là thiểu số. Điều này đòi hỏi mô hình phải xử lý tốt vấn đề mất cân bằng.
3. **Nhiều đặc trưng**: 80+ đặc trưng, nhưng không phải tất cả đều quan trọng. ReliefF của bạn là một bước tốt để giảm chiều.
4. **Phức tạp phi tuyến**: Các cuộc tấn công trong CICIDS2017 có thể có mối quan hệ phi tuyến giữa các đặc trưng, đòi hỏi mô hình đủ mạnh để nắm bắt.

---

### Phân tích từng giải pháp

#### 1. Random Forest (RF) - Hiện tại bạn đang dùng

- **Phù hợp**: Rất cao.
- **Lý do**:
  - **Xử lý dữ liệu lớn**: RF trong Spark MLlib được tối ưu hóa để xử lý dữ liệu phân tán, phù hợp với kích thước của CICIDS2017.
  - **Không cân bằng**: RF hỗ trợ tham số `weightCol` để điều chỉnh trọng số lớp, giúp cải thiện hiệu suất trên các lớp tấn công thiểu số.
  - **Phi tuyến**: RF là mô hình dựa trên cây, có khả năng học các mối quan hệ phi tuyến phức tạp, rất phù hợp với dữ liệu mạng như CICIDS2017.
  - **Tương thích với ReliefF**: ReliefF chọn các đặc trưng quan trọng dựa trên trọng số, và RF hoạt động tốt với dữ liệu đã được giảm chiều.
- **Điểm mạnh**:
  - Dễ triển khai trên Spark.
  - Hiệu quả cao với dữ liệu không cân bằng khi được tinh chỉnh (ví dụ: tăng số cây, điều chỉnh ngưỡng quyết định).
- **Hạn chế**:
  - Có thể không tối ưu hóa tốt bằng các mô hình tuần tự như GBT nếu không tinh chỉnh kỹ.
- **Đề xuất**: Tiếp tục sử dụng RF làm baseline, nhưng thử nghiệm thêm các kỹ thuật như:
  - Tăng `numTrees` (ví dụ: 200-300) để cải thiện độ chính xác.
  - Dùng `classWeight` để ưu tiên lớp thiểu số.

#### 2. Gradient-Boosted Trees (GBT)

- **Phù hợp**: Cao.
- **Lý do**:
  - **Hiệu suất vượt trội**: GBT tối ưu hóa tuần tự các cây, thường cho kết quả tốt hơn RF trong các bài toán phân loại phức tạp như CICIDS2017.
  - **Phi tuyến**: Tương tự RF, GBT xử lý tốt các mối quan hệ phi tuyến trong dữ liệu mạng.
  - **Không cân bằng**: Hỗ trợ trọng số lớp, tương tự RF, giúp cải thiện recall trên lớp tấn công.
- **Điểm mạnh**:
  - Có thể đạt độ chính xác cao hơn RF nếu được tinh chỉnh (ví dụ: số lần lặp, độ sâu cây).
  - Phù hợp với dữ liệu đã qua ReliefF vì nó tập trung vào các đặc trưng quan trọng.
- **Hạn chế**:
  - Tốn tài nguyên hơn RF (ít song song hóa hơn trên Spark).
  - Thời gian huấn luyện lâu hơn, đặc biệt với dữ liệu lớn như CICIDS2017.
- **Đề xuất**: Thử GBT như một bước nâng cấp từ RF. Bắt đầu với tham số mặc định, sau đó tinh chỉnh `maxIter` (số lần lặp) và `maxDepth`.

#### 3. Logistic Regression

- **Phù hợp**: Trung bình.
- **Lý do**:
  - **Tuyến tính**: Logistic Regression giả định mối quan hệ tuyến tính giữa các đặc trưng và nhãn, trong khi CICIDS2017 có thể chứa các mẫu phi tuyến phức tạp (ví dụ: tấn công DDoS vs. benign).
  - **Không cân bằng**: Hỗ trợ trọng số lớp, nhưng hiệu suất trên lớp thiểu số thường kém hơn các mô hình dựa trên cây như RF/GBT.
- **Điểm mạnh**:
  - Nhanh và nhẹ, phù hợp để thử nghiệm nhanh trên Spark.
  - Đầu ra xác suất giúp dễ dàng điều chỉnh ngưỡng phân loại.
- **Hạn chế**:
  - Có thể không bắt được hết sự phức tạp của CICIDS2017, ngay cả sau khi ReliefF giảm chiều.
- **Đề xuất**: Sử dụng Logistic Regression như một mô hình phụ để so sánh với RF/GBT, đặc biệt nếu bạn muốn một giải pháp đơn giản và nhanh chóng. Không nên dùng làm mô hình chính.

#### 4. Support Vector Machines (SVM) - Linear SVM trong Spark

- **Phù hợp**: Thấp đến trung bình.
- **Lý do**:
  - **Tuyến tính**: Spark MLlib chỉ hỗ trợ Linear SVM, trong khi dữ liệu CICIDS2017 có thể yêu cầu kernel phi tuyến (như RBF), vốn không khả dụng.
  - **Không cân bằng**: Linear SVM có thể gặp khó khăn với dữ liệu mất cân bằng nếu không được xử lý kỹ (ví dụ: bằng cách tái cân bằng dữ liệu trước).
- **Điểm mạnh**:
  - Hiệu quả với dữ liệu chiều cao (high-dimensional) trước khi ReliefF giảm chiều.
  - Nhanh trên Spark với Linear SVM.
- **Hạn chế**:
  - Hiệu suất có thể kém hơn RF/GBT do giới hạn tuyến tính.
- **Đề xuất**: Không ưu tiên SVM trừ khi bạn nghi ngờ dữ liệu sau ReliefF có ranh giới tuyến tính rõ ràng (khả năng thấp với CICIDS2017).

#### 5. Naive Bayes

- **Phù hợp**: Thấp.
- **Lý do**:
  - **Giả định độc lập**: Naive Bayes giả định các đặc trưng độc lập với nhau, điều này không thực tế với CICIDS2017 vì các đặc trưng mạng (như số gói tin, kích thước luồng) thường có mối quan hệ chặt chẽ.
  - **Không cân bằng**: Có thể hoạt động, nhưng thường kém hiệu quả hơn RF/GBT trên dữ liệu phức tạp.
- **Điểm mạnh**:
  - Rất nhanh và nhẹ, phù hợp để thử nghiệm ban đầu.
- **Hạn chế**:
  - Hiệu suất thấp với dữ liệu mạng thực tế.
- **Đề xuất**: Chỉ dùng Naive Bayes nếu bạn muốn một baseline đơn giản để so sánh, nhưng không nên kỳ vọng cao.

#### 6. Multilayer Perceptron (MLP)

- **Phù hợp**: Trung bình đến cao.
- **Lý do**:
  - **Phi tuyến**: MLP là mạng nơ-ron, có khả năng học các mẫu phi tuyến phức tạp, rất phù hợp với CICIDS2017.
  - **Không cân bằng**: Có thể cải thiện bằng cách điều chỉnh trọng số lớp hoặc tái cân bằng dữ liệu.
- **Điểm mạnh**:
  - Linh hoạt hơn RF/GBT trong việc học các mối quan hệ phức tạp nếu được tinh chỉnh tốt.
  - Tích hợp tốt với Spark MLlib.
- **Hạn chế**:
  - Tốn tài nguyên tính toán hơn RF/GBT.
  - Yêu cầu tinh chỉnh nhiều siêu tham số (số lớp ẩn, số nơ-ron).
- **Đề xuất**: Thử MLP nếu bạn muốn khám phá hướng mạng nơ-ron và có đủ tài nguyên. Bắt đầu với cấu trúc đơn giản (ví dụ: 2 lớp ẩn).

---

### So sánh và khuyến nghị

| Mô hình                | Phù hợp với CICIDS2017 | Xử lý không cân bằng | Phi tuyến | Tốc độ trên Spark | Khuyến nghị         |
| ---------------------- | ---------------------- | -------------------- | --------- | ----------------- | ------------------- |
| Random Forest          | Rất cao                | Tốt                  | Tốt       | Nhanh             | Tiếp tục tối ưu     |
| Gradient-Boosted Trees | Cao                    | Tốt                  | Tốt       | Trung bình        | Thử nghiệm nâng cao |
| Logistic Regression    | Trung bình             | Trung bình           | Kém       | Nhanh             | So sánh phụ         |
| Linear SVM             | Thấp - Trung bình      | Trung bình           | Kém       | Nhanh             | Không ưu tiên       |
| Naive Bayes            | Thấp                   | Kém                  | Kém       | Rất nhanh         | Không ưu tiên       |
| Multilayer Perceptron  | Trung bình - Cao       | Tốt (nếu tinh chỉnh) | Tốt       | Chậm              | Thử nghiệm bổ sung  |

#### Khuyến nghị cụ thể:

1. **Tiếp tục với Random Forest**: Đây là lựa chọn tốt nhất hiện tại với CICIDS2017. Tối ưu hóa bằng cách:

   - Tăng số cây (`numTrees`).
   - Sử dụng `weightCol` để ưu tiên lớp tấn công.
   - Điều chỉnh ngưỡng phân loại để cải thiện recall trên lớp thiểu số.

2. **Thử Gradient-Boosted Trees**: Là bước nâng cấp tự nhiên từ RF, có thể mang lại hiệu suất tốt hơn nếu bạn chấp nhận thời gian huấn luyện lâu hơn.

3. **Xem xét Multilayer Perceptron**: Nếu bạn muốn thử hướng mạng nơ-ron và có tài nguyên, MLP là lựa chọn đáng cân nhắc.

4. **Bỏ qua Naive Bayes và SVM**: Hai mô hình này không phù hợp lắm với CICIDS2017 do hạn chế về giả định và khả năng xử lý phi tuyến.

---

### Kế hoạch hành động

- **Bước 1**: Tối ưu RF hiện tại với CICIDS2017 (tăng số cây, điều chỉnh trọng số).
- **Bước 2**: Chạy thử GBT trên cùng dữ liệu đã qua ReliefF và so sánh F1-score/recall với RF.
- **Bước 3**: Nếu cần, thử MLP với cấu trúc đơn giản (ví dụ: 2 lớp ẩn, 100-50 nơ-ron).

---

### ✅ **1\. Logistic Regression – Tham số thường dùng với CICIDS2017**

| Tham số           | Giá trị phổ biến        | Giá trị trong scikit-learn | Giá trị trong Spark        | Ghi chú                              |
| ----------------- | ----------------------- | -------------------------- | -------------------------- | ------------------------------------ |
| **Penalty**       | `l2`                    | `l2`                       | `elasticNetParam=0.0` (L2) | Regularization phổ biến              |
| **Solver**        | `liblinear` hoặc `saga` | `sag`                      | L-BFGS (mặc định)          | Phù hợp cho dữ liệu vừa và lớn       |
| **C**             | 0.1 – 1.0               | `C=100`                    | `regParam=0.01` (1/C)      | Inverse của regularization strength  |
| **Max_iter**      | 100 – 1000              | `max_iter=15000`           | `maxIter=15000`            | Số vòng lặp tối đa                   |
| **Class_weight**  | `balanced`              | Không sử dụng              | `weightCol` (tùy chọn)     | Giúp mô hình xử lý mất cân bằng nhãn |
| **Learning rate** | Không có trực tiếp      | Không có trực tiếp         | Không có trực tiếp         | Ảnh hưởng bởi `C` hoặc `regParam`    |

> 📌 **Ghi chú**:
>
> - Trong scikit-learn, `C=100` và `solver="sag"` được sử dụng theo yêu cầu.
> - Trong Spark MLlib, `regParam = 1/C = 0.01`, và Spark sử dụng L-BFGS thay vì `sag` (vì Spark không hỗ trợ `sag`).
> - Nguồn tham khảo từ các paper: [IEEE 10540382](https://ieeexplore.ieee.org/document/10540382), [Springer LNCS 2023](https://link.springer.com/chapter/10.1007/978-3-031-46584-0_13)

---

### ✅ **2\. Gradient Boosted Trees (XGBoost / LightGBM) – Tham số điển hình**

| Tham số              | Giá trị phổ biến  | Giá trị trong scikit-learn (XGBoost) | Giá trị trong Spark (GBTClassifier)                | Ghi chú                              |
| -------------------- | ----------------- | ------------------------------------ | -------------------------------------------------- | ------------------------------------ |
| **n_estimators**     | 100 – 500         | `n_estimators=200`                   | `maxIter=200`                                      | Số cây trong mô hình                 |
| **max_depth**        | 5 – 10            | `max_depth=10`                       | `maxDepth=10`                                      | Độ sâu mỗi cây                       |
| **learning_rate**    | 0.01 – 0.1        | `learning_rate=0.05`                 | `stepSize=0.05`                                    | Tốc độ học                           |
| **subsample**        | 0.7 – 0.9         | `subsample=0.8`                      | `subsamplingRate=0.8`                              | Tỷ lệ dữ liệu dùng cho mỗi cây       |
| **colsample_bytree** | 0.7 – 1.0         | `colsample_bytree=0.8`               | Không có trực tiếp (dùng `featureSubsetStrategy`)  | Tỷ lệ cột dùng cho mỗi cây           |
| **objective**        | `binary:logistic` | `objective="binary:logistic"`        | Không cần (mặc định cho binary)                    | Dùng cho bài toán phân loại nhị phân |
| **eval_metric**      | `auc`, `logloss`  | `eval_metric="logloss"`              | Không có trực tiếp (dùng `metric` trong evaluator) | Đánh giá mô hình trong huấn luyện    |

> 📌 **Ghi chú**:
>
> - Trong scikit-learn (hoặc XGBoost), các tham số được chọn dựa trên giá trị phổ biến và phù hợp với dữ liệu lớn.
> - Trong Spark MLlib, `GBTClassifier` được sử dụng thay cho XGBoost/LightGBM. Một số tham số như `colsample_bytree` không có trực tiếp, nhưng có thể thay thế bằng `featureSubsetStrategy` (ví dụ: `featureSubsetStrategy="0.8"`).
> - Nguồn từ [IEEE 10387439](https://ieeexplore.ieee.org/document/10387439), [ACM 2019](https://dl.acm.org/doi/abs/10.1145/3299815.3314439)

---

### ✅ **3\. Random Forest (RF) – Tham số phổ biến trong IDS**

> **Lưu ý**: Bảng gốc đề cập đến các tham số của mô hình Transformer (GPT-2/GPT-Neo), không phải Random Forest. Tôi sẽ sửa lại để tập trung vào Random Forest và thêm các tham số phù hợp.

| Tham số               | Giá trị phổ biến  | Giá trị trong scikit-learn | Giá trị trong Spark            | Ghi chú                             |
| --------------------- | ----------------- | -------------------------- | ------------------------------ | ----------------------------------- |
| **n_estimators**      | 100 – 500         | `n_estimators=200`         | `numTrees=200`                 | Số cây trong mô hình                |
| **max_depth**         | 10 – 50           | `max_depth=15`             | `maxDepth=15`                  | Độ sâu tối đa của mỗi cây           |
| **min_samples_split** | 2 – 10            | `min_samples_split=2`      | `minInstancesPerNode=2`        | Số mẫu tối thiểu để chia node       |
| **max_features**      | `sqrt`, `log2`    | `max_features="sqrt"`      | `featureSubsetStrategy="sqrt"` | Số đặc trưng tối đa khi chia node   |
| **criterion**         | `gini`, `entropy` | `criterion="gini"`         | `impurity="gini"`              | Tiêu chí để đo độ không thuần khiết |
| **random_state**      | Bất kỳ số nguyên  | `random_state=42`          | `seed=42`                      | Đảm bảo tính tái lập                |

> 📌 **Ghi chú**:
>
> - Trong Spark MLlib, các tham số tương ứng là `numTrees=200` và `maxDepth=15` vì spark chỉ hỗ trợ tối đa max depth = 30.
> - Nguồn tham khảo: [IEEE 10835438](https://ieeexplore.ieee.org/abstract/document/10835438), [arXiv 2411.03354](https://arxiv.org/pdf/2411.03354)

---

Dựa trên thông tin bạn cung cấp và các tài liệu tham khảo từ IEEE, Springer, Elsevier, cũng như notebook từ GitHub, tôi sẽ đánh giá và chỉnh sửa code của bạn để đảm bảo bước tiền xử lý dữ liệu trước khi chạy **ReliefF** tuân thủ các khuyến nghị từ các bài báo khoa học. Cụ thể, tôi sẽ loại bỏ các cột không phù hợp (như `Flow Bytes/s`, `Flow Packets/s`, và các cột định danh) trước khi thực hiện feature selection.

---

### **1. Đánh giá các cột cần loại bỏ**

Dựa trên các tài liệu tham khảo và lý do được liệt kê:

| **Feature Name**                 | **Lý do loại bỏ**                      | **Tài liệu tham khảo** |
| -------------------------------- | -------------------------------------- | ---------------------- |
| **Flow ID**                      | ID duy nhất, không mang thông tin học  | IEEE 9416558           |
| **Source IP, Destination IP**    | Dữ liệu định danh, gây overfitting     | Wiley CPE 2023         |
| **Timestamp**                    | Không hữu ích cho mô hình học          | ScienceDirect 2024     |
| **Flow Bytes/s, Flow Packets/s** | Gây lỗi NaN/Inf, không ổn định         | RomJIST 2020           |
| **Label (khi training)**         | Phải được tách riêng làm biến mục tiêu | Mọi nghiên cứu         |
| **Protocol, Destination Port**   | Gây bias mạnh nếu không xử lý kỹ       | Springer LNCS          |

#### **Phân tích**

- **Cột định danh (Flow ID, Source IP, Destination IP, Timestamp)**: Những cột này không mang thông tin hữu ích cho việc học máy và có thể gây overfitting, vì chúng chỉ là các giá trị định danh hoặc thời gian không liên quan trực tiếp đến đặc trưng của dữ liệu.
- **Cột lỗi thống kê (Flow Bytes/s, Flow Packets/s)**: Các cột này thường chứa giá trị NaN hoặc Infinity, gây lỗi trong quá trình huấn luyện và không ổn định cho mô hình. Nhiều nghiên cứu đã khuyến nghị loại bỏ chúng trước khi thực hiện feature selection.
- **Cột `Label`**: Đúng như bạn đã làm, cột `Label` (và các cột liên quan như `Label_Category`, `Attack`) cần được tách riêng để làm biến mục tiêu, không nên đưa vào feature selection.
- **Cột `Protocol`, `Destination Port`**: Những cột này có thể gây bias nếu không được xử lý kỹ (ví dụ: mã hóa thành dạng số hoặc chuẩn hóa). Tuy nhiên, trong trường hợp của bạn, bạn đang sử dụng `ReliefF` để chọn đặc trưng số, nên các cột này có thể không được chọn nếu chúng không phải kiểu số (`double`, `integer`, `float`).

#### **Kết luận**

- **Cần loại bỏ trước khi chạy ReliefF**:
  - Các cột định danh: `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`.
  - Các cột lỗi thống kê: `Flow Bytes/s`, `Flow Packets/s`.
  - Các cột liên quan đến nhãn: `Label`, `Label_Category`, `Attack` (đã được xử lý trong code của bạn).
- **Cột `Protocol`, `Destination Port`**:
  - Nếu chúng không phải kiểu số, chúng sẽ không được chọn trong bước `feature_cols` (vì bạn đã lọc chỉ lấy các cột kiểu `double`, `integer`, `float`).
  - Nếu chúng là kiểu số, bạn nên cân nhắc loại bỏ chúng trước để tránh bias, hoặc mã hóa chúng (ví dụ: sử dụng `StringIndexer` để chuyển thành dạng số nếu cần).

---
