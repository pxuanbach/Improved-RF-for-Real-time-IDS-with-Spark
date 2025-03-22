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
# Phân tích logic trong file huấn luyện

### 1. Các bước chính trong file huấn luyện

**Đọc dữ liệu:**

- Load dữ liệu từ volume_files (2 file trong trường hợp này).
- `repartition(18)` để chia dữ liệu thành 18 partition.

**Tiền xử lý:**

- Đổi tên cột Label, loại bỏ giá trị không hợp lệ (Heartbleed, Web Attack – Sql Injection, Infiltration), thay thế Web Attack – Brute Force và Web Attack – XSS.
- Tạo cột Attack (0 cho BENIGN, 1 cho tấn công).
- Tạo cột Label_Category bằng cách ánh xạ Label sang attack_group.
- Loại bỏ các cột không phải số và tạo vector đặc trưng bằng VectorAssembler.

**ReliefFSelector:**

- Chia dữ liệu thành các split (mỗi split 50,000 bản ghi), chạy ReliefFSelector trên từng split, lấy trung bình trọng số để chọn global_top_features.

**Xử lý NaN/Infinity:**

- Kiểm tra và thay thế NaN/Infinity bằng giá trị trung bình.

**Huấn luyện Random Forest:**

- Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%).
- Huấn luyện mô hình Random Forest và lưu mô hình, global_top_features, label_to_name vào S3.

**Dự đoán và đánh giá:**

- Dự đoán trên tập kiểm tra, tính F1-score, precision, recall, accuracy.

### 2. Logic áp dụng cho file dự đoán

**Load mô hình và metadata:**

- Load mô hình Random Forest từ S3.
- Load global_top_features và label_to_name từ S3.

**Đọc và tiền xử lý dữ liệu mới:**

- Đọc file CSV mới, áp dụng các bước tiền xử lý giống hệt file huấn luyện (đổi tên cột, loại bỏ giá trị không hợp lệ, ánh xạ Label_Category, v.v.).

**Xử lý NaN/Infinity:**

- Kiểm tra và thay thế NaN/Infinity giống file huấn luyện.

**Chuẩn bị dữ liệu:**

- Chỉ giữ lại các cột trong global_top_features, tạo vector đặc trưng.

**Dự đoán:**

- Dùng mô hình đã load để dự đoán, ánh xạ nhãn dự đoán về nhãn gốc bằng label_to_name.

**Đánh giá:**

- Tính F1-score, precision, recall, accuracy giống file huấn luyện.

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
| Mô hình             | Phù hợp với CICIDS2017 | Xử lý không cân bằng | Phi tuyến | Tốc độ trên Spark | Khuyến nghị          |
|---------------------|------------------------|----------------------|-----------|-------------------|---------------------|
| Random Forest       | Rất cao                | Tốt                  | Tốt       | Nhanh             | Tiếp tục tối ưu     |
| Gradient-Boosted Trees | Cao                 | Tốt                  | Tốt       | Trung bình         | Thử nghiệm nâng cao |
| Logistic Regression | Trung bình              | Trung bình           | Kém       | Nhanh             | So sánh phụ         |
| Linear SVM          | Thấp - Trung bình       | Trung bình           | Kém       | Nhanh             | Không ưu tiên       |
| Naive Bayes         | Thấp                   | Kém                  | Kém       | Rất nhanh         | Không ưu tiên       |
| Multilayer Perceptron | Trung bình - Cao      | Tốt (nếu tinh chỉnh) | Tốt       | Chậm              | Thử nghiệm bổ sung  |

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

