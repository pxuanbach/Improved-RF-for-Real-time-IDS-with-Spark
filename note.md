## Đánh giá mô hình RF trong môi trường Spark

### 1. Tổng quan về cách tiếp cận

Mô hình hiện tại sử dụng Spark để huấn luyện Random Forest (RF) với quy trình như sau:

1. Mỗi node chạy thuật toán **ReliefF Feature Selection** trên một phần dữ liệu (subset) để chọn đặc trưng quan trọng.
2. Tổng hợp danh sách đặc trưng từ các node để có một tập hợp đặc trưng toàn cục.
3. Huấn luyện RF trên toàn bộ tập dữ liệu đã giảm chiều.

### 2. RF có được training trên từng node worker không?

Có hai trường hợp chính:

#### Trường hợp 1: RF được huấn luyện trên từng worker (khả thi nhưng không phổ biến)

-   Mỗi worker huấn luyện một mô hình RF riêng biệt trên subset dữ liệu của nó.
-   Cần một phương pháp để **tổng hợp nhiều mô hình RF** từ các worker thành một mô hình RF toàn cục.
-   Điều này không phổ biến trong Spark MLlib vì RF dựa trên **bagging** và việc hợp nhất các mô hình RF là không đơn giản.

#### Trường hợp 2: RF chỉ huấn luyện trên driver (cách phổ biến trong Spark MLlib)

-   Apache Spark MLlib triển khai RF bằng cách **huấn luyện trên driver node**, trong khi dữ liệu được phân phối trên các worker để tối ưu tính toán.
-   Các worker chỉ hỗ trợ **phân phối dữ liệu và tính toán song song**, chứ không huấn luyện RF độc lập trên từng phần dữ liệu.
-   Spark sẽ chia dữ liệu thành các RDD (Resilient Distributed Dataset), sau đó driver sẽ điều phối việc huấn luyện bằng cách sử dụng dữ liệu từ các worker.
-   Đây là cách triển khai phổ biến nhất của Random Forest trong Spark MLlib, vì RF dựa trên bagging (bootstrap aggregating) và việc tổng hợp nhiều cây từ các worker là không cần thiết.

### 3. Cách kiểm tra RF có chạy trên từng worker hay không

Có thể kiểm tra bằng cách:

-   **Xem log của Spark Driver**: Nếu RF được huấn luyện trên từng worker, log sẽ hiển thị nhiều tiến trình RF trên các worker.
-   **Kiểm tra số lượng mô hình RF được tạo**: Nếu chỉ có một mô hình RF duy nhất, thì RF đang chạy trên driver.
-   **Dùng `explainParams()` trong Spark ML**:
    ```python
    from pyspark.ml.classification import RandomForestClassifier
    rf = RandomForestClassifier()
    print(rf.explainParams())
    ```
    Điều này giúp xác định cách RF được triển khai trong Spark.

### 4. Cách triển khai RF trên từng worker

Nếu muốn RF chạy trên từng worker, có thể thử:

-   **Huấn luyện nhiều mô hình RF riêng biệt trên từng worker**, sau đó dùng phương pháp **model averaging hoặc stacking** để hợp nhất kết quả.
-   **Sử dụng PySpark RDD thay vì Spark MLlib**, rồi triển khai RF theo từng phần dữ liệu phân tán.

### 5. Cách triển khai đang dùng

-   Từng worker chạy ReliefF trên subset dữ liệu của nó để chọn ra đặc trưng quan trọng.
-   Danh sách đặc trưng từ các worker được tổng hợp để có danh sách toàn cục.
-   Random Forest huấn luyện trên tập dữ liệu đã giảm chiều trên toàn bộ Spark cluster.

### 6. Gợi ý tối ưu hóa

-   **Kiểm tra tính đồng nhất của danh sách đặc trưng toàn cục** để tránh chênh lệch giữa các worker.
-   **Đánh giá hiệu suất RF sau khi giảm chiều** để đảm bảo việc chọn đặc trưng không ảnh hưởng tiêu cực đến mô hình.
-   **Thử nghiệm các phương pháp thay thế ReliefF**, như Boruta hoặc Mutual Information, để chọn đặc trưng hiệu quả hơn trong dữ liệu mạng.
