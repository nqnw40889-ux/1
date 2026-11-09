# Nhóm 16 · Đề tài 11

Dự đoán hỏng hóc máy phục vụ bảo trì dự đoán trên dữ liệu cảm biến mất cân bằng. Chương trình phân loại từng bản ghi AI4I 2020 và xếp các ca cần ưu tiên kiểm tra. AI4I là dữ liệu mô phỏng và nhãn mô tả trạng thái ở bản ghi hiện tại.

- [Danh sách đề tài, hàng 18](https://docs.google.com/spreadsheets/d/1abLmTsfN_qA_GXK9RPDaYKCorZ8KCsRQ/edit?gid=1201908943#gid=1201908943)
- [Nguồn AI4I 2020, UCI](https://archive.ics.uci.edu/dataset/601/ai4i)

## Chạy demo

1. Giải nén ZIP. Trên Windows, cài Python 3.12 rồi mở `CHAY_DEMO.bat`. Lần đầu cần mạng để cài thư viện.
2. Vào **Sàng lọc từ CSV**, chọn **Tệp minh họa** rồi bấm **Sàng lọc ngay**. Mở một ca trong danh sách ưu tiên để xem thông số và điểm. Bạn cũng có thể tải CSV riêng.
3. Vào **So sánh mô hình** để xem 12 cấu hình, thử thay đổi chi phí trên validation và xem kết quả của các ngưỡng đã chốt.
4. Vào **Kiểm chứng test**, bấm chạy 1.500 mẫu và lọc trường hợp bỏ sót hoặc báo nhầm.
5. Nếu dùng **Đọc ảnh bảng đo**, chọn ảnh mẫu hoặc tải ảnh cùng mẫu nhãn tiếng Anh, bấm **Đọc thông số**, sửa nếu cần rồi xác nhận. Tesseract cần có trong PATH trên máy chạy app. `packages.txt` khai báo các gói này cho môi trường triển khai Linux của Streamlit. Ảnh đầu vào là ảnh bảng đo.

Có thể chạy từ dòng lệnh trong thư mục vừa giải nén.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Để đưa lên Streamlit, đưa các tệp đã giải nén lên GitHub, giữ đúng thư mục `data/`, `reports/` và `.streamlit/`. Chọn `app.py` làm main file và Python 3.12. Trang web nạp mô hình và báo cáo đã lưu để suy luận.

## Dữ liệu và thí nghiệm

| Tập | Tổng | Bình thường | Hỏng |
|---|---:|---:|---:|
| Train | 7.000 | 6.763 | 237 |
| Validation | 1.500 | 1.449 | 51 |
| Test | 1.500 | 1.449 | 51 |
| Toàn bộ | 10.000 | 9.661 | 339 |

- Mục tiêu `Machine failure`. Đầu vào gồm `Type`, hai nhiệt độ (K), tốc độ quay (rpm), mô-men xoắn (Nm) và độ mòn (phút sử dụng). Loại `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` khỏi đầu vào.
- Chia phân tầng 70/15/15 với seed 42. Bốn thuật toán là Logistic Regression, Random Forest, XGBoost và LightGBM. Mỗi thuật toán chạy ở ba cấu hình gốc, tăng trọng số lớp hỏng và SMOTENC. XGBoost và LightGBM dùng `scale_pos_weight` trong cấu hình có trọng số.
- SMOTENC cân bằng riêng tập train lên 6.763 mẫu mỗi lớp. Chuẩn hóa biến số khi tìm láng giềng để các thang đo đóng góp hợp lý. `Type` được xử lý như biến phân loại. Mẫu tổng hợp là dữ liệu dùng để huấn luyện.
- Chọn LightGBM bản gốc vì có AP validation cao nhất, **0,7835**. AP của bản trọng số là **0,7314** và của SMOTENC là **0,7252**. Báo cáo lưu riêng AP và PR-AUC tích phân hình thang.
- Với chi phí quy ước `5 × bỏ sót + 1 × báo nhầm`, ngưỡng tối ưu trên validation là **19,8519%**, chi phí **64**. Ngưỡng tối đa F1 là **26,6037%**, chi phí **67**. Nếu đồng hạng, chọn ngưỡng cao hơn để giảm số ca phải kiểm tra. Tỷ lệ 5:1 dùng để minh họa, chưa có dữ liệu chi phí bảo trì thực tế.
- Các ngưỡng được lưu trước khi chạy test. Tập test đã được xem ở phiên bản demo trước, vì vậy kết quả được trình bày là so sánh trên cùng bộ kiểm thử AI4I. Kiểm chứng trên thiết bị thật cần dữ liệu riêng.

## Kết quả trên 1.500 mẫu test

| Quy tắc | Phát hiện đúng | Bỏ sót | Báo nhầm | Chi phí 5:1 | F1 |
|---|---:|---:|---:|---:|---:|
| Tối thiểu chi phí | 40 | 11 | 21 | 76 | 0,7143 |
| Tối đa F1 trên validation | 38 | 13 | 16 | 81 | 0,7238 |
| Ngưỡng 0,5 | 32 | 19 | 5 | 100 | 0,7273 |
| Luôn đoán bình thường | 0 | 51 | 0 | 255 | 0 |

PR-AUC test của LightGBM là **0,7993**, AP là **0,8006**. Các ngưỡng trên cùng một mô hình dùng cùng dãy điểm, vì vậy có cùng AP và PR-AUC. Accuracy ở ngưỡng 19,8519% là 97,87%, recall là 78,43% và precision là 65,57%.

## Tái lập kết quả

Mở `Nhom16_DeTai11_HuanLuyen.ipynb` để chạy lại trên Colab hoặc Jupyter. Nếu chạy bằng dòng lệnh trong thư mục demo, dùng lệnh dưới đây. Mỗi lần chạy nên dùng tên thư mục mới.

```bash
python -m pip install -r requirements.txt -r requirements-train.txt
python train.py --csv data/ai4i2020.csv --output trained_run --miss-cost 5
python -m unittest discover -s tests -v
```

Các tệp trong `reports/` chứa bảng so sánh 12 cấu hình, bảng chi phí theo ngưỡng và dự đoán từng mẫu theo `source_row`. Tệp `metadata.json` ghi ngưỡng đầy đủ, phiên bản thư viện và dấu kiểm SHA-256 của CSV gốc. Tệp `mo_hinh_demo.joblib` lưu mô hình cùng bước tiền xử lý. 
