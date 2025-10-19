import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
import warnings

# Tắt các cảnh báo không quan trọng để giao diện gọn gàng hơn
warnings.filterwarnings('ignore', category=UserWarning)

def create_dataset(data, look_back=3):
    """
    Hàm này chuyển đổi một chuỗi dữ liệu thành một tập dữ liệu phù hợp cho việc học máy.
    Nó sẽ lấy 'look_back' số trước đó làm đầu vào (X) và số ngay sau đó làm đầu ra (y).
    Ví dụ: với look_back=3 và chuỗi [0, 1, 1, 0], nó sẽ tạo ra cặp ( [0, 1, 1], 0 )
    """
    dataX, dataY = [], []
    if len(data) <= look_back:
        return np.array([]), np.array([])
        
    for i in range(len(data) - look_back):
        # Lấy một đoạn gồm 'look_back' phần tử
        a = data[i:(i + look_back)]
        dataX.append(a)
        # Lấy phần tử ngay sau đoạn đó làm kết quả
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

class BinaryPredictor:
    """
    Lớp chính để huấn luyện mô hình và thực hiện dự đoán.
    """
    def __init__(self, look_back=3, model_type='logistic'):
        """
        Khởi tạo predictor.
        :param look_back: Số lượng bước thời gian trước đó được sử dụng để dự đoán bước tiếp theo.
        :param model_type: Loại mô hình để sử dụng ('logistic' hoặc 'dummy').
        """
        if look_back < 1:
            raise ValueError("look_back phải lớn hơn hoặc bằng 1")
        self.look_back = look_back
        self.model_type = model_type

        if self.model_type == 'logistic':
            self.model = LogisticRegression()
        elif self.model_type == 'dummy':
            # DummyClassifier là một mô hình cơ sở, hữu ích để so sánh.
            # Chiến lược 'most_frequent' sẽ luôn dự đoán lớp phổ biến nhất trong tập huấn luyện.
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            raise ValueError("model_type không hợp lệ. Vui lòng chọn 'logistic' hoặc 'dummy'.")

    def train(self, data):
        """
        Huấn luyện mô hình với dữ liệu được cung cấp.
        :param data: Một danh sách hoặc mảng các số 0 và 1.
        """
        print(f"Bắt đầu huấn luyện mô hình '{self.model_type}' với {len(data)} điểm dữ liệu...")
        X, y = create_dataset(data, self.look_back)
        
        if X.shape[0] == 0:
            print("Lỗi: Không đủ dữ liệu để huấn luyện. Cần ít nhất look_back + 1 điểm dữ liệu.")
            return

        self.model.fit(X, y)
        print("Huấn luyện hoàn tất!")

    def predict(self, input_sequence):
        """
        Dự đoán giá trị tiếp theo dựa trên một chuỗi đầu vào.
        :param input_sequence: Một danh sách hoặc mảng các số 0 và 1 có độ dài bằng 'look_back'.
        :return: Một tuple chứa (dự đoán, xác suất của dự đoán).
        """
        if len(input_sequence) != self.look_back:
            raise ValueError(f"Chuỗi đầu vào phải có độ dài chính xác là {self.look_back}")

        try:
            # Chuyển đổi đầu vào thành định dạng numpy phù hợp
            input_array = np.array(input_sequence).reshape(1, -1)
            
            # Dự đoán
            prediction = self.model.predict(input_array)[0]
            
            # Lấy xác suất
            probabilities = self.model.predict_proba(input_array)[0]
            
            # Xác suất của kết quả được dự đoán
            confidence = probabilities[prediction]
            
            return prediction, confidence
        except NotFittedError:
            print("\nLỗi: Mô hình chưa được huấn luyện. Vui lòng gọi phương thức 'train' trước.")
            return None, None
        except Exception as e:
            print(f"\nĐã xảy ra lỗi trong quá trình dự đoán: {e}")
            return None, None

def main():
    """
    Hàm chính để chạy giao diện dòng lệnh cho công cụ.
    """
    print("--- Công cụ Dự đoán sự xuất hiện của 0 và 1 ---")
    
    # Dữ liệu mẫu: một chuỗi các số 0 và 1
    # BẠN NÊN THAY THẾ DỮ LIỆU NÀY BẰNG DỮ LIỆU THỰC TẾ CỦA MÌNH ĐỂ CÓ KẾT QUẢ TỐT HƠN
    sample_data = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    
    # Thiết lập số bước nhìn lại quá khứ để dự đoán
    look_back_period = 3
    
    # CHỌN LOẠI MÔ HÌNH: 'logistic' hoặc 'dummy'
    # 'dummy' sẽ luôn dự đoán giá trị phổ biến nhất trong dữ liệu, hữu ích để làm cơ sở so sánh.
    model_choice = 'logistic' # <-- BẠN CÓ THỂ THAY ĐỔI THÀNH 'dummy' ĐỂ THỬ NGHIỆM
    
    # Khởi tạo và huấn luyện mô hình
    predictor = BinaryPredictor(look_back=look_back_period, model_type=model_choice)
    predictor.train(sample_data)
    
    print("\nBây giờ bạn có thể nhập một chuỗi để dự đoán giá trị tiếp theo.")
    print(f"Vui lòng nhập chính xác {look_back_period} số (0 hoặc 1), cách nhau bởi dấu cách.")
    print("Ví dụ: 1 0 1")
    print("Nhập 'exit' để thoát.")

    while True:
        user_input = input("\nNhập chuỗi của bạn: ").strip().lower()
        
        if user_input == 'exit':
            print("Cảm ơn bạn đã sử dụng công cụ!")
            break
            
        try:
            # Xử lý đầu vào của người dùng
            parts = user_input.split()
            input_sequence = [int(p) for p in parts]

            if any(val not in [0, 1] for val in input_sequence):
                 print("Lỗi: Vui lòng chỉ nhập số 0 hoặc 1.")
                 continue

            if len(input_sequence) != look_back_period:
                print(f"Lỗi: Vui lòng nhập đúng {look_back_period} số.")
                continue

            # Thực hiện dự đoán
            prediction, confidence = predictor.predict(input_sequence)
            
            if prediction is not None:
                print(f"   -> Chuỗi đầu vào: {input_sequence}")
                print(f"   => Dự đoán số tiếp theo là: {prediction}")
                print(f"   -> Độ tin cậy (xác suất): {confidence:.2%}")

        except ValueError:
            print("Lỗi: Đầu vào không hợp lệ. Vui lòng nhập các số 0 hoặc 1, cách nhau bởi dấu cách.")
        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()

