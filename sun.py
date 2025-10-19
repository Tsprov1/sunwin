import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError

class BinaryPredictor:
    """
    Một lớp để dự đoán giá trị nhị phân tiếp theo (0 hoặc 1) dựa trên một chuỗi dữ liệu lịch sử.
    """
    def __init__(self, look_back_period=3, model_type='logistic'):
        """
        Khởi tạo predictor.
        :param look_back_period: Số lượng các điểm dữ liệu trước đó được sử dụng để dự đoán điểm tiếp theo.
        :param model_type: Loại mô hình sử dụng, 'logistic' hoặc 'dummy'.
        """
        if look_back_period < 1:
            raise ValueError("look_back_period phải lớn hơn hoặc bằng 1.")
        self.look_back = look_back_period
        
        if model_type == 'logistic':
            self.model = LogisticRegression(solver='liblinear')
        elif model_type == 'dummy':
            self.model = DummyClassifier(strategy='most_frequent')
        else:
            raise ValueError("model_type phải là 'logistic' hoặc 'dummy'.")
            
        self.is_fitted = False

    def _create_dataset(self, data):
        """
        Chuyển đổi một chuỗi dữ liệu phẳng thành một tập dữ liệu có giám sát.
        """
        X, y = [], []
        if len(data) <= self.look_back:
            return np.array(X), np.array(y)
            
        for i in range(len(data) - self.look_back):
            feature = data[i : i + self.look_back]
            target = data[i + self.look_back]
            X.append(feature)
            y.append(target)
        return np.array(X), np.array(y)

    def train(self, data):
        """
        Huấn luyện mô hình trên dữ liệu được cung cấp.
        :param data: Một danh sách hoặc mảng các số 0 và 1.
        """
        print(f"Bắt đầu huấn luyện với {len(data)} điểm dữ liệu...")
        X, y = self._create_dataset(data)
        
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            print("Cảnh báo: Không đủ dữ liệu hoặc chỉ có một lớp trong dữ liệu. Không thể huấn luyện mô hình.")
            self.is_fitted = False
            return
            
        self.model.fit(X, y)
        self.is_fitted = True
        print("Huấn luyện hoàn tất.")

    def predict(self, sequence):
        """
        Dự đoán giá trị tiếp theo dựa trên một chuỗi đầu vào.
        :param sequence: Một danh sách hoặc mảng các giá trị 0 và 1 có độ dài bằng look_back_period.
        :return: Một tuple chứa (dự đoán, xác suất của dự đoán).
        """
        if not self.is_fitted:
            raise NotFittedError("Mô hình chưa được huấn luyện. Vui lòng gọi phương thức train() trước.")
        
        if len(sequence) != self.look_back:
            raise ValueError(f"Độ dài của chuỗi đầu vào phải là {self.look_back}.")

        sequence_array = np.array(sequence).reshape(1, -1)
        prediction = self.model.predict(sequence_array)[0]
        
        try:
            probabilities = self.model.predict_proba(sequence_array)[0]
            confidence = probabilities[prediction]
        except AttributeError:
            confidence = 1.0 
            
        return prediction, confidence

def main():
    """
    Hàm chính để chạy công cụ dự đoán.
    """
    # == DỮ LIỆU TỔNG HỢP TỪ TẤT CẢ 7 HÌNH ẢNH ==
    full_data = [
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,1,0,1,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,0,0,1,1,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,
        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1

        
    ]

    look_back = 3
    model_choice = 'logistic' 
    
    predictor = BinaryPredictor(look_back_period=look_back, model_type=model_choice)
    predictor.train(full_data)

    print("\n--- Công cụ Dự đoán Nhị phân (Dữ liệu Hoàn chỉnh) ---")
    print(f"Nhập một chuỗi gồm {look_back} số 0 hoặc 1, cách nhau bởi dấu cách (ví dụ: 1 0 1).")
    print("Nhập 'exit' để thoát.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'exit':
            break

        try:
            input_sequence = [int(x) for x in user_input.split()]
            
            if len(input_sequence) != look_back:
                print(f"Lỗi: Vui lòng nhập đúng {look_back} số.")
                continue

            prediction, confidence = predictor.predict(input_sequence)
            print(f"   -> Dự đoán số tiếp theo là: {prediction} (Độ tin cậy: {confidence:.2%})")

        except ValueError:
            print("Lỗi: Vui lòng chỉ nhập các số 0 hoặc 1, cách nhau bởi dấu cách.")
        except NotFittedError:
            print("Lỗi: Mô hình chưa được huấn luyện do không đủ dữ liệu.")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()

