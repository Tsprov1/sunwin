import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

def print_boxed(lines, pad=1):
    """
    In một hoặc nhiều dòng (str hoặc list[str]) trong khung ASCII.
    lines: str hoặc list[str]; pad: số khoảng trống xung quanh nội dung.
    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = [str(l) for l in lines]
    maxw = max(len(l) for l in lines) if lines else 0
    inner_w = maxw + pad*2
    top = "+" + "-" * (inner_w) + "+"
    print(top)
    for l in lines:
        print("|" + " " * pad + l.ljust(maxw) + " " * pad + "|")
    print(top)

def create_dataset(data, look_back=3):
    """
    Hàm này chuyển đổi một chuỗi dữ liệu thành một tập dữ liệu phù hợp cho việc học máy.
    Nó sẽ lấy 'look_back' số trước đó làm đầu vào (X) và số ngay sau đó làm đầu ra (y).
    Ví dụ: với look_back=3 và chuỗi [0 1, 1, 0], nó sẽ tạo ra cặp ( [0, 1, 1], 0 )
    """
    dataX, dataY = [], []
    if len(data) <= look_back:
        return np.array([]), np.array([])
        
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

class BinaryPredictor:
    """
    Lớp chính để huấn luyện mô hình và thực hiện dự đoán.
    """
    def __init__(self, look_back=3, model_type='logistic', random_state=42):
        """
        Khởi tạo predictor.
        :param look_back: Số lượng bước thời gian trước đó được sử dụng để dự đoán bước tiếp theo.
        :param model_type: Loại mô hình để sử dụng ('logistic' hoặc 'dummy').
        """
        if look_back < 1:
            raise ValueError("look_back phải lớn hơn hoặc bằng 1")
        self.look_back = look_back
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = None
        self.is_trained = False
        # New: bật/tắt luật xử lý trường hợp đặc biệt
        self.use_special_rules = True

        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == 'dummy':
            self.model = DummyClassifier(strategy='most_frequent')
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'ensemble':
            # placeholder: constructed later in improve_and_train or train
            self.model = None
        else:
            raise ValueError("model_type không hợp lệ. Vui lòng chọn 'logistic', 'dummy', 'random_forest' hoặc 'ensemble'.")

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
        self.is_trained = True
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

            # FIRST: kiểm tra luật đặc biệt nếu bật
            if getattr(self, "use_special_rules", False):
                is_special, sp_pred, sp_conf, reason = self.check_special_case(input_sequence)
                if is_special:
                    # trả về ngay theo luật đặc biệt (không qua model)
                    # in ngắn gọn lý do (đóng khung)
                    print_boxed([f"Rule applied: {reason}", f"Prediction: {int(sp_pred)}", f"Confidence: {sp_conf:.2%}" if sp_conf is not None else "Confidence: N/A"])
                    return int(sp_pred), float(sp_conf)

            # Nếu có scaler (khi dùng improve_and_train), áp dụng transform
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    input_array = self.scaler.transform(input_array)
                except Exception:
                    # nếu scaler không phù hợp, bỏ qua
                    pass
            
            # Dự đoán
            prediction = self.model.predict(input_array)[0]
            
            # Lấy xác suất (an toàn: ánh xạ nhãn sang chỉ số cột theo self.model.classes_)
            try:
                probabilities = None
                # VotingClassifier với voting='soft' hỗ trợ predict_proba
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(input_array)[0]
                # đảm bảo lấy đúng cột tương ứng với nhãn được dự đoán
                if probabilities is not None:
                    class_index = list(self.model.classes_).index(prediction)
                    confidence = probabilities[class_index]
                else:
                    confidence = None
            except AttributeError:
                # model không hỗ trợ predict_proba (ví dụ một số mô hình tùy chỉnh)
                confidence = None
            
            return prediction, confidence
        except NotFittedError:
            print("\nLỗi: Mô hình chưa được huấn luyện. Vui lòng gọi phương thức 'train' trước.")
            return None, None
        except Exception as e:
            print(f"\nĐã xảy ra lỗi trong quá trình dự đoán: {e}")
            return None, None

    # New: đánh giá mô hình trên tập test
    def evaluate_model(self, model, X_test, y_test):
        """
        Trả về dict các chỉ số và in ra báo cáo ngắn.
        """
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        print("\n--- Báo cáo đánh giá trên tập kiểm tra ---")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

    # New: cải tiến huấn luyện với train/test split, scaler, grid search và ensemble
    def improve_and_train(self, data, test_size=0.2, do_grid_search=True, use_scaler=True):
        """
        Cải tiến huấn luyện:
         - chia train/test (stratify)
         - chuẩn hóa (StandardScaler) nếu use_scaler
         - GridSearchCV cho Logistic/RandomForest nếu do_grid_search
         - hỗ trợ ensemble (model_type == 'ensemble')
         - in kết quả đánh giá
        """
        print(f"\nBắt đầu huấn luyện nâng cao cho '{self.model_type}'...")
        X, y = create_dataset(data, self.look_back)
        if X.shape[0] == 0:
            print("Lỗi: Không đủ dữ liệu để huấn luyện. Cần ít nhất look_back + 1 điểm dữ liệu.")
            return

        # chia theo stratify để giữ tỷ lệ lớp
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if len(np.unique(y))>1 else None
        )

        # scaler
        if use_scaler:
            self.scaler = StandardScaler().fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)

        # tùy chọn grid search
        model_to_use = self.model
        if self.model_type == 'logistic':
            if do_grid_search:
                param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
                gs = GridSearchCV(LogisticRegression(random_state=self.random_state, max_iter=2000), param_grid, cv=5, scoring='f1')
                gs.fit(X_train, y_train)
                model_to_use = gs.best_estimator_
                print(f"GridSearch best params: {gs.best_params_}")
            else:
                model_to_use = LogisticRegression(random_state=self.random_state, max_iter=2000)
                model_to_use.fit(X_train, y_train)

        elif self.model_type == 'random_forest':
            if do_grid_search:
                param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
                gs = GridSearchCV(RandomForestClassifier(random_state=self.random_state), param_grid, cv=5, scoring='f1')
                gs.fit(X_train, y_train)
                model_to_use = gs.best_estimator_
                print(f"GridSearch best params: {gs.best_params_}")
            else:
                model_to_use = RandomForestClassifier(random_state=self.random_state)
                model_to_use.fit(X_train, y_train)

        elif self.model_type == 'ensemble':
            # tạo base learners (có thể grid-search riêng nếu muốn)
            lr = LogisticRegression(random_state=self.random_state, max_iter=2000)
            rf = RandomForestClassifier(random_state=self.random_state, n_estimators=100)
            # nếu muốn, có thể tune lr/rf riêng bằng GridSearchCV trước khi tạo VotingClassifier
            ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
            ensemble.fit(X_train, y_train)
            model_to_use = ensemble

        elif self.model_type == 'dummy':
            model_to_use = DummyClassifier(strategy='most_frequent')
            model_to_use.fit(X_train, y_train)

        else:
            # fallback: dùng self.model nếu đã được cấu hình
            try:
                model_to_use.fit(X_train, y_train)
            except Exception as e:
                print(f"Lỗi khi huấn luyện mô hình: {e}")
                return

        # lưu model và đánh giá
        self.model = model_to_use
        self.is_trained = True

        # in cross-val score (ngắn)
        try:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
            print(f"Cross-val F1 trên tập train (5-fold): mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")
        except Exception:
            pass

        metrics = self.evaluate_model(self.model, X_test, y_test)
        return metrics

    # New: phát hiện các trường hợp đặc biệt và trả về (is_special, prediction, confidence, reason)
    def check_special_case(self, seq):
        """
        Heuristics đơn giản để xử lý các pattern đặc biệt:
         - all same
         - alternating (period 2)
         - repeating subpattern (period p)
         - single-flip (chỉ 1 khác biệt)
         - strong majority (>75%)
        Trả về tuple: (True/False, pred, confidence, reason)
        """
        seq = list(seq)
        n = len(seq)
        if n == 0:
            return False, None, None, None

        # all same
        if all(x == seq[0] for x in seq):
            return True, seq[0], 0.90, "all_same"

        # alternating / period 2 (010101 or 101010)
        if n >= 2:
            alt = True
            for i in range(2, n):
                if seq[i] != seq[i-2]:
                    alt = False
                    break
            if alt and seq[-1] != seq[-2]:
                # next will follow alternation: next = seq[-2]
                pred = seq[-2]
                return True, pred, 0.85, "alternating_period2"

        # repeating smaller period
        for p in range(1, max(2, n//2 + 1)):
            ok = True
            for i in range(n):
                if seq[i] != seq[i % p]:
                    ok = False
                    break
            if ok and p < n:
                pred = seq[n % p]
                return True, pred, 0.85, f"repeating_period_{p}"

        # single-flip (only one element differs)
        diffs = sum(1 for x in seq if x != round(sum(seq)/n))
        if diffs == 1:
            majority = 1 if sum(seq) >= (n/2) else 0
            return True, majority, 0.80, "single_flip_majority"

        # strong majority
        ones = sum(seq)
        if ones / n >= 0.75:
            return True, 1, ones / n, "strong_majority_ones"
        if (n - ones) / n >= 0.75:
            return True, 0, (n - ones) / n, "strong_majority_zeros"

        return False, None, None, None

    # New: thống kê dataset (số mẫu thực tế và tỉ lệ lớp)
    @staticmethod
    def dataset_stats(data, look_back=3):
        X, y = create_dataset(data, look_back)
        N = X.shape[0]
        if N == 0:
            print_boxed("Không có mẫu (kiểm tra look_back).")
            return {"N": 0}
        ones = int(np.sum(y == 1))
        zeros = int(np.sum(y == 0))
        lines = [
            "Thống kê dữ liệu",
            f"Số mẫu (N) = {N}  (mỗi mẫu = cửa sổ {look_back} -> 1 label)",
            f"Số lớp 1: {ones}  ({ones/N:.2%})",
            f"Số lớp 0: {zeros}  ({zeros/N:.2%})"
        ]
        if N < 10 * look_back:
            lines.append("Cảnh báo: Số mẫu nhỏ so với số đặc trưng. Cân nhắc giảm look_back hoặc thu thêm dữ liệu.")
        print()
        print_boxed(lines)
        return {"N": N, "ones": ones, "zeros": zeros}

    # New: tạo estimator theo model_type (dùng cho learning curve)
    @staticmethod
    def _estimator_for_type(model_type, random_state=42):
        if model_type == 'logistic':
            return LogisticRegression(random_state=random_state, max_iter=2000)
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=random_state, n_estimators=100)
        if model_type == 'ensemble':
            lr = LogisticRegression(random_state=random_state, max_iter=2000)
            rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
            return VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
        return DummyClassifier(strategy='most_frequent')

    # New: vẽ learning curve (dùng learning_curve)
    @staticmethod
    def plot_learning_curve(estimator, X, y, scaler=None, cv=5, scoring='f1'):
        if X.shape[0] == 0:
            print("Không có dữ liệu để vẽ learning curve.")
            return
        # nếu có scaler, gói vào pipeline để tránh leak
        if scaler is not None:
            pipe = Pipeline([('scaler', scaler), ('est', estimator)])
            est = pipe
        else:
            est = estimator
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                est, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            plt.figure(figsize=(6,4))
            plt.plot(train_sizes, train_mean, 'o-', label='Train')
            plt.plot(train_sizes, test_mean, 'o-', label='Validation')
            plt.title('Learning Curve')
            plt.xlabel('Training examples')
            plt.ylabel(scoring)
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Không thể vẽ learning curve: {e}")

    # New: simple ASCII learning curve printed to console
    @staticmethod
    def plot_learning_curve_ascii(estimator, X, y, scaler=None, cv=5, scoring='f1', width=40):
        """
        Tính learning_curve rồi in ra màn hình dạng bảng + thanh ASCII đơn giản.
        width: chiều rộng tối đa của thanh (số ký tự).
        """
        if X.shape[0] == 0:
            print("Không có dữ liệu để vẽ learning curve.")
            return
        # nếu có scaler, gói vào pipeline để tránh leak
        if scaler is not None:
            est = Pipeline([('scaler', scaler), ('est', estimator)])
        else:
            est = estimator

        try:
            train_sizes, train_scores, test_scores = learning_curve(
                est, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            # In bảng tóm tắt
            print("\n--- Learning curve (ASCII) ---")
            print(f"{'Train examples':>14} | {'Train '+scoring:>10} | {'Val '+scoring:>10} | {'Train bar':<{width}} | {'Val bar':<{width}}")
            for n, tr, te in zip(train_sizes.astype(int), train_mean, test_mean):
                # map scores [0,1] -> bar length
                tr_len = int(round(tr * width))
                te_len = int(round(te * width))
                tr_bar = "#" * tr_len + "-" * (width - tr_len)
                te_bar = "#" * te_len + "-" * (width - te_len)
                print(f"{n:14d} | {tr:10.3f} | {te:10.3f} | {tr_bar} | {te_bar}")
            print("--- end ---\n")
        except Exception as e:
            print(f"Không thể tính/vẽ learning curve ASCII: {e}")

def main():
    """
    Hàm chính để chạy giao diện dòng lệnh cho công cụ.
    """
    print_boxed("--- Công cụ Dự đoán sự xuất hiện của 0 và 1 ---")
    
   
    sample_data = [ 
0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1,0,0,0,
0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,
1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,0,
0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,
0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,
1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,
0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,
1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,
0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,0,0,1,1

  
                ]
    
    
    # Hỏi người dùng muốn dùng bao nhiêu bước nhìn lại (mặc định 3)
    try:
        raw = input("Nhập số bước nhìn lại (look_back) mong muốn (mặc định 3): ").strip()
        if raw == "":
            look_back_period = 3
        else:
            look_back_period = int(raw)
            if look_back_period < 1:
                raise ValueError
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng mặc định look_back = 3.")
        look_back_period = 3

    # Nếu look_back lớn hơn dữ liệu mẫu cho phép, hạ xuống tối đa có thể
    max_possible = max(1, len(sample_data) - 1)
    if look_back_period > max_possible:
        print(f"Cảnh báo: look_back quá lớn cho dữ liệu mẫu ({len(sample_data)}). Sẽ sử dụng {max_possible}.")
        look_back_period = max_possible
    
    # CHỌN LOẠI MÔ HÌNH: 'logistic' hoặc 'dummy'
    # 'dummy' sẽ luôn dự đoán giá trị phổ biến nhất trong dữ liệu, hữu ích để làm cơ sở so sánh.
    model_choice = 'logistic' # <-- BẠN CÓ THỂ THAY ĐỔI THÀNH 'dummy' ĐỂ THỬ NGHIỆM
    
    # Khởi tạo và huấn luyện mô hình
    predictor = BinaryPredictor(look_back=look_back_period, model_type=model_choice)
    predictor.train(sample_data)
    
    # New: in thống kê dataset
    predictor.dataset_stats(sample_data, look_back_period)

    print("\nBây giờ bạn có thể nhập một chuỗi để dự đoán giá trị tiếp theo.")
    print(f"Vui lòng nhập chính xác {predictor.look_back} số (0 hoặc 1), cách nhau bởi dấu cách.")
    print("Ví dụ: 1 0 1")
    print("Nhập 'exit' để thoát.")
    print("Lệnh thêm: 'rules on' / 'rules off' để bật/tắt luật xử lý trường hợp đặc biệt.")
    print("        'plot' hoặc 'curve' để vẽ learning curve cho loại mô hình hiện tại.")

    while True:
        user_input = input("\nNhập chuỗi của bạn: ").strip().lower()
        
        if user_input == 'exit':
            print("Cảm ơn bạn đã sử dụng công cụ!")
            break

        # xử lý lệnh vẽ learning curve
        if user_input in ('plot', 'curve'):
            X_all, y_all = create_dataset(sample_data, predictor.look_back)
            if X_all.shape[0] == 0:
                print("Không đủ dữ liệu để vẽ.")
                continue
            est = predictor._estimator_for_type(predictor.model_type, predictor.random_state)
            # nếu trước đó đã dùng scaler trong improve_and_train, dùng scaler ở đây
            scaler = predictor.scaler if getattr(predictor, 'scaler', None) is not None else None
            print("Đang vẽ learning curve... (dạng ASCII sẽ in ra màn hình)")
            predictor.plot_learning_curve_ascii(est, X_all, y_all, scaler=scaler)
            continue

        # New: bật/tắt luật
        if user_input in ('rules on', 'rules off'):
            predictor.use_special_rules = (user_input == 'rules on')
            print(f"Special-case rules set to: {predictor.use_special_rules}")
            continue
            
        try:
            # Xử lý đầu vào của người dùng
            parts = user_input.split()
            input_sequence = [int(p) for p in parts]

            if any(val not in [0, 1] for val in input_sequence):
                 print("Lỗi: Vui lòng chỉ nhập số 0 hoặc 1.")
                 continue

            if len(input_sequence) != predictor.look_back:
                print(f"Lỗi: Vui lòng nhập đúng {predictor.look_back} số.")
                continue

            # Thực hiện dự đoán
            prediction, confidence = predictor.predict(input_sequence)
            
            if prediction is not None:
                lines = [
                    f"Chuỗi đầu vào: {input_sequence}",
                    f"Dự đoán tiếp theo: {int(prediction)}",
                    f"Độ tin cậy: {('N/A' if confidence is None else f'{confidence:.2%}') }"
                ]
                print_boxed(lines)

        except ValueError:
            print("Lỗi: Đầu vào không hợp lệ. Vui lòng nhập các số 0 hoặc 1, cách nhau bởi dấu cách.")
        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn: {e}")

if __name__ == "__main__":
    main()
