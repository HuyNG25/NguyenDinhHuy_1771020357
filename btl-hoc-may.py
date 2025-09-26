# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Khám phá và Tải Dữ liệu
#------------------------------------------------------------------------------------------------------
print("Bước 1: Khám phá và Tải Dữ liệu")
file_path = 'arxiv-metadata-oai-snapshot.json'
data_list = []
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # Giới hạn 50000 dòng để xử lý nhanh hơn, tránh quá tải bộ nhớ
        for i, line in enumerate(f):
            if i >= 50000:
                break
            data_list.append(json.loads(line))
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Tạo DataFrame từ dữ liệu đã đọc
df = pd.DataFrame(data_list)
print(f"Số lượng mẫu được tải: {len(df)}")
print(f"Cấu trúc dữ liệu:\n{df.info()}")

# 2. Tiền xử lý văn bản
#------------------------------------------------------------------------------------------------------
print("\nBước 2: Tiền xử lý văn bản")
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_abstract'] = df['abstract'].apply(preprocess_text)

# Tạo nhãn từ cột 'categories' và lọc các nhãn không phổ biến
df['label'] = df['categories'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'unknown')
label_counts = df['label'].value_counts()
top_labels = label_counts[label_counts > 100].index
df_filtered = df[df['label'].isin(top_labels)].copy()
print(f"\nSố lượng mẫu sau khi lọc: {len(df_filtered)}")
print("Phân phối nhãn:\n", df_filtered['label'].value_counts())

# 3. Vector hóa văn bản với TF-IDF
#------------------------------------------------------------------------------------------------------
print("\nBước 3: Vector hóa văn bản với TF-IDF")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df_filtered['clean_abstract'])
y = df_filtered['label']
print(f"Kích thước ma trận TF-IDF: {X.shape}")

# 4. Chia tập dữ liệu
#------------------------------------------------------------------------------------------------------
print("\nBước 4: Chia tập dữ liệu")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Huấn luyện và Đánh giá các mô hình
#------------------------------------------------------------------------------------------------------
print("\nBước 5: Huấn luyện và Đánh giá các mô hình")

# Mô hình 1: Naive Bayes
print("\n--- Huấn luyện Mô hình Naive Bayes ---")
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Độ chính xác của Naive Bayes: {accuracy_nb:.4f}")
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_nb, zero_division=0))

# Mô hình 2: Logistic Regression
print("\n--- Huấn luyện Mô hình Logistic Regression ---")
model_lr = LogisticRegression(max_iter=5000, solver='liblinear')
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Độ chính xác của Logistic Regression: {accuracy_lr:.4f}")
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_lr, zero_division=0))

# Mô hình 3: Support Vector Machine (SVM)
print("\n--- Huấn luyện Mô hình SVM ---")
# Sử dụng 'C' nhỏ để tránh quá khớp và tăng tốc độ xử lý
model_svm = SVC(kernel='linear', C=1.0)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Độ chính xác của SVM: {accuracy_svm:.4f}")
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred_svm, zero_division=0))

# 6. Trực quan hóa và Phân tích kết quả
#------------------------------------------------------------------------------------------------------
print("\nBước 6: Trực quan hóa và Phân tích kết quả")

# Tạo DataFrame tổng hợp kết quả
results_df = pd.DataFrame({
    'Model': ['Naive Bayes', 'Logistic Regression', 'SVM'],
    'Accuracy': [accuracy_nb, accuracy_lr, accuracy_svm]
})

print("\nBảng so sánh độ chính xác của các mô hình:")
print(results_df)

# Trực quan hóa so sánh
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('So sánh Độ chính xác của các Mô hình Phân loại')
plt.ylabel('Độ chính xác')
plt.xlabel('Mô hình')
plt.ylim(0, 1.0)
plt.show()

# Trực quan hóa ma trận nhầm lẫn cho mô hình tốt nhất (ví dụ: Logistic Regression)
print("\nMa trận nhầm lẫn cho Mô hình Logistic Regression (thường là tốt nhất):")
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=top_labels)
plt.figure(figsize=(15, 12))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=top_labels, yticklabels=top_labels)
plt.title('Ma trận nhầm lẫn (Confusion Matrix) của Logistic Regression')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.show()