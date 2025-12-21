import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Modeli kaydetmek ve yüklemek için

# preprocessing.py dosyamızdan gerekli fonksiyonları import ediyoruz
from preprocessing import create_preprocessor, load_csv_data

def run_training_pipeline(file_path, test_size=0.2, random_state=42):
    """
    Veri yükleme, ön işleme, model eğitimi ve değerlendirme adımlarını içeren
    tüm ML iş akışını çalıştırır. GUI tarafından çağrılmak üzere tasarlanmıştır.

    Args:
        file_path (str): Eğitilecek veriyi içeren CSV dosyasının yolu.
        test_size (float): Test setine ayrılacak veri oranı.
        random_state (int): Tekrarlanabilir sonuçlar için random state.

    Returns:
        dict: Eğitim süreci sonunda elde edilen metrikleri ve bilgileri içeren bir sözlük.
              Örn: {'accuracy': 0.87, 'report': '...', 'model_path': '...'}
        None: Bir hata oluşursa None döner.
    """
    # --- ADIM 1: Veri Yükleme ve Ön İşleme ---
    raw_df = load_csv_data(file_path)
    if raw_df is None:
        print(f"Veri seti yüklenemedi: {file_path}")
        return None

    # Bu veri setindeki hedef sütunun adı 'loan_status'
    # ve değerleri 'PAID' veya 'DEFAULT' olabilir. Bunu 0 ve 1'e çevirelim.
    TARGET = 'loan_status'

    # Ham veriyi özellikler (X) ve hedef (y) olarak ayır
    X_raw = raw_df.drop(columns=[TARGET])
    y = raw_df[TARGET]

    print("--- Veri Ön İşleme Başlatılıyor ---")
    # Ön işlemciyi oluştur
    preprocessor = create_preprocessor(X_raw)

    # --- ADIM 2: Veriyi Eğitim ve Test Setlerine Ayırma ---
    # Ham veriyi eğitim ve test setlerine ayırıyoruz.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Ham eğitim seti boyutu: {X_train_raw.shape}")
    print(f"Ham test seti boyutu: {X_test_raw.shape}")
    print("-" * 30)
    
    # ÖNEMLİ: Ön işlemciyi SADECE eğitim verisiyle eğitiyoruz (fit) ve sonra eğitim verisini dönüştürüyoruz (transform).
    # Bu, test verisinden herhangi bir bilginin eğitim sürecine sızmasını engeller.
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # --- ADIM 3: Modeli Eğitme ---
    print("--- Model Eğitimi Başlatılıyor ---")
    # Lojistik Regresyon modelini oluşturuyoruz
    model = LogisticRegression(random_state=random_state)

    # Modeli eğitim verisiyle eğitiyoruz
    model.fit(X_train, y_train)
    print("Model başarıyla eğitildi.")

    # Eğitilmiş modeli diske kaydediyoruz
    model_filename = 'logistic_regression_model.joblib'
    joblib.dump(model, model_filename)

    # Eğitilmiş ÖN İŞLEMCİYİ de diske kaydediyoruz. Bu çok önemli!
    preprocessor_filename = 'preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Ön işlemci '{preprocessor_filename}' olarak kaydedildi.")
    print(f"Model '{model_filename}' olarak kaydedildi.")
    print("-" * 30)


    # --- ADIM 4: Modeli Değerlendirme ---
    print("--- Model Değerlendirmesi ---")
    # Modelin test verisi üzerinde tahmin yapmasını sağlıyoruz
    y_pred = model.predict(X_test)

    # Modelin doğruluğunu (accuracy) hesaplıyoruz
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Modelin Test Doğruluğu: {accuracy:.2f}")
    print("\nSınıflandırma Raporu:")
    print(report)

    # GUI'ye göndermek için sonuçları bir sözlükte topla
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'model_path': model_filename,
        'preprocessor_path': preprocessor_filename,
        'training_data_shape': X_train.shape,
        'test_data_shape': X_test.shape
    }
    return results


if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında varsayılan ayarlarla eğitimi başlatır.
    # Bu, GUI olmadan da test yapabilmenizi sağlar.
    print("--- train.py doğrudan çalıştırıldı, varsayılan eğitim başlıyor ---")
    file_path = 'credit_risk.csv'
    run_training_pipeline(file_path=file_path)
