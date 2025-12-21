import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Optional

def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Verilen dosya yolundan bir CSV dosyasını yükler ve pandas DataFrame olarak döndürür.
    
    Args:
        file_path (str): Yüklenecek CSV dosyasının yolu.
        
    Returns:
        Optional[pd.DataFrame]: Yüklenen veriyi içeren DataFrame veya hata durumunda None.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"'{file_path}' başarıyla yüklendi. Veri boyutu: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı - {file_path}")
        return None
    except Exception as e:
        print(f"HATA: Veri yüklenirken bir hata oluştu: {e}")
        return None

# Örnek kullanım (test için):
# sample_df = load_csv_data('path/to/your/dataset.csv')
# if sample_df is not None:
#     print(sample_df.head())

def create_preprocessor(X_features: pd.DataFrame, numeric_scaler: str = 'standard') -> ColumnTransformer:
    """
    Veri setine dayalı bir ColumnTransformer ön işlemcisi oluşturur.
    Bu işlemci, eksik verileri doldurur (imputation), kategorik verileri dönüştürür (one-hot encoding)
    ve sayısal verileri ölçeklendirir.

    Args:
        X_features (pd.DataFrame): Modelde kullanılacak özellik (feature) DataFrame'i (eğitilmemiş).
        numeric_scaler (str): Sayısal özellikler için kullanılacak ölçeklendirici. 
                              'standard' veya 'minmax' olabilir.

    Returns:
        ColumnTransformer: Henüz eğitilmemiş, hazır ön işlemci nesnesi.
    """
    
    # Veri türlerine göre sütun adlarını belirle
    numeric_features = X_features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_features.select_dtypes(exclude=np.number).columns.tolist()

    # Seçilen scaler'a göre pipeline oluştur
    if numeric_scaler == 'minmax':
        scaler = MinMaxScaler()
    else: # Varsayılan olarak StandardScaler kullan
        scaler = StandardScaler()

    # Sayısal sütunlar için bir ön işleme ardışık düzeni (pipeline) oluştur
    # 1. Adım: Eksik değerleri sütunun ortanca (median) değeri ile doldur.
    # 2. Adım: Veriyi standartlaştır (StandardScaler).
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])

    # Kategorik sütunlar için bir ön işleme ardışık düzeni (pipeline) oluştur
    # 1. Adım: Eksik değerleri en sık tekrar eden değerle doldur.
    # 2. Adım: Kategorileri one-hot encoding ile sayısal forma dönüştür.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer ile bu iki ardışık düzeni birleştir
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Kalan sütunları (varsa) olduğu gibi bırak
    )
    return preprocessor

def _run_preprocessing_tests():
    """
    create_preprocessor fonksiyonunun doğruluğunu kontrol etmek için birim testlerini çalıştırır.
    Bu test, hem sayısal ölçeklendirmeyi hem de kategorik kodlamayı doğrular.
    """
    print("="*50)
    print("--- BAŞLANGIÇ: Ön İşleme Birim Testleri ---")
    print("="*50)

    # --- TEST: One-Hot Encoding ve StandardScaler Testi ---
    print("\n--- TEST: One-Hot Encoding ve StandardScaler Testi Başlatılıyor ---")

    # Test için basit bir DataFrame oluştur
    df_test = pd.DataFrame({
        'age': [20, 30, 40], # Sayısal
        'city': ['Ankara', 'Istanbul', 'Ankara'], # Kategorik
        'gender': ['M', 'F', 'M'], # Kategorik
    })

    try:
        # 1. Ön işlemciyi test verisiyle oluştur
        preprocessor = create_preprocessor(df_test)
        
        # 2. Veriyi hem eğit (fit) hem de dönüştür (transform)
        X_processed = preprocessor.fit_transform(df_test)

        # BEKLENEN SÜTUN SAYISI HESABI:
        # ColumnTransformer, 'transformers' listesindeki sırayla işlem yapar.
        # Bizim fonksiyonumuzda sıra: ('num', numeric_features), ('cat', categorical_features)
        # 1. Sayısal Sütunlar ('age'): 1 sütun
        # 2. Kategorik Sütunlar ('city': Ankara, Istanbul): 2 sütun
        # 3. Kategorik Sütunlar ('gender': F, M): 2 sütun
        # Toplam: 1 + 2 + 2 = 5 sütun beklenir.
        EXPECTED_SHAPE = (len(df_test), 5) 

        print("\nİşlenmiş matris (NumPy array):")
        print(X_processed) 
        print(f"\nMatris Şekli: {X_processed.shape} (Beklenti: {EXPECTED_SHAPE})")
        
        assert X_processed.shape == EXPECTED_SHAPE, f"Şekil hatası: {X_processed.shape} != {EXPECTED_SHAPE}"
        print("✅ TEST BAŞARILI: Matris şekli beklendiği gibi.")

    except Exception as e:
        print(f"❌ TEST BAŞARISIZ: Test sırasında bir hata oluştu: {e}")

# Bu dosya doğrudan çalıştırıldığında testleri başlatır.
if __name__ == "__main__":
    _run_preprocessing_tests()

def preprocess_data(df, preprocessor, target_column):
    """
    Verilen bir DataFrame'i, daha önce eğitilmiş bir ön işlemci kullanarak dönüştürür.

    Args:
        df (pd.DataFrame): İşlenecek ham veri.
        preprocessor (ColumnTransformer): Eğitilmiş ön işlemci nesnesi.
        target_column (str): Hedef değişkenin (y) sütun adı.

    Returns:
        np.ndarray: İşlenmiş özellik matrisi (X).
        pd.Series: Hedef değişken (y).
    """
    X_processed = preprocessor.transform(df.drop(columns=[target_column]))
    y = df[target_column]
    
    print("Ön işleme tamamlandı.")
    print(f"İşlenmiş veri boyutu: {X_processed.shape}")
    
    return X_processed, y
