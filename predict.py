import pandas as pd
import joblib

def predict_new_data(new_data_df, model_path, preprocessor_path):
    """
    Kayıtlı modeli ve ön işlemciyi kullanarak yeni veri üzerinde tahmin yapar.

    Args:
        new_data_df (pd.DataFrame): Tahmin yapılacak yeni veriyi içeren DataFrame.
        model_path (str): Kayıtlı modelin dosya yolu.
        preprocessor_path (str): Kayıtlı ön işlemcinin dosya yolu.
    """
    try:
        # Kayıtlı modeli ve ön işlemciyi yükle
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        print(f"Model '{model_path}' başarıyla yüklendi.")
        print(f"Ön işlemci '{preprocessor_path}' başarıyla yüklendi.")

        # Yeni veriyi, yüklenen ön işlemci ile DÖNÜŞTÜR (transform)
        X_processed = preprocessor.transform(new_data_df)

        # Tahmin yap ve olasılıkları hesapla
        prediction = model.predict(X_processed)
        prediction_proba = model.predict_proba(X_processed)

        return prediction, prediction_proba

    except FileNotFoundError:
        print(f"Hata: Model veya ön işlemci dosyası bulunamadı.")
        return None, None
    except Exception as e:
        print(f"Tahmin sırasında bir hata oluştu: {e}")
        return None, None

if __name__ == "__main__":
    # Tahmin yapmak istediğimiz yeni müşteri verisi (credit_risk.csv sütunlarına uygun)
    new_customer = pd.DataFrame({
        'person_age': [25],
        'person_income': [59000],
        'person_home_ownership': ['RENT'],
        'person_emp_length': [3.0],
        'loan_intent': ['PERSONAL'],
        'loan_grade': ['D'],
        'loan_amnt': [35000],
        'loan_int_rate': [16.02],
        'loan_percent_income': [0.59],
        'cb_person_default_on_file': ['Y'],
        'cb_person_cred_hist_length': [3]
    })

    MODEL_FILE = 'logistic_regression_model.joblib'
    PREPROCESSOR_FILE = 'preprocessor.joblib'

    print("--- Yeni Müşteri İçin Kredi Onay Tahmini ---")
    print("Gelen Veri:")
    print(new_customer)
    print("-" * 30)

    prediction, probabilities = predict_new_data(new_customer, MODEL_FILE, PREPROCESSOR_FILE)

    if prediction is not None:
        print("-" * 30)
        # Veri setindeki hedef: 0 = Ödeyen (Non-Default), 1 = Temerrüt (Default)
        approval_status = 'ONAYLANDI' if prediction[0] == 0 else 'REDDEDİLDİ'
        print(f"Tahmin Sonucu: {approval_status}")
        print(f"Reddedilme Olasılığı (Default) (%): {probabilities[0][1] * 100:.2f}")
        print(f"Onaylanma Olasılığı (Non-Default) (%): {probabilities[0][0] * 100:.2f}")
