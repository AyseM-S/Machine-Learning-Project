# c:\Users\emuwe\Desktop\ML_Project\main_controller.py

# Backend fonksiyonlarını import et
from train import run_training_pipeline
from predict import predict_new_data
import pandas as pd

# --- GUI'nin Eğitim İçin Çağıracağı Fonksiyon ---
def start_training(file_path, test_size_percent):
    """
    GUI'den gelen bilgilere göre eğitim sürecini başlatır.
    
    Args:
        file_path (str): Kullanıcının seçtiği dataset yolu.
        test_size_percent (int): Kullanıcının slider ile seçtiği test yüzdesi (örn: 20).

    Returns:
        dict: Eğitim metriklerini ve sonuçlarını içeren sözlük.
    """
    print(f"--- Kontrolcü: Eğitim süreci başlatılıyor ---")
    print(f"Dataset: {file_path}")
    print(f"Test Oranı: %{test_size_percent}")

    test_size_float = test_size_percent / 100.0
    
    # Backend'deki ana eğitim fonksiyonunu çağır
    training_results = run_training_pipeline(file_path=file_path, test_size=test_size_float)
    
    if training_results:
        print("--- Kontrolcü: Eğitim başarıyla tamamlandı. Sonuçlar GUI'ye gönderiliyor. ---")
        return training_results
    else:
        print("--- Kontrolcü: Eğitimde bir hata oluştu. ---")
        return None

# --- GUI'nin Tahmin İçin Çağıracağı Fonksiyon ---
def make_prediction(customer_data_dict):
    """
    GUI'deki input alanlarından gelen müşteri verisiyle tahmin yapar.

    Args:
        customer_data_dict (dict): Sütun adlarını ve kullanıcı girdilerini içeren sözlük.
                                   Örn: {'person_age': 25, 'person_income': 59000, ...}
    
    Returns:
        tuple: (tahmin_sonucu, olasılıklar) veya (None, None)
    """
    try:
        print("--- Kontrolcü: Tahmin süreci başlatılıyor ---")
        
        # Gelen sözlüğü pandas DataFrame'e çevir
        new_customer_df = pd.DataFrame([customer_data_dict])
        
        print("Gelen Veri:")
        print(new_customer_df)
        
        MODEL_FILE = 'logistic_regression_model.joblib'
        PREPROCESSOR_FILE = 'preprocessor.joblib'

        # Backend'deki tahmin fonksiyonunu çağır
        prediction, probabilities = predict_new_data(new_customer_df, MODEL_FILE, PREPROCESSOR_FILE)

        if prediction is not None:
            # Veri setindeki hedef: 0 = Ödeyen (Non-Default), 1 = Temerrüt (Default)
            approval_status = 'ONAYLANDI' if prediction[0] == 0 else 'REDDEDİLDİ'
            
            # Olasılıkları yüzde olarak formatla
            prob_default = probabilities[0][1] * 100
            prob_non_default = probabilities[0][0] * 100
            
            prediction_results = {
                'status': approval_status,
                'default_probability': f"{prob_default:.2f}%",
                'non_default_probability': f"{prob_non_default:.2f}%"
            }
            print(f"--- Kontrolcü: Tahmin tamamlandı. Sonuç: {prediction_results} ---")
            return prediction_results
            
    except Exception as e:
        print(f"Kontrolcüde tahmin hatası: {e}")
    
    return None

