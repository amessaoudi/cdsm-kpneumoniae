# ============================================================
# SAUVEGARDE DES MODÈLES — À exécuter dans Colab
# Après avoir exécuté le pipeline ML complet
# ============================================================
# Ce script sauvegarde les modèles XGBoost entraînés
# pour les utiliser dans l'application Streamlit
# ============================================================

import joblib
import os

# Créer le dossier models dans Drive
model_dir = '/content/drive/MyDrive/CDSM_models'
os.makedirs(model_dir, exist_ok=True)

print("💾 Sauvegarde des modèles XGBoost...")

for ab, model in best_models.items():
    # Sauvegarder le modèle
    model_path = f'{model_dir}/{ab}_xgb.joblib'
    joblib.dump(model, model_path)

    # Sauvegarder les features
    features_path = f'{model_dir}/{ab}_features.joblib'
    joblib.dump(all_results[ab]['features'], features_path)

    print(f"  ✅ {ab} sauvegardé")

# Sauvegarder les poids optimaux et l'efficacité régionale
config = {
    'best_w1':       best_w1,
    'regional_eff':  regional_eff,
    'antibiotics':   list(best_models.keys()),
}
joblib.dump(config, f'{model_dir}/config.joblib')

print(f"\n✅ Tous les modèles sauvegardés dans :")
print(f"   {model_dir}")
print(f"\nFichiers créés :")
for f in os.listdir(model_dir):
    size = os.path.getsize(f'{model_dir}/{f}') / 1024
    print(f"  📁 {f} ({size:.1f} KB)")

print(f"\n🚀 PROCHAINE ÉTAPE :")
print(f"1. Téléchargez le dossier 'CDSM_models' depuis Google Drive")
print(f"2. Placez-le dans le même dossier que app.py")
print(f"3. Renommez-le 'models'")
print(f"4. Lancez : streamlit run app.py")
