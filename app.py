# ============================================================
# 🏥 CDSM — Application Web Streamlit
# Clinical Decision Support for K. pneumoniae
# Version 1.0 | AMR Data Challenge 2026
# ============================================================
# Installation : pip install streamlit xgboost scikit-learn
#                pip install imbalanced-learn joblib plotly
# Lancement    : streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from datetime import datetime

# ────────────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CDSM — K. pneumoniae Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Couleurs principales */
    :root {
        --primary:   #1565C0;
        --success:   #2E7D32;
        --warning:   #E65100;
        --danger:    #B71C1C;
        --light-bg:  #F8F9FA;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0 0; }

    /* Cartes résultats */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .result-susceptible {
        background: #E8F5E9;
        border-color: #2E7D32;
    }
    .result-resistant {
        background: #FFEBEE;
        border-color: #B71C1C;
    }
    .result-uncertain {
        background: #FFF3E0;
        border-color: #E65100;
    }

    /* Recommandation */
    .recommendation-box {
        background: linear-gradient(135deg, #1565C0 0%, #1976D2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .recommendation-box h2 { font-size: 1.8rem; margin: 0; }
    .recommendation-box p  { font-size: 1rem; opacity: 0.9; }

    .second-line-box {
        background: #E3F2FD;
        color: #1565C0;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #1565C0;
    }

    /* Warning box */
    .warning-box {
        background: #FFF3E0;
        border: 1px solid #FF9800;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .danger-box {
        background: #FFEBEE;
        border: 1px solid #F44336;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Disclaimer */
    .disclaimer {
        background: #F5F5F5;
        border: 1px solid #BDBDBD;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #616161;
        margin-top: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9E9E9E;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #E0E0E0;
    }

    /* Cacher menu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# DONNÉES DE CONFIGURATION
# ────────────────────────────────────────────────────────────

ANTIBIOTICS = ['Meropenem', 'Imipenem', 'Ceftriaxone',
               'Ceftazidime', 'Amikacin']

ANTIBIOTIC_INFO = {
    'Meropenem':   {'class': 'Carbapenem',        'color': '#E53935', 'emoji': '💊'},
    'Imipenem':    {'class': 'Carbapenem',        'color': '#E53935', 'emoji': '💊'},
    'Ceftriaxone': {'class': 'Céphalosporine 3G', 'color': '#FB8C00', 'emoji': '💉'},
    'Ceftazidime': {'class': 'Céphalosporine 3G', 'color': '#FB8C00', 'emoji': '💉'},
    'Amikacin':    {'class': 'Aminoglycoside',    'color': '#43A047', 'emoji': '🔬'},
}

# Efficacité régionale LMIC (issue de l'analyse ATLAS 2021-2024)
REGIONAL_EFF = {
    'Meropenem':   0.703,
    'Imipenem':    0.694,
    'Ceftriaxone': 0.500,
    'Ceftazidime': 0.436,
    'Amikacin':    0.747,
}

# Résistance régionale par région (heatmap ATLAS)
REGIONAL_RESISTANCE = {
    'MENA_Africa':   {'Meropenem': 28.4, 'Imipenem': 31.2, 'Ceftazidime': 48.3, 'Amikacin': 22.1, 'Ceftriaxone': 42.1},
    'Asia_Pacific':  {'Meropenem': 18.7, 'Imipenem': 21.9, 'Ceftazidime': 44.1, 'Amikacin': 17.6, 'Ceftriaxone': 38.4},
    'Latin_America': {'Meropenem': 14.3, 'Imipenem': 17.8, 'Ceftazidime': 39.2, 'Amikacin': 12.4, 'Ceftriaxone': 32.1},
    'Eastern_Europe':{'Meropenem': 12.1, 'Imipenem': 15.3, 'Ceftazidime': 37.8, 'Amikacin': 11.8, 'Ceftriaxone': 29.8},
    'Western_HIC':   {'Meropenem':  7.4, 'Imipenem':  7.4, 'Ceftazidime': 27.9, 'Amikacin':  7.2, 'Ceftriaxone': 26.7},
}

LMIC_COUNTRIES = [
    'Tunisia', 'Morocco', 'Algeria', 'Egypt', 'Jordan', 'Lebanon',
    'Saudi Arabia', 'India', 'China', 'Brazil', 'Mexico', 'Argentina',
    'Colombia', 'Thailand', 'South Africa', 'Turkey', 'Russia', 'Chile',
    'Kuwait', 'Malaysia', 'Philippines', 'Vietnam', 'Pakistan', 'Nigeria',
    'Kenya', 'Ghana', 'Cameroon', 'Uganda', 'Romania', 'Serbia', 'Qatar',
    'Ivory Coast', 'Libya', 'Syria', 'Iraq', 'Yemen', 'Sudan', 'Ethiopia',
    'Tanzania', 'Senegal', 'Côte d\'Ivoire', 'DR Congo', 'Zimbabwe',
    'Zambia', 'Rwanda', 'Bangladesh', 'Nepal', 'Myanmar', 'Cambodia',
    'Indonesia', 'Bolivia', 'Paraguay', 'Peru', 'Ecuador', 'Guatemala',
    'Honduras', 'El Salvador', 'Nicaragua', 'Panama', 'Dominican Republic',
]

HIC_COUNTRIES = [
    'United States', 'France', 'Germany', 'United Kingdom', 'Spain',
    'Italy', 'Belgium', 'Canada', 'Australia', 'Japan', 'Switzerland',
    'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Austria',
    'Israel', 'Korea, South', 'Taiwan', 'Singapore', 'New Zealand',
    'Czech Republic', 'Poland', 'Hungary', 'Portugal', 'Greece', 'Ireland',
]

ALL_COUNTRIES = sorted(list(set(LMIC_COUNTRIES + HIC_COUNTRIES + [
    'Afghanistan', 'Albania', 'Angola', 'Armenia', 'Azerbaijan',
    'Bahrain', 'Belarus', 'Benin', 'Bhutan', 'Bosnia', 'Botswana',
    'Burkina Faso', 'Burundi', 'Cape Verde', 'Central African Republic',
    'Chad', 'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba',
    'Cyprus', 'Djibouti', 'Dominican Republic', 'Equatorial Guinea',
    'Eritrea', 'Estonia', 'Eswatini', 'Fiji', 'Gabon', 'Gambia',
    'Georgia', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
    'Jamaica', 'Kazakhstan', 'Kyrgyzstan', 'Laos', 'Latvia',
    'Lesotho', 'Liberia', 'Lithuania', 'Madagascar', 'Malawi',
    'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius',
    'Moldova', 'Mongolia', 'Montenegro', 'Mozambique', 'Namibia',
    'Niger', 'North Macedonia', 'Oman', 'Papua New Guinea',
    'Slovak Republic', 'Slovenia', 'Somalia', 'Sri Lanka',
    'Suriname', 'Tajikistan', 'Timor-Leste', 'Togo', 'Trinidad',
    'Turkmenistan', 'Ukraine', 'Uruguay', 'Uzbekistan',
    'Venezuela', 'Other / Unknown'
])))

# ────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ────────────────────────────────────────────────────────────

def classify_income(country):
    if country in LMIC_COUNTRIES: return 'LMIC'
    elif country in HIC_COUNTRIES: return 'HIC'
    else: return 'Other'

def classify_region(country):
    mena = ['Tunisia','Morocco','Algeria','Egypt','Jordan','Lebanon',
            'Saudi Arabia','Kuwait','Qatar','Libya','Syria','Iraq',
            'Yemen','Sudan','Oman','Nigeria','Kenya','Ghana','Cameroon',
            'Uganda','South Africa','Ethiopia','Tanzania','Senegal',
            'Ivory Coast','DR Congo','Zimbabwe','Zambia','Rwanda',
            'Mauritius','Namibia','Botswana','Djibouti','Somalia',
            'Eritrea','Mali','Niger','Chad','Burkina Faso','Benin',
            'Togo','Gambia','Guinea','Guinea-Bissau','Sierra Leone',
            'Liberia','Malawi','Mozambique','Madagascar','Angola',
            'Gabon','Congo','Central African Republic','Equatorial Guinea',
            'Cape Verde','Comoros','Lesotho','Eswatini']
    asia = ['India','China','Philippines','Thailand','Malaysia','Vietnam',
            'Pakistan','Indonesia','Japan','Korea, South','Taiwan',
            'Singapore','Australia','New Zealand','Bangladesh','Nepal',
            'Myanmar','Cambodia','Sri Lanka','Laos','Mongolia',
            'Bhutan','Timor-Leste','Maldives','Papua New Guinea','Fiji']
    latam = ['Brazil','Mexico','Argentina','Colombia','Chile','Venezuela',
             'Peru','Ecuador','Bolivia','Paraguay','Uruguay','Guyana',
             'Suriname','Trinidad','Cuba','Dominican Republic','Haiti',
             'Jamaica','Guatemala','Honduras','El Salvador','Nicaragua',
             'Panama','Costa Rica']
    europe = ['Russia','Turkey','Romania','Serbia','Czech Republic',
              'Poland','Hungary','Bulgaria','Ukraine','Latvia',
              'Lithuania','Croatia','Slovakia','Slovenia','Estonia',
              'Belarus','Moldova','Georgia','Armenia','Azerbaijan',
              'Kazakhstan','Kyrgyzstan','Tajikistan','Turkmenistan',
              'Uzbekistan','Bosnia','North Macedonia','Montenegro',
              'Albania','Kosovo']
    if country in mena:   return 'MENA_Africa'
    elif country in asia: return 'Asia_Pacific'
    elif country in latam:return 'Latin_America'
    elif country in europe:return 'Eastern_Europe'
    else:                  return 'Western_HIC'

def predict_resistance(ab, patient_data, income_group, who_region,
                       known_resistances, models_loaded):
    """Prédit la résistance avec le modèle ML ou utilise les résultats connus."""

    known = known_resistances.get(ab)
    if known is not None:
        return float(known), 'known'

    if not models_loaded or ab not in st.session_state.get('models', {}):
        # Mode démo : utilise résistance régionale + ajustements
        region_data = REGIONAL_RESISTANCE.get(who_region, REGIONAL_RESISTANCE['Western_HIC'])
        base_r = region_data.get(ab, 20) / 100
        # Ajustements cliniques
        if patient_data['speciality'] in ['Medicine ICU', 'Surgery ICU', 'Pediatric ICU']:
            base_r = min(base_r * 1.3, 0.95)
        if patient_data['source'] == 'Blood':
            base_r = min(base_r * 1.15, 0.95)
        # Ajustement MDR
        n_resistant = sum(1 for v in known_resistances.values() if v == 1)
        if n_resistant >= 2:
            base_r = min(base_r * 1.25, 0.95)
        return base_r, 'demo'

    # Mode ML réel
    try:
        model    = st.session_state['models'][ab]
        features = st.session_state['features'][ab]
        gender_map  = {'Male':1,'Female':0,'Unknown':2}
        age_map     = {'0 - 17':0,'18 - 30':1,'31 - 60':2,'61+':3,'Unknown':4}
        spec_map    = {'Clinic / Office':0,'Emergency Room':1,
                       'Medicine General':2,'Medicine ICU':3,
                       'None Given':4,'Nursing Home / Rehab':5,
                       'Pediatric General':6,'Pediatric ICU':7,
                       'Surgery General':8,'Surgery ICU':9}
        source_map  = {'Aspirate':0,'Blood':1,'Bodily Fluids':2,
                       'Bronchus':3,'Genitourinary: Other':4,
                       'Nose':5,'Sputum':6,'Urine':7,'Wound':8}
        income_map  = {'HIC':0,'LMIC':1,'Other':2}
        region_map  = {'Asia_Pacific':0,'Eastern_Europe':1,
                       'Latin_America':2,'MENA_Africa':3,'Western_HIC':4}

        row = {}
        for feat in features:
            if feat == 'Year':
                row[feat] = patient_data['year']
            elif feat == 'Gender_enc':
                row[feat] = gender_map.get(patient_data['gender'], 2)
            elif feat == 'Age Group_enc':
                row[feat] = age_map.get(patient_data['age_group'], 4)
            elif feat == 'Speciality_enc':
                row[feat] = spec_map.get(patient_data['speciality'], 4)
            elif feat == 'Source_enc':
                row[feat] = source_map.get(patient_data['source'], 7)
            elif feat == 'Income_Group_enc':
                row[feat] = income_map.get(income_group, 2)
            elif feat == 'WHO_Region_enc':
                row[feat] = region_map.get(who_region, 4)
            elif feat == 'MDR_flag':
                row[feat] = 1 if sum(1 for v in known_resistances.values() if v==1) >= 3 else 0
            elif feat.endswith('_roll3'):
                ab_r = feat.replace('_roll3','')
                row[feat] = 1 - REGIONAL_EFF.get(ab_r, 0.5)
            elif feat.endswith('_binary'):
                other = feat.replace('_binary','')
                row[feat] = known_resistances.get(other, 0) or 0
            elif feat.endswith('_present'):
                row[feat] = 0
            else:
                row[feat] = 0

        X = pd.DataFrame([row])[features].fillna(0)
        p_res = model.predict_proba(X)[0, 1]
        return float(p_res), 'ml'
    except Exception as e:
        return 0.5, 'error'

def compute_cdsm_score(p_resistance, regional_eff, w1=0.5):
    w2 = 1 - w1
    return w1 * (1 - p_resistance) + w2 * regional_eff

def get_resistance_label(p_res, known=False):
    if known:
        return ('✅ SUSCEPTIBLE confirmé', 'susceptible') if p_res == 0 \
               else ('❌ RÉSISTANT confirmé', 'resistant')
    if p_res >= 0.70:   return ('❌ RÉSISTANT probable',  'resistant')
    elif p_res >= 0.40: return ('⚠️  INCERTAIN',           'uncertain')
    else:               return ('✅ SUSCEPTIBLE probable', 'susceptible')

# ────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES (si disponibles)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Charge les modèles sauvegardés depuis le pipeline ML."""
    models   = {}
    features = {}
    loaded   = False
    model_dir = './models'
    if os.path.exists(model_dir):
        for ab in ANTIBIOTICS:
            model_path   = f'{model_dir}/{ab}_xgb.joblib'
            feature_path = f'{model_dir}/{ab}_features.joblib'
            if os.path.exists(model_path) and os.path.exists(feature_path):
                models[ab]   = joblib.load(model_path)
                features[ab] = joblib.load(feature_path)
                loaded = True
    return models, features, loaded

models, features_dict, models_loaded = load_models()
if 'models' not in st.session_state:
    st.session_state['models']   = models
    st.session_state['features'] = features_dict

# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=80)
    st.markdown("## 🏥 CDSM v1.0")
    st.markdown("**Clinical Decision Support**")
    st.markdown("*K. pneumoniae* AMR Prediction")
    st.divider()

    st.markdown("### 📊 Statut des modèles")
    if models_loaded:
        st.success(f"✅ Modèles ML chargés ({len(models)} antibiotiques)")
    else:
        st.warning("⚠️ Mode démonstration\n\nPour le mode ML complet,\nsauvegardez les modèles\ndepuis Colab (voir guide).")

    st.divider()
    st.markdown("### 📈 Performances (ATLAS)")
    perf_data = {
        'Antibiotique': ['Meropenem', 'Imipenem', 'Ceftriaxone', 'Amikacin', 'Ceftazidime'],
        'AUC': ['0.9962', '0.9740', '0.9766', '0.9530', '0.8896']
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    **Dataset :** Pfizer ATLAS (106,946 isolats)
    **Période :** 2004–2024
    **Pays :** 68 pays
    **Modèle :** XGBoost + SHAP
    **Validation :** Temporelle stricte

    📄 *Messaoudi et al., 2026*
    *Vivli AMR Data Challenge*
    """)

# ────────────────────────────────────────────────────────────
# EN-TÊTE PRINCIPAL
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 CDSM — Aide à la Décision Antibiotique</h1>
    <p>Système de recommandation empirique pour infections à <em>Klebsiella pneumoniae</em></p>
    <p style="font-size:0.85rem; opacity:0.7;">
        Basé sur l'analyse de 106,946 isolats — ATLAS Global Surveillance Dataset 2004–2024
    </p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# FORMULAIRE PATIENT
# ────────────────────────────────────────────────────────────
st.markdown("## 📋 Données Patient")
st.markdown("*Remplissez les informations disponibles à l'admission*")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🌍 Localisation")
    country = st.selectbox("Pays", ALL_COUNTRIES,
                           index=ALL_COUNTRIES.index('Tunisia') if 'Tunisia' in ALL_COUNTRIES else 0)
    year = st.number_input("Année", min_value=2020, max_value=2030,
                           value=datetime.now().year)

with col2:
    st.markdown("#### 👤 Patient")
    age_group = st.selectbox("Groupe d'âge",
                             ['0 - 17', '18 - 30', '31 - 60', '61+', 'Unknown'],
                             index=3)
    gender = st.selectbox("Sexe", ['Male', 'Female', 'Unknown'])

with col3:
    st.markdown("#### 🏨 Clinique")
    source = st.selectbox("Source de l'infection",
                          ['Urine', 'Blood', 'Sputum', 'Aspirate',
                           'Wound', 'Bronchus', 'Bodily Fluids'])
    speciality = st.selectbox("Service / Spécialité",
                              ['Emergency Room', 'Medicine ICU', 'Surgery ICU',
                               'Medicine General', 'Pediatric General',
                               'Pediatric ICU', 'Surgery General',
                               'Nursing Home / Rehab', 'Clinic / Office'])

st.divider()

# Résistances connues
st.markdown("## 🧪 Résistances déjà connues (optionnel)")
st.markdown("*Si vous avez déjà des résultats partiels, indiquez-les ici*")

known_resistances = {}
cols = st.columns(5)
ab_list_display = ['Meropenem', 'Imipenem', 'Ceftriaxone', 'Ceftazidime', 'Amikacin']
extra = ['Ciprofloxacin']

for i, ab in enumerate(ab_list_display):
    with cols[i]:
        val = st.radio(ab, ['Inconnu', 'Susceptible', 'Résistant'],
                       key=f'known_{ab}', horizontal=False)
        if val == 'Susceptible':   known_resistances[ab] = 0
        elif val == 'Résistant':   known_resistances[ab] = 1
        else:                      known_resistances[ab] = None

st.markdown("**Autres antibiotiques testés :**")
cols2 = st.columns(4)
extra_abs = ['Ciprofloxacin', 'Gentamicin', 'Piperacillin-Tazobactam', 'Colistin']
for i, ab in enumerate(extra_abs):
    with cols2[i]:
        val = st.radio(ab, ['Inconnu', 'Susceptible', 'Résistant'],
                       key=f'known_{ab}', horizontal=False)
        if val == 'Susceptible':  known_resistances[ab] = 0
        elif val == 'Résistant':  known_resistances[ab] = 1

st.divider()

# Paramètre de pondération
with st.expander("⚙️ Paramètres avancés (optionnel)"):
    w1 = st.slider("Poids w1 (résistance prédite vs efficacité régionale)",
                   min_value=0.3, max_value=0.7, value=0.5, step=0.1)
    st.caption(f"w1={w1:.1f} → poids résistance prédite | w2={1-w1:.1f} → poids efficacité régionale")

# ────────────────────────────────────────────────────────────
# BOUTON ANALYSE
# ────────────────────────────────────────────────────────────
st.markdown("")
analyse_btn = st.button("🔍  ANALYSER ET RECOMMANDER",
                        type="primary", use_container_width=True)

# ────────────────────────────────────────────────────────────
# RÉSULTATS
# ────────────────────────────────────────────────────────────
if analyse_btn:

    # Classification géographique
    income_group = classify_income(country)
    who_region   = classify_region(country)

    patient_data = {
        'country':    country,
        'year':       year,
        'age_group':  age_group,
        'gender':     gender,
        'source':     source,
        'speciality': speciality,
    }

    # Prédictions
    with st.spinner("⏳ Analyse en cours..."):
        predictions = {}
        scores      = {}

        for ab in ANTIBIOTICS:
            p_res, mode = predict_resistance(
                ab, patient_data, income_group, who_region,
                known_resistances, models_loaded
            )
            e_reg = REGIONAL_EFF.get(ab, 0.5)
            score = compute_cdsm_score(p_res, e_reg, w1)
            label, status = get_resistance_label(
                p_res, known=(known_resistances.get(ab) is not None))

            predictions[ab] = {
                'p_resistance': p_res,
                'score':        score,
                'label':        label,
                'status':       status,
                'mode':         mode,
                'e_reg':        e_reg,
            }
            scores[ab] = score

    # Ranking (exclure les résistants connus)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_filtered = [(ab, s) for ab, s in ranked
                       if known_resistances.get(ab) != 1]

    rec1 = ranked_filtered[0][0] if len(ranked_filtered) > 0 else ranked[0][0]
    rec2 = ranked_filtered[1][0] if len(ranked_filtered) > 1 else None

    # ── RÉSULTATS ──
    st.markdown("---")
    st.markdown("## 📊 Résultats de l'Analyse")

    # Info patient
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("🌍 Pays", country)
    with col_b:
        st.metric("💰 Groupe revenus", income_group)
    with col_c:
        st.metric("🗺️ Région OMS", who_region.replace('_', '/'))
    with col_d:
        st.metric("🏥 Source", source)

    st.markdown("")

    # ── RECOMMANDATION ──
    st.markdown(f"""
    <div class="recommendation-box">
        <p style="font-size:0.9rem; opacity:0.8; margin-bottom:0.5rem;">
            🥇 RECOMMANDATION EMPIRIQUE — 1ÈRE LIGNE
        </p>
        <h2>💊 {rec1.upper()}</h2>
        <p>Score CDSM : {scores[rec1]:.3f} |
           Efficacité régionale LMIC : {REGIONAL_EFF.get(rec1,0)*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    if rec2:
        st.markdown(f"""
        <div class="second-line-box">
            <strong>🥈 2ème ligne (si contre-indication) : {rec2}</strong>
            — Score : {scores[rec2]:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── PROBABILITÉS PAR ANTIBIOTIQUE ──
    st.markdown("### 🔬 Probabilités de Résistance par Antibiotique")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Graphique Plotly
        ab_names_plot  = [ab for ab, _ in ranked]
        p_values_plot  = [predictions[ab]['p_resistance'] * 100 for ab, _ in ranked]
        colors_plot    = []
        for ab, _ in ranked:
            p = predictions[ab]['p_resistance']
            if known_resistances.get(ab) == 1:
                colors_plot.append('#9E9E9E')
            elif p >= 0.70:
                colors_plot.append('#E53935')
            elif p >= 0.40:
                colors_plot.append('#FB8C00')
            else:
                colors_plot.append('#43A047')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ab_names_plot[::-1],
            x=p_values_plot[::-1],
            orientation='h',
            marker_color=colors_plot[::-1],
            marker_line_color='white',
            marker_line_width=2,
            text=[f'{p:.1f}%' for p in p_values_plot[::-1]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probabilité résistance : %{x:.1f}%<extra></extra>'
        ))
        fig.add_vline(x=70, line_dash="dash", line_color="red",
                      annotation_text="Seuil résistance", annotation_position="top right")
        fig.add_vline(x=40, line_dash="dash", line_color="orange",
                      annotation_text="Seuil incertitude", annotation_position="top right")
        fig.update_layout(
            title='Probabilité de Résistance par Antibiotique',
            xaxis_title='Probabilité de Résistance (%)',
            xaxis=dict(range=[0, 115]),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Détail par antibiotique :**")
        for ab, _ in ranked:
            pred   = predictions[ab]
            status = pred['status']
            label  = pred['label']
            p      = pred['p_resistance']

            if status == 'resistant':
                css_class = 'result-resistant'
            elif status == 'uncertain':
                css_class = 'result-uncertain'
            else:
                css_class = 'result-susceptible'

            mode_badge = '🔬 ML' if pred['mode'] == 'ml' else \
                        '✓ Connu' if pred['mode'] == 'known' else '📊 Épidémio'

            st.markdown(f"""
            <div class="result-card {css_class}">
                <strong>{ab}</strong>
                <span style="float:right; font-size:0.75rem">{mode_badge}</span>
                <br>{label}
                <br><small>Probabilité : {p*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

    # ── SCORE CDSM ──
    st.markdown("### 💊 Score Décisionnel CDSM")

    fig2 = go.Figure()
    score_colors = []
    for ab, _ in ranked:
        if ab == rec1:               score_colors.append('#1565C0')
        elif ab == rec2:             score_colors.append('#42A5F5')
        elif known_resistances.get(ab) == 1: score_colors.append('#9E9E9E')
        else:                        score_colors.append('#90CAF9')

    fig2.add_trace(go.Bar(
        y=[ab for ab, _ in ranked][::-1],
        x=[scores[ab] for ab, _ in ranked][::-1],
        orientation='h',
        marker_color=score_colors[::-1],
        marker_line_color='white',
        marker_line_width=2,
        text=[f'{scores[ab]:.3f}' for ab, _ in ranked][::-1],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score CDSM : %{x:.3f}<extra></extra>'
    ))
    fig2.update_layout(
        title='Ranking CDSM — Score composite (Efficacité − Risque résistance)',
        xaxis_title='Score Décisionnel CDSM',
        xaxis=dict(range=[0, 1.15]),
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=12)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── CONTEXTE ÉPIDÉMIOLOGIQUE ──
    st.markdown("### 🌍 Contexte Épidémiologique Régional")

    region_data = REGIONAL_RESISTANCE.get(who_region, REGIONAL_RESISTANCE['Western_HIC'])
    lmic_data   = REGIONAL_RESISTANCE['MENA_Africa']

    epi_df = pd.DataFrame({
        'Antibiotique': list(region_data.keys()),
        f'Résistance {who_region.replace("_"," ")} (%)': list(region_data.values()),
        'Résistance LMIC globale (%)': [
            REGIONAL_RESISTANCE.get('MENA_Africa', {}).get(ab, 0)
            if income_group == 'LMIC' else
            REGIONAL_RESISTANCE.get('Western_HIC', {}).get(ab, 0)
            for ab in region_data.keys()
        ]
    })

    fig3 = px.bar(epi_df, x='Antibiotique',
                  y=[f'Résistance {who_region.replace("_"," ")} (%)',
                     'Résistance LMIC globale (%)'],
                  barmode='group',
                  color_discrete_sequence=['#1565C0', '#90CAF9'],
                  title=f'Résistance régionale — {who_region.replace("_", " ")} vs LMIC global')
    fig3.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── AVERTISSEMENTS ──
    st.markdown("### ⚠️ Alertes Cliniques")

    mero_p  = predictions.get('Meropenem', {}).get('p_resistance', 0)
    ceft_p  = predictions.get('Ceftazidime', {}).get('p_resistance', 0)
    cipro_r = known_resistances.get('Ciprofloxacin') == 1

    alerts = []
    if mero_p > 0.5:
        alerts.append(('danger', '🔴 Résistance carbapénème probable détectée',
                        'Consultez un infectiologue. Envisager : Colistine, '
                        'Tigecycline, Ceftazidime-avibactam selon disponibilité locale.'))
    if ceft_p > 0.6:
        alerts.append(('warning', '🟠 Résistance ESBL probable',
                        'Éviter les céphalosporines de 3ème génération. '
                        'Préférer Amikacin ou Carbapénèmes si susceptibles.'))
    if cipro_r:
        alerts.append(('warning', '🟠 Résistance fluoroquinolones confirmée',
                        'Éviter toute la classe des fluoroquinolones '
                        '(Ciprofloxacine, Lévofloxacine, Moxifloxacine).'))
    if income_group == 'LMIC':
        alerts.append(('warning', '⚠️ Contexte LMIC',
                        'La résistance est 2.8-3× plus élevée en LMIC qu\'en pays à hauts revenus. '
                        'Les guidelines EUCAST standard peuvent sous-estimer le risque réel.'))
    if not alerts:
        alerts.append(('success', '✅ Aucune alerte critique détectée',
                        'Le profil de résistance semble gérable avec les antibiotiques recommandés.'))

    for alert_type, title, message in alerts:
        css = 'danger-box' if alert_type == 'danger' else 'warning-box'
        st.markdown(f"""
        <div class="{css}">
            <strong>{title}</strong><br>
            <small>{message}</small>
        </div>
        """, unsafe_allow_html=True)

    # ── RÉSUMÉ EXPORTABLE ──
    st.markdown("### 📄 Résumé de la Consultation")

    summary = f"""
CDSM — RAPPORT DE RECOMMANDATION ANTIBIOTIQUE
{'='*50}
Date          : {datetime.now().strftime('%d/%m/%Y %H:%M')}
Patient       : {age_group} | {gender}
Pays          : {country} ({income_group})
Région OMS    : {who_region}
Source        : {source} | Service : {speciality}

RECOMMANDATION :
  1ère ligne : {rec1}
  2ème ligne : {rec2 if rec2 else 'N/A'}

PROBABILITÉS DE RÉSISTANCE :
"""
    for ab, _ in ranked:
        p = predictions[ab]['p_resistance']
        summary += f"  {ab:<15} : {p*100:.1f}%\n"

    summary += f"""
SCORES CDSM :
  w1={w1:.1f} (résistance prédite) | w2={1-w1:.1f} (efficacité régionale)
"""
    for ab, s in ranked:
        summary += f"  {ab:<15} : {s:.3f}\n"

    summary += f"""
MODÈLE     : XGBoost — ATLAS 2004-2024 (Messaoudi et al., 2026)
AVERTISSEMENT : Outil d'aide à la décision uniquement.
               Le médecin reste seul responsable du choix thérapeutique.
"""
    st.download_button(
        label="📥 Télécharger le rapport",
        data=summary,
        file_name=f"CDSM_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

    # ── DISCLAIMER ──
    st.markdown(f"""
    <div class="disclaimer">
        ⚕️ <strong>Avertissement médical :</strong>
        Ce système est un outil d'aide à la décision clinique basé sur des données
        de surveillance épidémiologique (ATLAS, Pfizer, 106,946 isolats, 2004–2024).
        Il ne remplace pas le jugement clinique du médecin, l'antibiogramme,
        ni les guidelines institutionnelles. Le médecin reste seul responsable
        de la décision thérapeutique. En cas de doute, consultez un infectiologue.
        <br><br>
        Mode actuel : {'✅ Modèles ML chargés' if models_loaded else '⚠️ Mode démonstration (épidémiologique)'}
        | Source : Messaoudi et al., Vivli AMR Data Challenge 2026
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    🏥 CDSM v1.0 — Clinical Decision Support for <em>K. pneumoniae</em> |
    Messaoudi A., Rebhi S., Laajili S. |
    Vivli AMR Data Challenge 2026 |
    Dataset : Pfizer ATLAS 2004–2024 (106,946 isolats, 68 pays)
</div>
""", unsafe_allow_html=True)
