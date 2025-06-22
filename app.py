import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
from datetime import datetime
import folium
from streamlit_folium import folium_static
import time
import pickle
import os
from groq import Groq
from dotenv import load_dotenv

# Configuration de la page 
st.set_page_config(
    page_title="BerlEstate | Intelligence Immobilière",
    page_icon="🏙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.BerlEstate.pro/support',
        'Report a bug': "https://www.BerlEstate.pro/bug",
        'About': "### Solution IA d'évaluation immobilière\nVersion 2.2\n© 2024 RealEstatePredict Pro"
    }
)

# Style CSS premium amélioré avec plus de couleurs et un fond
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #2E86C1; /* Bleu Ciel Profond */
            --secondary-color: #EBF5FB; /* Bleu très clair, presque blanc */
            --accent-color: #AF7AC5; /* Violet doux */
            --text-color: #333333;
            --header-color: #1A5276; /* Bleu plus foncé pour les titres */
            --card-bg: #FFFFFF; /* Fond des cartes/panneaux */
            --button-hover: #1F618D; /* Bleu plus foncé pour le hover du bouton */
            --tab-active-bg: var(--primary-color);
            --tab-inactive-bg: #D6EAF8; /* Bleu clair pour les onglets inactifs */
            --border-color: #BDC3C7; /* Gris clair pour les bordures */
            --success-color: #28A745; /* Vert pour les succès */
            --info-color: #17A2B8; /* Cyan pour les infos */
            --warning-color: #FFC107; /* Jaune pour les avertissements */
            --danger-color: #DC3545; /* Rouge pour les erreurs */
        }

        /* Background global avec une image */
        .main {
            background: linear-gradient(to bottom right, #EBF5FB, #D6EAF8); /* Un dégradé simple */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed; /* Fixe l'image de fond lors du défilement */
            background-position: center center;
            color: var(--text-color); /* Couleur du texte par défaut */
            padding: 2rem; /* Espace autour du contenu */
            box-sizing: border-box; /* Inclure le padding dans la largeur */
        }

        /* Pour s'assurer que le contenu principal est lisible sur l'image de fond */
        .stApp {
            background-color: rgba(255, 255, 255, 0.9); /* Fond semi-transparent pour le contenu */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
            margin: 20px auto; /* Centrer le contenu */
            max-width: 1400px; /* Largeur maximale du contenu */
        }

        /* Sidebar */
        .stSidebar {
            background-color: var(--card-bg) !important; /* Couleur de fond de la sidebar */
            border-right: 1px solid var(--border-color);
            box-shadow: 2px 0px 10px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            color: var(--text-color);
        }

        .stSidebar .stImage {
            margin-bottom: 2rem;
            border-radius: 8px; /* Coins arrondis pour l'image */
            overflow: hidden;
        }

        /* Boutons */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 8px; /* Plus arrondi */
            padding: 0.7rem 1.5rem; /* Plus grand */
            border: none;
            font-weight: bold;
            transition: background-color 0.3s ease; /* Transition douce */
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); /* Ombre pour le relief */
        }

        .stButton>button:hover {
            background-color: var(--button-hover);
            transform: translateY(-2px); /* Léger effet 3D au survol */
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }

        /* Inputs (Text, Number, Selectbox) */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div[data-baseweb="select"] {
            background-color: var(--secondary-color) !important; /* Un fond plus clair */
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            color: var(--text-color) !important;
        }

        /* Titres */
        h1, h2, h3, h4, h5, h6 {
            color: var(--header-color);
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            font-weight: 700; /* Plus gras */
        }
        
        .stMarkdown h3 {
            color: var(--header-color);
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            font-size: 1.3rem;
            font-weight: 600;
        }

        /* Onglets */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.7rem; /* Plus d'espace entre les onglets */
            background-color: var(--secondary-color); /* Fond de la barre d'onglets */
            border-radius: 10px;
            padding: 0.5rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: var(--tab-inactive-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important; /* Coins plus arrondis */
            padding: 0.8rem 2rem !important; /* Plus de padding */
            color: var(--header-color) !important;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--tab-active-bg) !important;
            color: white !important;
            border-color: var(--tab-active-bg) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2); /* Ombre pour l'onglet actif */
        }

        /* Infos, Success, Warning, Error boxes */
        .stAlert {
            border-radius: 8px;
            padding: 1rem 1.5rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        .stAlert.info { background-color: rgba(23,162,184,0.1); color: var(--info-color); border-left: 5px solid var(--info-color); }
        .stAlert.success { background-color: rgba(40,167,69,0.1); color: var(--success-color); border-left: 5px solid var(--success-color); }
        .stAlert.warning { background-color: rgba(255,193,7,0.1); color: var(--warning-color); border-left: 5px solid var(--warning-color); }
        .stAlert.error { background-color: rgba(220,53,69,0.1); color: var(--danger-color); border-left: 5px solid var(--danger-color); }

        /* Expander */
        .streamlit-expander {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background-color: var(--card-bg);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .streamlit-expanderContent {
            padding: 1rem;
            color: var(--text-color);
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        [data-testid="stMetricLabel"] {
            color: var(--header-color);
            font-size: 1.1em;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--primary-color);
            font-size: 1.8em;
            font-weight: 700;
        }
        [data-testid="stMetricDelta"] {
            color: var(--accent-color);
            font-size: 0.9em;
        }

        /* Markdown text consistency */
        .stMarkdown p, .stMarkdown ul, .stMarkdown li {
            font-size: 1rem;
            line-height: 1.6;
            color: var(--text-color);
            margin-bottom: 0.5rem;
        }

        /* Headers with dividers */
        .stHeader h1, .stHeader h2 {
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }

        /* Styling for the legend box in Geospatial tab */
        .legend-box {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        .legend-color-circle {
            width: 25px;
            height: 25px;
            border-radius: 50%;
            margin-right: 12px;
            border: 1px solid #ddd;
            flex-shrink: 0; /* Prevent shrinking */
        }
        .legend-text {
            font-size: 1.05rem;
            color: var(--text-color);
            line-height: 1.4;
        }
                
        /* Styles spécifiques pour le chat_input de Streamlit dans le pop-up */
        /* Ces styles seront appliqués partout où st.chat_input est utilisé */
        .stChatInput input {
            border-radius: 20px !important; /* Rendre le champ de saisie rond */
            padding: 8px 15px !important;
            border: 1px solid var(--primary-color) !important;
            background-color: white !important;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }
        .stChatInput div[data-baseweb="button"] { /* Bouton d'envoi du chat_input */
            background-color: var(--primary-color) !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0 !important;
            transition: background-color 0.3s ease;
        }
        .stChatInput div[data-baseweb="button"]:hover {
            background-color: var(--button-hover) !important;
        }
        .stChatInput div[data-baseweb="button"] svg { /* Icône du bouton d'envoi */
            color: white !important;
        }

        /* Styles pour les messages de chat (st.chat_message) */
        .stChatMessage {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .stChatMessage[data-element-type="stChatMessage"].user {
            background-color: var(--primary-color);
            color: white;
            align-self: flex-end; /* Aligner à droite pour l'utilisateur */
            border-bottom-right-radius: 4px; /* Petite entaille pour le message utilisateur */
        }
        .stChatMessage[data-element-type="stChatMessage"].assistant {
            background-color: var(--secondary-color);
            color: var(--text-color);
            align-self: flex-start; /* Aligner à gauche pour l'assistant */
            border-bottom-left-radius: 4px; /* Petite entaille pour le message assistant */
        }
        .stChatMessage .st-emotion-cache-p5m8rg { /* Sélectionne l'avatar */
            display: none !important; /* Cache les avatars de chat_message */
        }
        .stChatMessage .st-emotion-cache-1wmy9hp { /* Sélectionne le contenu du message */
            margin-left: 0 !important; /* Supprime la marge de l'avatar caché */
        }

        /* Pour cacher le label "Envoyer un message" de st.chat_input */
        .stChatInput > label {
            display: none !important;
        }

        /* Cache les avatars des messages Streamlit */
        .stChatMessage [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }

        /* Ajuste la marge du contenu du message quand l'avatar est caché */
        .stChatMessage [data-testid="stChatMessageContent"] {
            margin-left: 0 !important;
        }

        /* Styles pour le bouton flottant du chatbot */
        .chatbot-button-container {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
        }
        .chatbot-button-container .stButton > button {
            background-color: var(--accent-color);
            border-radius: 50% !important;
            width: 60px;
            height: 60px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            border: 2px solid white;
        }
        .chatbot-button-container .stButton > button:hover {
            background-color: #9B6BB0; /* Lighter accent color on hover */
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }

        /* Styles pour le pop-up du chatbot */
        .chatbot-popup {
            position: fixed;
            bottom: 100px; /* Adjust based on button position */
            right: 30px;
            width: 350px;
            height: 500px;
            background-color: var(--card-bg);
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        .chatbot-header {
            background-color: var(--primary-color);
            color: white;
            padding: 15px;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.1rem;
            font-weight: bold;
        }
        .chatbot-close-button {
            background: none;
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
        }
        .chatbot-messages-container {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f9f9f9; /* Light background for messages */
        }
        .chatbot-input-container {
            padding: 15px;
            border-top: 1px solid var(--border-color);
            background-color: var(--card-bg);
        }

    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialisation Groq Client
load_dotenv() # Charge les variables d'environnement du fichier .env

# client_groq = Groq(api_key=st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY"))
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY")) 

# Chargement des données optimisé avec stabilité
@st.cache_data(ttl=3600, show_spinner="Chargement des données...")
def load_data(full=False):
    data = pd.read_csv('kc_house_data.csv')

    # Prétraitement stable
    data['date'] = pd.to_datetime(data['date'])
    data['year_sold'] = data['date'].dt.year
    data['month_sold'] = data['date'].dt.month
    data['age'] = data['year_sold'] - data['yr_built']
    data['renovated'] = np.where(data['yr_renovated'] > 0, 1, 0)

    data.fillna({
        'yr_renovated': 0,
        'sqft_basement': 0,
        'view': 0,
        'waterfront': 0
    }, inplace=True)

    data['price_per_sqft'] = data['price'] / data['sqft_living']
    # Handle potential division by zero if sqft_lot is 0, though unlikely
    data['living_ratio'] = data['sqft_living'] / data['sqft_lot']
    data.loc[data['sqft_lot'] == 0, 'living_ratio'] = 0 # Set to 0 if sqft_lot is 0

    # Ajout des features nécessaires pour le modèle pickle
    data['is_renovated'] = (data['yr_renovated'] > 0).astype(int)
    data['years_since_renovation'] = np.where(
        data['yr_renovated'] > 0,
        2024 - data['yr_renovated'],
        2024 - data['yr_built']
    )
    data['sqft_per_bedroom'] = data['sqft_living'] / (data['bedrooms'] + 1)

    if full:
        return data
    else:
        sample_size = min(5000, len(data))
        return data.sample(n=sample_size, random_state=42)

# Initialisation session state avec vérification
if 'data' not in st.session_state:
    st.session_state.data = load_data()
    st.session_state.full_data = load_data(full=True)
    st.session_state.model_trained = False
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.model = None # Ensure model is initialized
    st.session_state.features = None # Ensure features are initialized
    if 'full_data_toggle' not in st.session_state:
        st.session_state.full_data_toggle = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False 
    
    # --- Initialisation spécifique pour le chatbot ---
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False # État pour contrôler l'ouverture du chatbot
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = [{"role": "assistant", "content": "Bonjour ! Je suis votre assistant immobilier. Posez-moi une question !"}]

data = st.session_state.data
full_data = st.session_state.full_data
mean_price_full_data = full_data['price'].mean() # Calculate mean price from full data for reference

# Ajoutez ce dictionnaire au début de votre fichier (ou chargez-le depuis un CSV si vous en avez un plus complet)
ZIPCODE_TO_CITY = {
    98001: "Auburn",
    98002: "Auburn",
    98003: "Federal Way",
    98004: "Bellevue",
    98005: "Bellevue",
    98006: "Bellevue",
    98007: "Bellevue",
    98008: "Bellevue",
    98010: "Black Diamond",
    98011: "Bothell",
    98014: "Carnation",
    98019: "Duvall",
    98022: "Enumclaw",
    98023: "Federal Way",
    98024: "Fall City",
    98027: "Issaquah",
    98028: "Kenmore",
    98029: "Issaquah",
    98030: "Kent",
    98031: "Kent",
    98032: "Kent",
    98033: "Kirkland",
    98034: "Kirkland",
    98038: "Maple Valley",
    98039: "Medina",
    98040: "Mercer Island",
    98042: "Kent",
    98045: "North Bend",
    98052: "Redmond",
    98053: "Redmond",
    98055: "Renton",
    98056: "Renton",
    98058: "Renton",
    98059: "Renton",
    98065: "Snoqualmie",
    98070: "Vashon",
    98072: "Woodinville",
    98074: "Sammamish",
    98075: "Sammamish",
    98077: "Woodinville",
    98092: "Auburn",
    98102: "Seattle",
    98103: "Seattle",
    98105: "Seattle",
    98106: "Seattle",
    98107: "Seattle",
    98108: "Seattle",
    98109: "Seattle",
    98112: "Seattle",
    98115: "Seattle",
    98116: "Seattle",
    98117: "Seattle",
    98118: "Seattle",
    98119: "Seattle",
    98122: "Seattle",
    98125: "Seattle",
    98126: "Seattle",
    98133: "Seattle",
    98136: "Seattle",
    98144: "Seattle",
    98146: "Seattle",
    98148: "Burien",
    98155: "Shoreline",
    98166: "Burien",
    98168: "Tukwila",
    98177: "Shoreline",
    98178: "Tukwila",
    98188: "Tukwila",
    98198: "Des Moines",
    98199: "Seattle"
    # Ajoutez d'autres zipcodes si besoin
}

# Sidebar stable
with st.sidebar:
    st.image("C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\data_scrap\\kc_house\\OIP (1).jpeg",
             use_container_width=True)

    st.markdown("""
    <div style='margin-bottom: 1.5rem;'>
        <h3 style='color: var(--header-color); margin-bottom: 0.5rem;'>Configuration</h3>
        <p style='font-size: 0.9rem; color: var(--text-color);'>Paramètres du modèle</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Mode d'analyse : Standard")

    if st.checkbox("Charger le jeu complet (20k+ propriétés)",
                   st.session_state.full_data_toggle, key="full_data_toggle"):
        data = full_data
        st.session_state.data = data
    else:
        data = st.session_state.full_data.sample(n=min(5000, len(st.session_state.full_data)), random_state=42)
        st.session_state.data = data

# Onglets avec état stable
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard",
    "📈 Analyse",
    "🤖 Modélisation",
    "🌍 Géospatial",
    "⚙ Administration",
    "💬 Chatbot"
])

with tab1:
    st.header("Tableau de Bord Immobilier", divider="blue")

    # KPI Cards stables
    cols = st.columns(4)
    metrics = [
        ("Propriétés analysées", f"{len(data):,}", ""),
        ("Prix moyen", f"${data['price'].mean():,.0f}", ""),
        ("Surface moyenne", f"{data['sqft_living'].mean():,.0f} sqft", ""),
        ("Prix au sqft", f"${data['price_per_sqft'].mean():.1f}", "")
    ]

    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)

    # Visualisations avec clés uniques
    fig1 = px.scatter(
        data,
        x='sqft_living',
        y='price',
        color='bedrooms',
        size='sqft_lot',
        hover_data=['zipcode', 'grade'],
        title="Relation Surface/Prix"
    )
    st.plotly_chart(fig1, use_container_width=True, key="scatter_chart")

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.box(data, y='price', title="Distribution des Prix")
        st.plotly_chart(fig2, use_container_width=True, key="box_chart")
    with col2:
        fig3 = px.histogram(data, x='price_per_sqft', nbins=50, title="Prix au sqft")
        st.plotly_chart(fig3, use_container_width=True, key="hist_chart")

with tab2:
    st.header("Analyse Avancée", divider="blue")

    analysis_cols = st.multiselect(
        "Variables à analyser",
        options=['price', 'sqft_living', 'grade', 'bathrooms', 'view', 'condition', 'age', 'yr_built', 'sqft_lot', 'bedrooms', 'floors'],
        default=['price', 'sqft_living', 'grade'],
        key="analysis_cols"
    )

    if len(analysis_cols) >= 2:
        # Matrice de corrélation avec clé unique
        st.subheader("Matrice de Corrélation")
        corr = data[analysis_cols].corr()
        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='Blues',
            title="Corrélation entre Variables"
        )
        st.plotly_chart(fig4, use_container_width=True, key="corr_matrix")

        # Pairplot avec clé unique
        st.subheader("Relations Multivariées")
        fig5 = px.scatter_matrix(
            data,
            dimensions=analysis_cols,
            color='price',
            height=800
        )
        st.plotly_chart(fig5, use_container_width=True, key="pairplot")

    #  Visualisation des Outliers
    st.subheader("Visualisation des Outliers")
    outlier_col = st.selectbox(
        "Sélectionnez une variable numérique pour visualiser les outliers",
        options=data.select_dtypes(include=np.number).columns.tolist(),
        index=data.select_dtypes(include=np.number).columns.tolist().index('price') if 'price' in data.select_dtypes(include=np.number).columns else 0,
        key="outlier_select_col"
    )
    if outlier_col:
        fig_outliers = px.box(data, y=outlier_col, title=f"Distribution et Outliers de '{outlier_col}'")
        st.plotly_chart(fig_outliers, use_container_width=True, key="outlier_box_plot")


with tab3:
    st.header("Modélisation Prédictive", divider="blue")

    # Chargement du modèle pickle déjà entraîné
    import pickle
    MODEL_PATH = "C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\data_scrap\\kc_house\\rf_king_county_model.pkl"
    try:
        with open(MODEL_PATH, "rb") as f:
            model_package = pickle.load(f)
        model = model_package['model']
        features = model_package['feature_names']
        metrics = model_package['performance_metrics']
        feature_importance = model_package['feature_importance']
        st.success("Modèle pré-entraîné chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        model = None
        features = []
        metrics = {}
        feature_importance = []

    # Affichage des métriques du modèle
    if model is not None:
       # st.subheader("Performance du Modèle")
       # cols = st.columns(3)
       # cols[0].metric("RMSE", f"${metrics['RMSE']:,.0f}")
       # cols[1].metric("R² Score", f"{metrics['R2']:.3f}")
       # cols[2].metric("MAPE", f"{metrics['MAPE']:.1f}%")

        # Section de prédiction en temps réel
        st.subheader("Prédiction du Prix d'une Propriété")
        st.markdown("Entrez les caractéristiques d'une propriété pour obtenir une estimation de son prix.")

        prediction_inputs = {}
        input_cols = st.columns(2)
        col_idx = 0

        for feature in features:
            if feature not in data.columns:
                continue
            current_data_type = data[feature].dtype
            with input_cols[col_idx % 2]:
                min_val = data[feature].min()
                max_val = data[feature].max()
                mean_val = data[feature].mean()
                if current_data_type in ['int64', 'float64']:
                    prediction_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()} :",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val),
                        step=1.0,
                        key=f"pred_input_{feature}"
                    )
                else:
                    prediction_inputs[feature] = st.selectbox(
                        f"{feature.replace('_', ' ').title()} :",
                        options=data[feature].unique().tolist(),
                        key=f"pred_input_{feature}"
                    )
            col_idx += 1

        if st.button("Estimer le Prix", key="predict_button"):
            input_df = pd.DataFrame([prediction_inputs])
            try:
                predicted_price = model.predict(input_df)[0]
                # Trouver la propriété réelle la plus proche du prix estimé
                idx_closest = (full_data['price'] - predicted_price).abs().idxmin()
                closest_row = full_data.loc[idx_closest]
                closest_zip = int(closest_row['zipcode'])
                closest_city = ZIPCODE_TO_CITY.get(closest_zip, "Ville inconnue")
                closest_real_price = closest_row['price']

                st.success(f"Le prix estimé pour cette propriété est de : *${predicted_price:,.2f}*")
                st.info(
                    f"Ville la plus proche pour ce prix : **{closest_city}** (Code postal : {closest_zip})\n\n"
                    f"Prix réel le plus proche dans la base : **${closest_real_price:,.0f}**"
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

        # Affichage de l'importance des variables (à déplacer ici)
        st.subheader("Importance des Variables")
        importance_df = pd.DataFrame(feature_importance)
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 des Variables Importantes'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Aucun modèle pré-entraîné disponible.")
        

with tab4:
    st.header("Analyse Géospatiale", divider="blue")

    st.subheader("Carte des Prix Immobiliers")

    # Selectbox specifically for filtering the map
    selected_zip_for_map = st.selectbox(
        "Filtrer la carte par code postal (facultatif)",
        options=['Tous'] + sorted(data['zipcode'].unique().tolist()), # Add 'Tous' option
        key="zipcode_select_map"
    )

    try:
        temp_data_for_percentile = full_data[['price']].copy()
        sorted_prices = np.sort(temp_data_for_percentile['price'])

        # Filter data for map based on selected_zip_for_map
        if selected_zip_for_map == 'Tous':
            
            current_map_data = data.copy()
        else:
            # Filter the main 'data' DataFrame by the selected zipcode
            current_map_data = data[data['zipcode'] == selected_zip_for_map].copy()

        # Ensure there's data to plot before proceeding
        if not current_map_data.empty:
            price_percentiles = np.searchsorted(sorted_prices, current_map_data['price']) / len(sorted_prices) * 100
            current_map_data['percentile'] = price_percentiles

            # Sample for performance, even after filtering by zipcode
            map_plot_data = current_map_data.sample(n=min(1000, len(current_map_data)), random_state=42)

            m = folium.Map(
                # Center map on the selected area, or overall mean if 'Tous'
                location=[map_plot_data['lat'].mean(), map_plot_data['long'].mean()],
                zoom_start=12 if selected_zip_for_map != 'Tous' else 10, # Zoom in more for a specific zip code
                tiles='cartodbpositron'
            )

            for idx, row in map_plot_data.iterrows():
                percentile = row['percentile']

                if percentile > 90:
                    color = 'red'
                elif percentile > 75:
                    color = 'orange'
                elif percentile > 50:
                    color = 'green'
                else:
                    color = 'blue'

                popup = f"""
                <b>Prix:</b> ${row['price']:,.0f}<br>
                <b>Percentile:</b> {percentile:.1f}%<br>
                <b>Surface:</b> {row['sqft_living']} sqft<br>
                <b>Chambres:</b> {row['bedrooms']}
                """

                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=5,
                    popup=popup,
                    color=color,
                    fill=True,
                    fill_color=color
                ).add_to(m)

            folium_static(m, width=1200, height=600)
        else:
            st.info(f"Aucune propriété trouvée pour le code postal {selected_zip_for_map}.")

        # Utilisation de la nouvelle classe CSS pour la légende
        st.markdown("""
        <div class="legend-box">
            <h4 style="color: var(--header-color); margin-top: 0;">Légende des couleurs :</h4>
            <div class="legend-item">
                <div class="legend-color-circle" style="background-color: red;"></div>
                <span class="legend-text">Top 10% des prix (percentile > 90%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color-circle" style="background-color: orange;"></div>
                <span class="legend-text">Top 25% (percentile 75-90%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color-circle" style="background-color: green;"></div>
                <span class="legend-text">Supérieur à la médiane (50-75%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color-circle" style="background-color: blue;"></div>
                <span class="legend-text">Inférieur à la médiane (<50%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors de la création de la carte : {str(e)}")

    st.subheader("Analyse par Zone Géographique")
    # This selectbox is for the aggregated statistics and histogram, not the map points
    selected_zip = st.selectbox(
        "Sélectionnez un code postal pour l'analyse des statistiques",
        options=sorted(data['zipcode'].unique()),
        key="zipcode_select" 
    )

    if selected_zip:
        zip_data = data[data['zipcode'] == selected_zip]

        if not zip_data.empty:
            cols = st.columns(2)
            metrics = [
                ("Prix moyen", f"${zip_data['price'].mean():,.0f}", ""),
                ("Surface moyenne", f"{zip_data['sqft_living'].mean():,.0f} sqft", ""),
                ("Prix médian", f"${zip_data['price'].median():,.0f}", ""),
                ("Propriétés analysées", len(zip_data), "")
            ]
            
            for col, (label, value, delta) in zip(cols, metrics):
                col.metric(label, value, delta)
            
            fig7 = px.histogram(
                zip_data,
                x='price',
                nbins=30,
                title=f"Distribution des Prix (CP: {selected_zip})"
            )
            st.plotly_chart(fig7, use_container_width=True, key="zip_hist")

with tab5:
    st.header("Administration", divider="blue")
    
    st.subheader("Gestion des Données")
    
    with st.expander("Importer des Données", expanded=False):
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type="csv",
            help="Format attendu : même structure que kc_house_data.csv",
            key="data_uploader"
        )
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                new_data['date'] = pd.to_datetime(new_data['date'])
                new_data['year_sold'] = new_data['date'].dt.year
                new_data['month_sold'] = new_data['date'].dt.month
                new_data['age'] = new_data['year_sold'] - new_data['yr_built']
                new_data['renovated'] = np.where(new_data['yr_renovated'] > 0, 1, 0)
                new_data.fillna({
                    'yr_renovated': 0,
                    'sqft_basement': 0,
                    'view': 0,
                    'waterfront': 0
                }, inplace=True)
                new_data['price_per_sqft'] = new_data['price'] / new_data['sqft_living']
                # Adding the living_ratio for imported data as well
                new_data['living_ratio'] = new_data['sqft_living'] / new_data['sqft_lot']
                new_data.loc[new_data['sqft_lot'] == 0, 'living_ratio'] = 0 
                
                # Update session state with new data
                st.session_state.data = new_data.sample(n=min(5000, len(new_data)), random_state=42)
                st.session_state.full_data = new_data
                st.session_state.model_trained = False # Reset model status as data changed
                st.success(f"Nouveau jeu de données chargé avec {len(new_data):,} propriétés.")
                st.experimental_rerun() # Rerun to apply changes
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {str(e)}")

    # --- Exportation et téléchargement du fichier de données ---
    st.subheader("Exportation des Données")
    export_data = st.session_state.full_data.copy()
    csv = export_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger le jeu de données complet (CSV)",
        data=csv,
        file_name="donnees_immobilieres.csv",
        mime="text/csv",
        key="download_full_data"
    )

    st.subheader("Rapport sur les Valeurs Manquantes")
    missing_data = data.isnull().sum()
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    missing_info = pd.DataFrame({
        'Valeurs Manquantes': missing_data,
        '% Manquant': missing_percentage
    })
    missing_info = missing_info[missing_info['Valeurs Manquantes'] > 0].sort_values(by='% Manquant', ascending=False)
    
    if not missing_info.empty:
        st.write("Voici un aperçu des colonnes avec des valeurs manquantes après le prétraitement initial :")
        st.dataframe(missing_info)
        st.info("Note : Les valeurs manquantes dans 'yr_renovated', 'sqft_basement', 'view', et 'waterfront' sont remplies par 0 lors du chargement des données.")
    else:
        st.success("Aucune valeur manquante détectée dans les données après le prétraitement initial.")

    if st.session_state.get('model_trained', False):
        with st.expander("Exporter les Résultats", expanded=False):
            if st.button("Exporter les Prédictions", key="export_button"):
                try:
                    model = st.session_state.model
                    features = st.session_state.features
                    predictions = model.predict(data[features])
                    export_data = data.copy()
                    export_data['predicted_price'] = predictions
                    
                    csv = export_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger les prédictions en CSV",
                        data=csv,
                        file_name="predictions_immobilieres.csv",
                        mime="text/csv",
                        key="download_predictions"
                    )
                    st.success("Prédictions exportées avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'export des prédictions : {str(e)}")


with tab6:
    st.header("Chatbot IA", divider="blue")
    st.markdown("""
        Discutez avec notre assistant IA pour obtenir des informations sur l'immobilier.
    """)

    # Display chat messages from history on app rerun
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Posez votre question..."):
        # Add user message to chat history
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                chat_completion = client_groq.chat.completions.create(
                    messages=[
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
                        for m in st.session_state.chatbot_messages
                    ],
                    model="llama3-8b-8192", # You can choose other models like 'mixtral-8x7b-32768' or 'gemma-7b-it'
                    temperature=0.7, # Adjust creativity (0.0 to 1.0)
                    max_tokens=500, # Max tokens in the response
                    stream=True
                )

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Désolé, une erreur est survenue lors de la communication avec l'IA: {e}"
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})

st.markdown("""
<div class="footer">
    <p>&copy; 2025 BERL ESTATE. Tous droits réservés.
    <a href="https://www.BERLESTATE.pro/politique-confidentialite" target="_blank">Politique de confidentialité</a> | 
    <a href="https://www.realestatepredict.pro/conditions-utilisation" target="_blank">Conditions d'utilisation</a></p>
</div>
""", unsafe_allow_html=True)

