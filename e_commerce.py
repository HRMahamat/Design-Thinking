import os, time, re, implicit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler as SC
from scipy.sparse import coo_matrix
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
import findspark

# ------------------------------------------------------------------
# 0️⃣ – CONFIG & CSS
# ------------------------------------------------------------------
st.set_page_config(page_title="📊 E-Commerce Cameroon IA 📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.sidebar .sidebar-content { background: #001f3f; color: #fff; }
.stButton>button, .stDownloadButton>button { border-radius:8px; font-weight:bold; }
.stButton>button { background: #0074D9; color:#fff; }
.stDownloadButton>button { background: #2ECC40; color:#fff; }
h1, h2, h3 { color: #001f3f; }
.block-container { padding:1rem 2rem; }
.css-1d391kg { background: #fff; border-radius:8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding:1rem; }
</style>
""", unsafe_allow_html=True)

ass = ["Sélectionnez la tranche d'âge", "-18", "18-24", "25-34", "35-44", "45+"]
ms  = ["Sélectionnez le mois", "Aucun",
       "Janvier","Février","Mars","Avril","Mai","Juin",
       "Juillet","Aout","Septembre","Octobre","Novembre","Décembre"]

# ------------------------------------------------------------------
# 1️⃣ – INIT SPARK & CHARGEMENT + NETTOYAGE
# ------------------------------------------------------------------
@st.cache_resource
def init_spark():
    findspark.init()
    return SparkSession.builder.appName("EComCameroonIA").getOrCreate()

@st.cache_resource
def load_and_clean():
    spark = init_spark()
    df = spark.read.option("header", True).option("inferSchema", True).csv("Avis clients.csv", sep=";")
    df = df.drop("Horodateur")

    # safe rename
    for c in df.columns:
        safe = re.sub(r'[^0-9A-Za-z]', '_', c).strip('_')
        df = df.withColumnRenamed(c, safe)

    # ages_bucket
    age_col = next(c for c in df.columns if "Quel__ge" in c)
    df = df.withColumn("ages_bucket",
        when(col(age_col).rlike("18-24"), "18-24")
       .when(col(age_col).rlike("25-34"), "25-34")
       .when(col(age_col).rlike("35-44"), "35-44")
       .when(col(age_col).rlike("45"),    "45+")
       .otherwise("-18")
    )

    # month_pref
    raw = "Pendant_quelles_p_riodes_de_l_ann_e_passez_vous_le_plus_d_achats_en_ligne____indiquez_les_mois_ou_les_saisons_que_vous_privil_giez"
    df = df.withColumn("month_pref",
        when(col(raw).rlike("Janvier"),  "Janvier")
       .when(col(raw).rlike("Février"), "Février")
       .when(col(raw).rlike("Mars"),    "Mars")
       .when(col(raw).rlike("Avril"),   "Avril")
       .when(col(raw).rlike("Mai"),     "Mai")
       .when(col(raw).rlike("Juin"),    "Juin")
       .when(col(raw).rlike("Juillet"), "Juillet")
       .when(col(raw).rlike("Août|Aout"), "Aout")
       .when(col(raw).rlike("Septembre"), "Septembre")
       .when(col(raw).rlike("Octobre"),   "Octobre")
       .when(col(raw).rlike("Novembre"),  "Novembre")
       .when(col(raw).rlike("Décembre"),  "Décembre")
       .otherwise("Autre")
    )

    # Abandon flag
    ab = next(c for c in df.columns if "abandonn" in c.lower())
    df = df.withColumn("Abandon_flag",
        when(col(ab).contains("Non"), 0).otherwise(1)
    )

    # Récupère en pandas **brut**
    pdf_raw = df.toPandas()
    # On garde les colonnes textuelles pour les filtres + visuels
    return pdf_raw

# charge
pdf_raw = load_and_clean()

# --------------------------------------------------------------------------------------------------
# 2️⃣ – SIDEBAR & FILTRES (on travaille toujours sur pdf_raw pour les visuels ET pour la segmentation ML)
# --------------------------------------------------------------------------------------------------
st.sidebar.title("📊 CommerceGenius")
page = st.sidebar.radio("", [
    "Accueil","Analytics Live","Segmentation",
    "Recommandations","Alertes","Visualisations",
    "Export CSV","Commentaires"
])
st.sidebar.markdown("---")

# noms de colonnes
age_col = "ages_bucket"
city_col = [c for c in pdf_raw.columns if "ville_habitez" in c.lower()][0]
month_col = "month_pref"
achat_col = [c for c in pdf_raw.columns if "fr_quence" in c.lower()][0]
mode_col = [c for c in pdf_raw.columns if "paiement" in c.lower()][0]
product_col = [c for c in pdf_raw.columns if "achetez_habituellement" in c.lower()][0]

# copie pour filtrer
sel_raw = pdf_raw.copy()

st.sidebar.subheader("Filtrer par : ")
# filtre tranche d'âge
a = st.sidebar.selectbox("Tranche d'âge", ass)
if a != ass[0]:
    sel_raw = sel_raw[ sel_raw[age_col] == a ]

# filtre ville
villes = ["Sélectionnez la ville"] + sorted(sel_raw[city_col].unique())
v = st.sidebar.selectbox("Ville", villes)
if v!="Sélectionnez la ville":
    sel_raw = sel_raw[ sel_raw[city_col] == v ]

# filtre période
m = st.sidebar.selectbox("Période", ms)
if m not in (ms[0], ms[1]):
    sel_raw = sel_raw[ sel_raw[month_col] == m ]

# --------------------------------------------------------------------------------------------------
# 3️⃣ – ACCUEIL
# --------------------------------------------------------------------------------------------------
if page=="Accueil":
    st.markdown("## 🎯 Solution IA – Comportement Client")
    st.markdown("Tableau de bord E-Commerce Cameroun : en temps réel, segmentation, recommandations.")
    img = Image.open("image.jpg")
    st.image(img.resize((1000, int((float(img.size[1]) * float((700 / float(img.size[0])))))), Image.FILTERED), use_container_width=False)

# --------------------------------------------------------------------------------------------------
# 4️⃣ – ANALYTICS LIVE
# --------------------------------------------------------------------------------------------------
elif page=="Analytics Live":
    st.markdown("## 📈 Analytics en Temps Réel")
    df_counts = pdf_raw[age_col].value_counts().reset_index()
    df_counts.columns = [age_col, "count"]
    fig = px.bar(df_counts,
        x=age_col, y="count", title="Répartition des clients par tranche d'âge",
        labels={age_col: "Tranche d'âge", "count": "Nombre de clients"},
        category_orders={age_col: ["-18", "18-24", "25-34", "35-44", "45+"]})
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------------------
# 5️⃣ – SEGMENTATION DYNAMIQUE
# --------------------------------------------------------------------------------------------------
elif page=="Segmentation":
    st.markdown("## 🔍 Segmentation Dynamique")
    data = pdf_raw[[age_col, month_col]].dropna().copy()

    # 2) On passe ages_bucket et month_pref en indices numériques
    data["age_idx"], age_labels = pd.factorize(data[age_col])
    data["month_idx"], month_labels = pd.factorize(data[month_col])

    # 3) On normalise nos deux axes
    scaler = SC()
    X = scaler.fit_transform(data[["age_idx", "month_idx"]])

    # 4) On calcule la silhouette pour k=2..6
    sil_scores = []
    for k in range(2, 21):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        sil_scores.append((k, sil))

    df_sil = pd.DataFrame(sil_scores, columns=["k", "silhouette"])
    fig_sil = px.line(df_sil, x="k", y="silhouette", markers=True, title="Silhouette Score selon k", labels={"k": "Nombre de clusters", "silhouette": "Silhouette"})
    st.plotly_chart(fig_sil, use_container_width=True)

    # 5) On choisit le meilleur k
    best_k = max(sil_scores, key=lambda x: x[1])[0]
    st.markdown(f"**👉 k optimal retenu : {best_k}**")

    # 6) On refait le clustering
    final_km = KMeans(n_clusters=best_k, random_state=42).fit(X)
    data["cluster"] = final_km.labels_

    # 7) Scatter des clusters (sur les indices)
    fig_clusters = px.scatter(data, x="age_idx", y="month_idx", color="cluster", title="Clusters comportementaux (indices)",
        labels={"age_idx": "Âge (index)", "month_idx": "Mois (index)", "cluster": "Cluster"})
    st.plotly_chart(fig_clusters, use_container_width=True)

    # 8) Centres (dans l’espace standardisé)
    centers = final_km.cluster_centers_
    df_centers = pd.DataFrame(centers, columns=["age_idx", "month_idx"])
    st.markdown("**Centres (standardisés)**")
    st.dataframe(df_centers.style.format("{:.2f}"))

    # 9) Effectifs par cluster
    counts = data["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Effectif"]
    fig_counts = px.bar(counts, x="Cluster", y="Effectif", title="Effectif par cluster", labels={"Effectif": "Clients", "Cluster": "Cluster"})
    st.plotly_chart(fig_counts, use_container_width=True)


# --------------------------------------------------------------------------------------------------
# 6️⃣ – RECOMMANDATIONS PERSO
# --------------------------------------------------------------------------------------------------
elif page=="Recommandations":
    st.markdown("## 🤖 Recommandations Personnalisées")
    df_seg = sel_raw[["Nom_d_utilisateur", product_col]].dropna().copy()
    if sel_raw.empty:
        st.warning("⚠️ Aucun utilisateur ne correspond à vos filtres.")
    else:
        df_seg["rating"] = 1
        df_seg["user_id"], users = pd.factorize(df_seg["Nom_d_utilisateur"])
        df_seg["item_id"], items = pd.factorize(df_seg[product_col])

        M = coo_matrix(
            (df_seg["rating"], (df_seg["user_id"], df_seg["item_id"])),
            shape=(len(users), len(items))
        )

        model_seg = implicit.als.AlternatingLeastSquares(
            factors=20, regularization=0.1, iterations=20, random_state=42
        )
        model_seg.fit(M.T)

        valid_uids = [u for u in df_seg["user_id"].unique()
                      if u < model_seg.user_factors.shape[0]]
        if not valid_uids:
            st.warning("⚠️ Pas assez de données pour profiler le segment.")
        else:
            segment_vec = model_seg.user_factors[valid_uids].mean(axis=0)
            scores = model_seg.item_factors.dot(segment_vec)

            top_n = 5
            top_idx = np.argsort(scores)[::-1][:top_n]
            recs = [(items[i], float(scores[i])) for i in top_idx]

            df_recs = pd.DataFrame(recs, columns=["Produit", "Score"])
            st.markdown(f"### 🎁 Top {top_n} recommandations pour votre segment avec : Rapidité de livraison (sans frais) et Respect de la transparence des produits")
            st.table(df_recs.style.format({"Score": "{:.2f}"}))

# --------------------------------------------------------------------------------------------------
# 7️⃣ – ALERTES AUTOMATIQUES
# --------------------------------------------------------------------------------------------------
elif page=="Alertes":
    st.markdown("## 🚨 Alertes Comportement")
    df_alert = pdf_raw[[age_col, month_col, achat_col, mode_col, "Abandon_flag"]].dropna().copy()
    # encodage simple des variables catégorielles
    df_alert["age_idx"], _ = pd.factorize(df_alert[age_col])
    df_alert["month_idx"], _ = pd.factorize(df_alert[month_col])
    df_alert["freq_idx"], _ = pd.factorize(df_alert[achat_col])
    df_alert["mode_idx"], _ = pd.factorize(df_alert[mode_col])

    # features pour IsolationForest
    features = df_alert[["age_idx", "month_idx", "freq_idx", "mode_idx", "Abandon_flag"]]

    # 2) IsolationForest pour détecter ~5% d’anomalies
    iso = IsolationForest(contamination=0.05, random_state=42)
    df_alert["anomaly"] = iso.fit_predict(features)

    # 3) Calcul du taux et affichage
    total = len(df_alert)
    outliers = df_alert[df_alert["anomaly"] == -1]
    rate = len(outliers) / total * 100
    st.metric("Taux d’anomalies détectées", f"{rate:.1f}%")

    if not outliers.empty:
        st.warning(f"⚠️ Comportements atypiques détectés ({len(outliers)} clients)")
        # Affiche un échantillon des clients problématiques
        st.dataframe(
            outliers[[age_col, month_col, achat_col, mode_col, "Abandon_flag"]]
            .reset_index(drop=True)
            .head(10)
        )
    else: st.success("✅ Aucune anomalie détectée.")

# --------------------------------------------------------------------------------------------------
# 8️⃣ – VISUALISATIONS INTERACTIVES
# --------------------------------------------------------------------------------------------------
elif page=="Visualisations":
    st.markdown("## 📊 Visualisations Interactives")
    if sel_raw.empty: st.warning("⚠️ Aucun client ne correspond à vos filtres.")
    else:
        st.subheader("🕒 Fréquence d'achat en ligne")
        fig1 = px.histogram(sel_raw, x=achat_col,
            category_orders={achat_col: ["Jamais", "Une fois par mois", "Plusieurs fois par mois", "Hebdomadairement", "Quotidiennement"]}, title="Répartition de la fréquence d'achat",
            labels={achat_col: "Fréquence"})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("💳 Modes de paiement favoris")
        df_pay = sel_raw[mode_col].value_counts().reset_index()
        df_pay.columns = ["Mode", "Nombre"]
        fig2 = px.bar(df_pay, x="Mode", y="Nombre", title="Top modes de paiement")
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------------------------------
# 9️⃣ – EXPORT CSV
# --------------------------------------------------------------------------------------------------
elif page=="Export CSV":
    st.markdown("## 📥 Export des données filtrées")
    csv = sel_raw.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger CSV", csv, "Results.csv", "text/csv")

# --------------------------------------------------------------------------------------------------
# 🔟 – COMMENTAIRES
# --------------------------------------------------------------------------------------------------
else:
    st.markdown("## 💬 Vos Commentaires")
    txt = st.text_area("Écrire un commentaire…")
    if st.button("Ajouter"):
        with open("comments.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} — {txt}\n")
        st.success("Commentaire ajouté !")
    if os.path.exists("comments.txt"):
        st.text(open("comments.txt", "r", encoding="utf-8").read())
