import os
import time
import re
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
import implicit

# ------------------------------------------------------------------
# 0️⃣ – CONFIG & CSS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="📊 E-Commerce Cameroon IA 📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
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
# 1️⃣ – CHARGEMENT & NETTOYAGE (pandas seul)
# ------------------------------------------------------------------
@st.cache_data
def load_and_clean():
    df = pd.read_csv("Avis clients.csv", sep=";")
    df.drop(columns=["Horodateur"], inplace=True)

    # safe rename colonnes
    df.columns = df.columns \
        .str.replace(r'[^0-9A-Za-z]', '_', regex=True) \
        .str.strip('_')

    # ages_bucket
    df["ages_bucket"] = df.filter(regex="Quel__ge").iloc[:,0].map({
        r".*18-24.*": "18-24",
        r".*25-34.*": "25-34",
        r".*35-44.*": "35-44",
        r".*45.*":   "45+",
    }).fillna("-18")

    # month_pref (premier mois trouvé)
    raw = [c for c in df.columns if "privil_giez" in c][0]
    mapping = {m:m for m in ms[2:]}
    df["month_pref"] = df[raw].replace(r".*(Janvier).*", "Janvier", regex=True) \
                              .replace(r".*(F[ée]vrier).*", "Février", regex=True) \
                              .replace(r".*(Mars).*",      "Mars", regex=True) \
                              .replace(r".*(Avril).*",     "Avril", regex=True) \
                              .replace(r".*(Mai).*",       "Mai", regex=True) \
                              .replace(r".*(Juin).*",      "Juin", regex=True) \
                              .replace(r".*(Juillet).*",   "Juillet", regex=True) \
                              .replace(r".*(Ao[uû]t).*",   "Aout", regex=True) \
                              .replace(r".*(Septembre).*", "Septembre", regex=True) \
                              .replace(r".*(Octobre).*",   "Octobre", regex=True) \
                              .replace(r".*(Novembre).*",  "Novembre", regex=True) \
                              .replace(r".*(D[ée]cembre).*","Décembre", regex=True) \
                              .fillna("Autre")

    # Abandon flag
    ab_col = [c for c in df.columns if "abandonn" in c.lower()][0]
    df["Abandon_flag"] = df[ab_col].map(lambda x: 0 if str(x).lower().startswith("non") else 1)
    return df

pdf_raw = load_and_clean()

# --------------------------------------------------------------------------------------------------
# 2️⃣ – SIDEBAR & FILTRES
# --------------------------------------------------------------------------------------------------
st.sidebar.title("📊 CommerceGenius")
page = st.sidebar.radio("", [
    "Accueil","Analytics Live","Segmentation",
    "Recommandations","Alertes","Visualisations",
    "Export CSV","Commentaires"
])
st.sidebar.markdown("---")

age_col     = "ages_bucket"
city_col    = [c for c in pdf_raw.columns if "ville_habitez" in c.lower()][0]
month_col   = "month_pref"
achat_col   = [c for c in pdf_raw.columns if "fr_quence" in c.lower()][0]
mode_col    = [c for c in pdf_raw.columns if "paiement" in c.lower()][0]
product_col = [c for c in pdf_raw.columns if "achetez_habituellement" in c.lower()][0]

sel_raw = pdf_raw.copy()
# filtres
a = st.sidebar.selectbox("Tranche d'âge", ass)
if a!=ass[0]: sel_raw = sel_raw[ sel_raw[age_col]==a ]
v = st.sidebar.selectbox("Ville", ["Toutes"]+sorted(sel_raw[city_col].unique()))
if v!="Toutes": sel_raw = sel_raw[ sel_raw[city_col]==v ]
m = st.sidebar.selectbox("Période", ms)
if m not in (ms[0],ms[1]): sel_raw = sel_raw[ sel_raw[month_col]==m ]

# --------------------------------------------------------------------------------------------------
# 3️⃣ – ACCUEIL
# --------------------------------------------------------------------------------------------------
if page=="Accueil":
    st.markdown("## 🎯 CommerceGenius – Comportement Client")
    st.markdown("Tableau de bord E-Commerce Cameroun : en temps réel, segmentation, recommandations.")
    img = Image.open("image.jpg")
    st.image(img.resize((1000, int((float(img.size[1]) * float((700 / float(img.size[0])))))), Image.FILTERED), use_container_width=False)

# --------------------------------------------------------------------------------------------------
# 4️⃣ – ANALYTICS LIVE
# --------------------------------------------------------------------------------------------------
elif page=="Analytics Live":
    st.markdown("## 📈 Analytics en Temps Réel")
    counts = pdf_raw[age_col].value_counts().reindex(ass[1:]).fillna(0).reset_index()
    counts.columns = [age_col,"count"]
    fig = px.bar(counts, x=age_col, y="count",
                 title="Répartition par tranche d'âge",
                 labels={age_col:"Âge", "count":"Clients"})
    st.plotly_chart(fig, use_container_width=True)
    st.write(df)

# --------------------------------------------------------------------------------------------------
# 5️⃣ – SEGMENTATION DYNAMIQUE
# --------------------------------------------------------------------------------------------------
elif page=="Segmentation":
    st.markdown("## 🔍 Segmentation Dynamique (KMeans)")
    df = sel_raw[[age_col, month_col]].dropna().copy()
    # factorize → indices
    df["age_idx"], _   = pd.factorize(df[age_col])
    df["month_idx"], _ = pd.factorize(df[month_col])
    X = SC().fit_transform(df[["age_idx","month_idx"]])
    sils=[]
    for k in range(2,7):
        km=KMeans(n_clusters=k,random_state=42).fit(X)
        sils.append((k, silhouette_score(X, km.labels_)))
    sil_df=pd.DataFrame(sils,columns=["k","silhouette"])
    st.plotly_chart(px.line(sil_df,x="k",y="silhouette",markers=True),use_container_width=True)
    best_k = sil_df.loc[sil_df.silhouette.idxmax(),"k"]
    st.markdown(f"**👉 k optimal : {int(best_k)}**")
    km = KMeans(n_clusters=int(best_k),random_state=42).fit(X)
    df["cluster"]=km.labels_
    st.plotly_chart(px.scatter(df,x="age_idx",y="month_idx",color="cluster"),use_container_width=True)

# --------------------------------------------------------------------------------------------------
# 6️⃣ – RECOMMANDATIONS PERSO (pandas + implicit)
# --------------------------------------------------------------------------------------------------
elif page=="Recommandations":
    st.markdown("## 🤖 Recommandations Personnalisées")
    df_int = sel_raw[["Nom_d_utilisateur", product_col]].dropna().copy()
    if df_int.empty:
        st.warning("⚠️ Pas de données après filtres.")
    else:
        df_int["rating"]=1
        df_int["user_id"], users = pd.factorize(df_int["Nom_d_utilisateur"])
        df_int["item_id"], items = pd.factorize(df_int[product_col])
        M = coo_matrix((df_int.rating, (df_int.user_id, df_int.item_id)),
                       shape=(len(users),len(items)))
        model = implicit.als.AlternatingLeastSquares(factors=10,regularization=0.1,iterations=15)
        model.fit(M.T)
        # on calcule profil segment : moyenne des facteurs users filtrés
        users_in_seg = df_int.user_id.unique()
        prof = model.user_factors[users_in_seg].mean(axis=0)
        scores = model.item_factors.dot(prof)
        top = np.argsort(scores)[::-1][:5]
        recs = [(items[i],scores[i]) for i in top]
        dfr=pd.DataFrame(recs,columns=["Produit","Score"])
        st.table(dfr.style.format({"Score":"{:.2f}"}))

# --------------------------------------------------------------------------------------------------
# 7️⃣ – ALERTES AUTOMATIQUES
# --------------------------------------------------------------------------------------------------
elif page=="Alertes":
    st.markdown("## 🚨 Alertes Comportement")
    df_a = sel_raw[[age_col,month_col,achat_col,mode_col,"Abandon_flag"]].dropna().copy()
    for c in [age_col,month_col,achat_col,mode_col]:
        df_a[f"{c}_idx"],_ = pd.factorize(df_a[c])
    feats = df_a[[f"{c}_idx" for c in [age_col,month_col,achat_col,mode_col]]+["Abandon_flag"]]
    iso=IsolationForest(contamination=0.05,random_state=42)
    df_a["anomaly"]=iso.fit_predict(feats)
    out=df_a[df_a.anomaly==-1]
    rate=len(out)/len(df_a)*100
    st.metric("Taux d’anomalies",f"{rate:.1f}%")
    if not out.empty:
        st.dataframe(out.head(10))
    else:
        st.success("✅ Aucune anomalie")

# --------------------------------------------------------------------------------------------------
# 8️⃣ – VISUALISATIONS INTERACTIVES
# --------------------------------------------------------------------------------------------------
elif page=="Visualisations":
    st.markdown("## 📊 Visualisations")
    if sel_raw.empty:
        st.warning("⚠️ Pas de données.")
    else:
        st.plotly_chart(px.histogram(sel_raw,x=achat_col,
            category_orders={achat_col:["Jamais","Une fois par mois","Plusieurs fois par mois"]}),use_container_width=True)
        st.plotly_chart(px.bar(sel_raw[mode_col].value_counts().reset_index().rename(columns={"index":"Mode",mode_col:"Nombre"}),
            x="Mode",y="Nombre"),use_container_width=True)

# --------------------------------------------------------------------------------------------------
# 9️⃣ – EXPORT CSV
# --------------------------------------------------------------------------------------------------
elif page=="Export CSV":
    st.download_button("⬇️ Télécharger CSV", sel_raw.to_csv(index=False), "export.csv","text/csv")

# --------------------------------------------------------------------------------------------------
# 🔟 – COMMENTAIRES
# --------------------------------------------------------------------------------------------------
else:
    txt=st.text_area("Commentaires…")
    if st.button("Ajouter"):
        with open("comments.txt","a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} — {txt}\n")
        st.success("👍")  
