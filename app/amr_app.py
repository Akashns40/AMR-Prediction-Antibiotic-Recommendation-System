import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------------------
# LOAD ALL REQUIRED FILES (MODELS, FEATURE SPACE, PCA, COLUMNS)
# ------------------------------------------------------------


@st.cache_resource
def load_artifacts():

    # Load trained models
    gene_model = pickle.load(open("multi_label_gene_model.pkl", "rb"))
    rf_models = pickle.load(open("rf_antibiotic_models.pkl", "rb"))

    # Load X feature space (for reference)
    X = pd.read_csv("X_features.csv", index_col=0)

    # Load PCA scalers + models
    try:
        pca_objects = pickle.load(open("pca_info.pkl", "rb"))
    except:
        pca_objects = None

    # Load gene columns used during training
    with open("gene_matrix_columns.txt") as f:
        gene_cols = [line.strip() for line in f]

    # Load metadata columns (optional)
    try:
        with open("metadata_columns.txt") as f:
            metadata_cols = [line.strip() for line in f]
    except:
        metadata_cols = []

    return gene_model, rf_models, X, pca_objects, gene_cols, metadata_cols


gene_model, rf_models, X, pca_objects, gene_cols, metadata_cols = load_artifacts()


# ------------------------------------------------------------
# ANTIBIOTIC RECOMMENDATION ENGINE
# ------------------------------------------------------------

def recommend_antibiotics(strain_features, rf_models, threshold=0.5):
    results = []

    for antibiotic, model in rf_models.items():
        prob_res = model.predict_proba([strain_features])[0][1]
        prob_sens = 1 - prob_res
        prediction = "Resistant" if prob_res >= threshold else "Sensitive"

        results.append({
            "Antibiotic": antibiotic,
            "Prob Sensitive": round(prob_sens, 3),
            "Prob Resistant": round(prob_res, 3),
            "Prediction": prediction
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Prob Sensitive", ascending=False)
    return df


# ------------------------------------------------------------
# PROCESS NEW UPLOADED GENOME INTO FEATURE VECTOR
# ------------------------------------------------------------

def prepare_new_genome(uploaded_df, gene_cols, metadata_cols, X):

    # Start all genes = 0 (absent)
    full_vec = {g: 0 for g in gene_cols}

    # Fill genes present in uploaded genome
    for g in uploaded_df['gene'].tolist():
        if g in full_vec:
            full_vec[g] = 1

    # Add metadata placeholders
    for m in metadata_cols:
        full_vec[m] = 0

    # Convert into DataFrame
    vec_df = pd.DataFrame([full_vec])

    # For PCA features → fill zero (cannot compute PCA for new genome)
    for col in X.columns:
        if "GENE_PC" in col or "RES_PC" in col:
            vec_df[col] = 0

    # Align to full training feature space
    for col in X.columns:
        if col not in vec_df.columns:
            vec_df[col] = 0

    vec_df = vec_df[X.columns]

    return vec_df


# ------------------------------------------------------------
# STREAMLIT APP INTERFACE
# ------------------------------------------------------------

st.title("🧬 Streptococcus Mitis Group — AMR Prediction & Antibiotic Recommender")
st.markdown(
    "Predict antimicrobial resistance and get ideal antibiotic recommendations.")

mode = st.radio("Choose Input Mode:", [
                "Select Existing Strain", "Upload Genome Gene List"])

# ------------------------------------------------------------
# MODE 1 — SELECT EXISTING STRAIN FROM DATASET
# ------------------------------------------------------------

if mode == "Select Existing Strain":

    strain_list = X.index.tolist()
    strain_id = st.selectbox("Select a Strain:", strain_list)

    if st.button("Predict"):
        strain_features = X.loc[strain_id].values
        recommendations = recommend_antibiotics(strain_features, rf_models)

        st.subheader(f"📌 AMR Recommendation Report for: {strain_id}")
        st.dataframe(recommendations)

        st.markdown("### 🟢 Recommended Antibiotics")
        for _, row in recommendations.iterrows():
            if row["Prediction"] == "Sensitive" and row["Prob Sensitive"] >= 0.7:
                st.success(
                    f"{row['Antibiotic']} — Sensitivity: {row['Prob Sensitive']}")

        st.markdown("### 🔴 Avoid These Antibiotics")
        for _, row in recommendations.iterrows():
            if row["Prediction"] == "Resistant":
                st.error(
                    f"{row['Antibiotic']} — Resistance: {row['Prob Resistant']}")


# ------------------------------------------------------------
# MODE 2 — UPLOAD NEW GENOME GENE LIST
# ------------------------------------------------------------

else:
    uploaded = st.file_uploader(
        "Upload genome gene list (CSV with 'gene' column)", type="csv")

    if uploaded:
        uploaded_df = pd.read_csv(uploaded)
        st.write("Uploaded genes:", uploaded_df.head())

        new_vec = prepare_new_genome(uploaded_df, gene_cols, metadata_cols, X)

        if st.button("Predict for Uploaded Genome"):
            recommendations = recommend_antibiotics(
                new_vec.iloc[0].values, rf_models)

            st.subheader("📌 AMR Recommendation for Uploaded Genome")
            st.dataframe(recommendations)

            st.markdown("### 🟢 Recommended Antibiotics")
            for _, row in recommendations.iterrows():
                if row["Prediction"] == "Sensitive" and row["Prob Sensitive"] >= 0.7:
                    st.success(
                        f"{row['Antibiotic']} — Sensitivity: {row['Prob Sensitive']}")

            st.markdown("### 🔴 Avoid These Antibiotics")
            for _, row in recommendations.iterrows():
                if row["Prediction"] == "Resistant":
                    st.error(
                        f"{row['Antibiotic']} — Resistance: {row['Prob Resistant']}")
