
import base64
from collections import Counter
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"

CLASS_INFO = {
    "crazing": ("Réseau de fines fissures", "#8B5CF6"),
    "inclusion": ("Inclusions dans le métal", "#EC4899"),
    "patches": ("Zones irrégulières", "#3B82F6"),
    "pitted_surface": ("Surface piquée", "#F59E0B"),
    "rolled-in_scale": ("Calamine incrustée", "#14B8A6"),
    "scratches": ("Rayures de surface", "#EF4444"),
}

st.set_page_config(
    page_title="SteelVision | Industrial AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. Style visuel
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #0B1020 0%, #11182D 55%, #17152D 100%);
        color: #F1F5F9;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #0D1426;
        border-right: 1px solid #27324D;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #312E81 65%, #4C1D95);
        border: 1px solid #45417F;
        margin-bottom: 24px;
    }

    .hero-label {
        color: #C4B5FD;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 34px;
        margin: 10px 0;
    }

    .hero p {
        color: #DCE5FF;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-title {
        color: #E0E7FF;
        font-size: 21px;
        font-weight: 700;
        margin: 24px 0 12px 0;
    }

    .metric-card {
        background: linear-gradient(145deg, #17213A, #1A2038);
        border: 1px solid #303C5C;
        border-radius: 15px;
        padding: 18px;
        min-height: 112px;
    }

    .metric-label {
        color: #A5B4D4;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 29px;
        font-weight: 750;
    }

    .metric-note {
        color: #A5B4D4;
        font-size: 11px;
        margin-top: 5px;
    }

    .class-card {
        background: #151D32;
        border: 1px solid #2B3652;
        border-radius: 12px;
        padding: 13px;
        margin-bottom: 9px;
    }

    .class-name {
        color: #F1F5F9;
        font-size: 14px;
        font-weight: 650;
    }

    .class-description {
        color: #A5B4D4;
        font-size: 12px;
        margin-top: 5px;
    }

    .dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .footer {
        border-top: 1px solid #303C5C;
        margin-top: 35px;
        padding-top: 18px;
        color: #94A3B8;
        font-size: 12px;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: #17213A;
        border: 1px solid #303C5C;
        padding: 15px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #B8C5E0;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    .stButton button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Barre latérale
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 SteelVision")
    st.caption("INDUSTRIAL COMPUTER VISION")
    st.divider()

    st.markdown("### À propos du modèle")
    st.write("**Architecture :** YOLO11n")
    st.write("**Dataset :** NEU-DET")
    st.write("**Classes :** 6 types de défauts")
    st.write("**Taille d'entrée :** 320 px")

    st.divider()
    st.markdown("### Les classes analysées")

    for class_name, (description, color) in CLASS_INFO.items():
        st.markdown(
            f"""
            <div class="class-card">
                <div class="class-name">
                    <span class="dot" style="background:{color}"></span>
                    {class_name}
                </div>
                <div class="class-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Projet académique — IA & Data Science")

# --------------------------------------------------
# 4. En-tête principal
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Computer Vision · Quality Inspection</div>
        <h1>Industrial Defect Detection</h1>
        <p>
            Détection et localisation automatique des défauts
            sur les surfaces métalliques grâce au Deep Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Cette interface permet d'analyser une image, de visualiser les
    zones détectées et de consulter les prédictions du modèle.
    """
)

# --------------------------------------------------
# 5. Importer une image
# --------------------------------------------------

st.markdown(
    '<div class="section-title">01 · Image à analyser</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Importer une image de surface métallique",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG et PNG.",
)

if uploaded_file is None:
    st.info(
        "Importez une image pour démarrer l'analyse. "
        "Vous pourrez ensuite comparer l'image originale "
        "et les prédictions du modèle."
    )

# --------------------------------------------------
# 6. Lancer l'analyse
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        preview_col, action_col = st.columns([2, 1])

        with preview_col:
            st.image(
                image,
                caption=f"Image originale · {uploaded_file.name}",
                width="stretch",
            )

        with action_col:
            st.markdown("### Prêt pour l'analyse ?")
            st.write(
                "Le modèle YOLO11n va rechercher les défauts "
                "visibles et estimer leur position."
            )

            analyze = st.button(
                "Lancer la détection",
                type="primary",
                use_container_width=True,
            )

        if analyze:
            with st.spinner(
                "Analyse en cours... Le modèle examine l'image."
            ):
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/jpeg",
                        )
                    },
                    timeout=120,
                )

                response.raise_for_status()
                data = response.json()

            detections = data["detections"]
            count = data["number_of_detections"]

            annotated_bytes = base64.b64decode(
                data["annotated_image"]
            )

            annotated_image = Image.open(
                BytesIO(annotated_bytes)
            ).convert("RGB")

            class_counts = Counter(
                detection["class_name"]
                for detection in detections
            )

            average_confidence = (
                sum(d["confidence"] for d in detections) / count
                if count > 0
                else 0
            )

            # ------------------------------------------
            # 7. Indicateurs
            # ------------------------------------------

            st.markdown(
                '<div class="section-title">02 · Vue d’ensemble</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Détections", count)

            with m2:
                st.metric(
                    "Classes identifiées",
                    len(class_counts),
                )

            with m3:
                st.metric(
                    "Confiance moyenne",
                    f"{average_confidence:.1%}",
                    help=(
                        "Moyenne des confiances des boîtes "
                        "détectées, et non une mesure globale "
                        "de la qualité du modèle."
                    ),
                )

            # ------------------------------------------
            # 8. Comparer les images
            # ------------------------------------------

            st.markdown(
                '<div class="section-title">03 · Résultat visuel</div>',
                unsafe_allow_html=True,
            )

            original_col, result_col = st.columns(2)

            with original_col:
                st.markdown("#### Image originale")
                st.image(
                    image,
                    width="stretch",
                )

            with result_col:
                st.markdown("#### Défauts localisés")
                st.image(
                    annotated_image,
                    caption="Rectangles verts : prédictions du modèle",
                    width="stretch",
                )

            if count == 0:
                st.warning(
                    "Aucune détection au seuil de confiance actuel. "
                    "Cela ne prouve pas que l'image est sans défaut."
                )

            # ------------------------------------------
            # 9. Répartition des classes
            # ------------------------------------------

            st.markdown(
                '<div class="section-title">04 · Répartition des prédictions</div>',
                unsafe_allow_html=True,
            )

            if count > 0:
                distribution_col, details_col = st.columns([1, 2])

                with distribution_col:
                    st.markdown("#### Nombre par classe")
                    st.bar_chart(
                        {
                            "Nombre de boîtes": dict(class_counts)
                        },
                        horizontal=True,
                    )

                with details_col:
                    st.markdown("#### Classes détectées")

                    for class_name, class_count in class_counts.most_common():
                        color = CLASS_INFO.get(
                            class_name,
                            ("", "#94A3B8"),
                        )[1]

                        st.markdown(
                            f"""
                            <div class="class-card">
                                <span class="dot"
                                      style="background:{color}"></span>
                                <span class="class-name">{class_name}</span>
                                <span style="float:right;color:#C4B5FD">
                                    {class_count} boîte(s)
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ------------------------------------------
            # 10. Tableau détaillé
            # ------------------------------------------

            st.markdown(
                '<div class="section-title">05 · Détail des détections</div>',
                unsafe_allow_html=True,
            )

            if detections:
                table_data = [
                    {
                        "Classe": d["class_name"],
                        "Confiance": f'{d["confidence"]:.1%}',
                        "X1": d["box"]["x1"],
                        "Y1": d["box"]["y1"],
                        "X2": d["box"]["x2"],
                        "Y2": d["box"]["y2"],
                    }
                    for d in detections
                ]

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = st.session_state.get(
                    "unused_csv_placeholder",
                    None,
                )

                import pandas as pd

                csv_bytes = pd.DataFrame(table_data).to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Exporter les résultats en CSV",
                    data=csv_bytes,
                    file_name="industrial_defect_predictions.csv",
                    mime="text/csv",
                )

            else:
                st.write(
                    "Aucune boîte à afficher dans le tableau."
                )

            # ------------------------------------------
            # 11. Note méthodologique
            # ------------------------------------------

            st.caption(
                "Important : les résultats sont des prédictions "
                "expérimentales. Plusieurs boîtes peuvent parfois "
                "correspondre à une même zone. La confiance du modèle "
                "n'est pas une garantie de détection correcte."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "API inaccessible. Vérifiez que FastAPI tourne "
            "sur http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:
        st.error(
            "L'analyse a dépassé le délai d'attente. "
            "Réessayez avec une autre image."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Erreur lors de l'appel à l'API : {exc}")

    except (ValueError, KeyError) as exc:
        st.error(f"Réponse inattendue de l'API : {exc}")

    except Exception as exc:
        st.error(f"Une erreur est survenue : {exc}")

# --------------------------------------------------
# 12. Pied de page
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        STEELVISION · INDUSTRIAL AI · YOLO11n · NEU-DET
        <br>
        Prototype académique de contrôle visuel industriel
    </div>
    """,
    unsafe_allow_html=True,
)
