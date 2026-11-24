import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "src")
)

import streamlit as st

from plagiatguard.config import MAX_UPLOAD_MB
from plagiatguard.detector import analyze_bytes
from plagiatguard.reporting import to_json, to_csv
from plagiatguard.storage import (
    init_db,
    save_analysis,
    audit,
)

init_db()

st.set_page_config(
    page_title="PlagiatGuard Advanced",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 PlagiatGuard Advanced")

st.warning(
    "Les résultats sont des indicateurs de similarité, "
    "pas une preuve automatique de plagiat."
)

username = st.text_input(
    "Identifiant local",
    value="local-user",
)

threshold = st.slider(
    "Seuil de similarité",
    min_value=0.10,
    max_value=0.95,
    value=0.55,
    step=0.05,
)

uploaded = st.file_uploader(
    "Importer un document",
    type=[
        "txt", "md", "pdf", "docx", "odt",
        "html", "htm", "xml", "json", "csv",
        "tsv", "py", "js", "ts", "java",
        "cpp", "c", "h", "ipynb",
    ],
)

if uploaded:
    content = uploaded.getvalue()

    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("Fichier trop volumineux.")
    elif st.button("Analyser", type="primary"):
        try:
            with st.spinner("Analyse en cours..."):
                _, report = analyze_bytes(
                    uploaded.name,
                    content,
                    threshold,
                )

                save_analysis(
                    username,
                    uploaded.name,
                    content,
                    report["adjusted_similarity"],
                    report,
                )

                audit(
                    username,
                    "analysis",
                    uploaded.name,
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Similarité maximale",
                f'{report["max_similarity"]:.1%}',
            )

            col2.metric(
                "Score ajusté indicatif",
                f'{report["adjusted_similarity"]:.1%}',
            )

            st.write(
                "**Résultat :**",
                "Correspondance à examiner"
                if report["flagged"]
                else "Aucune correspondance au-dessus du seuil",
            )

            st.info(report["notice"])

            if report["matches"]:
                st.subheader("Correspondances du corpus local")
                st.dataframe(
                    report["matches"],
                    use_container_width=True,
                )
            else:
                st.write("Aucune correspondance trouvée.")

            st.download_button(
                "Télécharger le rapport JSON",
                data=to_json(report),
                file_name="rapport.json",
                mime="application/json",
            )

            st.download_button(
                "Télécharger les correspondances CSV",
                data=to_csv(report),
                file_name="correspondances.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Erreur pendant l'analyse : {exc}")