import os
import requests
import streamlit as st

st.set_page_config(
    page_title="PlagiatGuard Advanced",
    page_icon="🔎",
    layout="wide",
)

API_URL = os.getenv(
    "PLAGIATGUARD_API_URL",
    "http://127.0.0.1:8000",
)

st.title("🔎 PlagiatGuard Advanced")

st.warning(
    "Les résultats sont des indicateurs de similarité, "
    "pas une preuve automatique de plagiat."
)

st.sidebar.header("Authentification")

token = st.sidebar.text_input(
    "Token JWT",
    type="password",
)

threshold = st.slider(
    "Seuil de similarité",
    min_value=0.10,
    max_value=0.95,
    value=0.55,
    step=0.05,
)

enable_web = st.checkbox(
    "Activer la recherche Web",
    value=False,
    help="La recherche Web nécessite une configuration serveur valide.",
)

uploaded = st.file_uploader(
    "Importer un document",
    type=["txt", "pdf", "docx", "odt"],
)

if uploaded and st.button(
    "Analyser",
    type="primary",
):

    if not token:
        st.error("Veuillez fournir un token JWT.")

    else:

        headers = {
            "Authorization": f"Bearer {token}",
        }

        files = {
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/octet-stream",
            )
        }

        params = {
            "threshold": threshold,
            "enable_web": enable_web,
        }

        try:
            with st.spinner("Analyse en cours..."):

                response = requests.post(
                    f"{API_URL}/analyze/file",
                    headers=headers,
                    files=files,
                    params=params,
                    timeout=180,
                )

                response.raise_for_status()

                result = response.json()

            report = result["report"]

            st.success("Analyse terminée.")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Similarité maximale",
                f'{report.get("max_similarity", 0):.1%}',
            )

            col2.metric(
                "Score ajusté",
                f'{report.get("adjusted_similarity", 0):.1%}',
            )

            col3.metric(
                "Quota restant",
                result.get("remaining_quota", "N/A"),
            )

            st.info(report.get("notice", ""))

            st.subheader("Résultats de l'analyse classique")

            st.write(
                "Correspondance à examiner"
                if report.get("flagged")
                else "Aucune correspondance au-dessus du seuil"
            )

            if report.get("matches"):
                st.dataframe(
                    report["matches"],
                    use_container_width=True,
                )

            advanced = report.get(
                "advanced_research",
                {},
            )

            if advanced.get("enabled"):

                research = advanced.get("results", {})

                st.subheader("Recherche sémantique locale")

                st.metric(
                    "Correspondances FAISS",
                    research.get("local_match_count", 0),
                )

                if research.get("local_matches"):
                    st.dataframe(
                        research["local_matches"],
                        use_container_width=True,
                    )

                st.subheader("Recherche Web")

                st.write(
                    "Recherche Web activée"
                    if research.get("web_enabled")
                    else "Recherche Web désactivée"
                )

                st.metric(
                    "Sources Web",
                    research.get("web_source_count", 0),
                )

                st.metric(
                    "Correspondances Web",
                    research.get("web_match_count", 0),
                )

                if research.get("web_error"):
                    st.warning(research["web_error"])

                if research.get("web_matches"):
                    st.dataframe(
                        research["web_matches"],
                        use_container_width=True,
                    )

                if research.get("web_sources"):
                    st.subheader("Sources externes")

                    for source in research["web_sources"]:
                        st.markdown(
                            f"- [{source.get('title', 'Source')}]"
                            f"({source.get('url', '')})"
                        )

            else:
                st.warning(
                    advanced.get(
                        "message",
                        "Recherche avancée indisponible.",
                    )
                )

            st.download_button(
                "Télécharger le rapport JSON",
                data=__import__("json").dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="rapport.json",
                mime="application/json",
            )

        except requests.exceptions.HTTPError as exc:
            try:
                detail = response.json().get(
                    "detail",
                    str(exc),
                )
            except Exception:
                detail = str(exc)

            st.error(f"Erreur API : {detail}")

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Impossible de contacter FastAPI : {exc}"
            )

        except Exception as exc:
            st.error(f"Erreur pendant l'analyse : {exc}")