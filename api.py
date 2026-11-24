
import os
import sys
import logging
import threading
import time

from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache

# ---------------------------------------------------------
# CONFIGURATION DES IMPORTS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Header,
    Request,
)

from pydantic import BaseModel, Field

from plagiatguard.storage import (
    init_db,
    save_analysis,
    audit,
)

from plagiatguard.security import decode_token
from plagiatguard.quotas import consume_quota

from plagiatguard.detector import (
    analyze_bytes,
    analyze_text,
)

from plagiatguard.config import MAX_UPLOAD_MB


# ---------------------------------------------------------
# IMPORTATION DU PIPELINE AVANCÉ
# ---------------------------------------------------------

try:
    from plagiatguard.research_retrieval import PassageIndex
    from plagiatguard.research_pipeline import ResearchPipeline

    RESEARCH_PIPELINE_AVAILABLE = True

except ImportError:
    PassageIndex = None
    ResearchPipeline = None

    RESEARCH_PIPELINE_AVAILABLE = False


# ---------------------------------------------------------
# CONFIGURATION GÉNÉRALE
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("plagiatguard.api")

init_db()

app = FastAPI(
    title="PlagiatGuard Advanced API",
    description=(
        "API de détection de similarité textuelle, "
        "de recherche sémantique et d'analyse documentaire."
    ),
    version="2.1.0",
)

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".odt",
}


# ---------------------------------------------------------
# CONFIGURATION DE LA RECHERCHE WEB
# ---------------------------------------------------------

def is_web_search_configured():
    """
    La recherche Web nécessite :
    1. ENABLE_WEB_SEARCH=true dans le serveur.
    2. Une clé TAVILY_API_KEY valide.
    """

    enabled = os.getenv(
        "ENABLE_WEB_SEARCH", "false"
    ).lower() in {"true", "1", "yes"}

    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    return enabled and bool(api_key)


WEB_SEARCH_CONFIGURED = is_web_search_configured()


# ---------------------------------------------------------
# LIMITATION DES REQUÊTES PAR IP
# ---------------------------------------------------------

IP_MAX_REQUESTS = 60
IP_WINDOW_SECONDS = 60

ip_requests = defaultdict(deque)
ip_lock = threading.Lock()


def check_ip_rate_limit(request: Request):
    """
    Limite le nombre de requêtes par adresse IP.
    Configuration mono-processus.
    """

    ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    now = time.time()
    cutoff = now - IP_WINDOW_SECONDS

    with ip_lock:

        requests = ip_requests[ip]

        while requests and requests[0] < cutoff:
            requests.popleft()

        if len(requests) >= IP_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Trop de requêtes. Réessayez dans une minute.",
            )

        requests.append(now)


# ---------------------------------------------------------
# GESTION DES RÔLES
# ---------------------------------------------------------

ROLE_LEVELS = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


# ---------------------------------------------------------
# AUTHENTIFICATION JWT
# ---------------------------------------------------------

def current_user(
    request: Request,
    authorization: str | None = Header(default=None),
):
    """
    Vérifie le token JWT et récupère l'utilisateur connecté.
    """

    check_ip_rate_limit(request)

    if (
        not authorization
        or not authorization.startswith("Bearer ")
    ):
        raise HTTPException(
            status_code=401,
            detail="Token Bearer requis.",
        )

    token = authorization.split(" ", 1)[1].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Token manquant.",
        )

    try:

        user = decode_token(token)

        if not isinstance(user, dict):
            raise ValueError("Payload JWT invalide.")

        if not user.get("sub"):
            raise ValueError("Identifiant utilisateur manquant.")

        if user.get("role") not in ROLE_LEVELS:
            raise ValueError("Rôle invalide.")

        return user

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Token invalide ou expiré.",
        )


# ---------------------------------------------------------
# AUTORISATION PAR RÔLE
# ---------------------------------------------------------

def require_role(minimum_role: str):
    """
    Vérifie que l'utilisateur possède le rôle requis.
    """

    if minimum_role not in ROLE_LEVELS:
        raise ValueError("Rôle inconnu.")

    def role_dependency(
        user=Depends(current_user),
    ):

        user_role = user.get("role", "viewer")

        if ROLE_LEVELS.get(user_role, 0) < ROLE_LEVELS[minimum_role]:

            raise HTTPException(
                status_code=403,
                detail="Permissions insuffisantes.",
            )

        return user

    return role_dependency


# ---------------------------------------------------------
# INITIALISATION DU PIPELINE DE RECHERCHE AVANCÉ
# ---------------------------------------------------------

def create_research_pipeline():
    """
    Crée et charge le moteur de recherche documentaire.
    """

    if not RESEARCH_PIPELINE_AVAILABLE:
        return None

    index_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "research_index"
    )

    passage_index = PassageIndex(
        index_dir=str(index_dir)
    )

    load_result = passage_index.load()

    if load_result is False or passage_index.backend is None:
        raise RuntimeError(
            "Impossible de charger l'index de recherche."
        )

    pipeline = ResearchPipeline(
        passage_index=passage_index,
        top_k=5,
        local_threshold=0.25,
    )

    logger.info(
        "Pipeline de recherche initialisé avec le moteur : %s",
        passage_index.backend,
    )

    return pipeline


@lru_cache(maxsize=1)
def get_research_pipeline():
    """
    Charge le pipeline une seule fois.
    """

    if not RESEARCH_PIPELINE_AVAILABLE:
        return None

    try:
        return create_research_pipeline()

    except Exception:

        logger.exception(
            "Impossible d'initialiser le pipeline de recherche."
        )

        return None


def reload_research_pipeline():
    """
    Vide le cache et recharge le pipeline.
    """

    get_research_pipeline.cache_clear()

    return get_research_pipeline()


# ---------------------------------------------------------
# INTÉGRATION DE L'ANALYSE AVANCÉE
# ---------------------------------------------------------

def add_research_analysis(
    report: dict,
    text: str,
    enable_web: bool = False,
):
    """
    Ajoute les résultats de recherche sémantique
    et éventuellement Web au rapport classique.

    La recherche Web est autorisée uniquement si
    elle est demandée par le client ET configurée côté serveur.
    """

    pipeline = get_research_pipeline()

    if pipeline is None:

        report["advanced_research"] = {
            "enabled": False,
            "message": "Pipeline avancé indisponible.",
        }

        return report

    # Contrôle de la configuration côté serveur.
    web_active = (
        enable_web
        and is_web_search_configured()
    )

    if enable_web and not web_active:
        logger.warning(
            "Recherche Web demandée mais non configurée côté serveur."
        )

    try:

        # Analyse documentaire avancée.
        advanced_report = pipeline.analyze(
            text,
            enable_web=web_active,
        )

        local_matches = advanced_report.get(
            "local_matches",
            [],
        )

        web_matches = advanced_report.get(
            "web_matches",
            [],
        )

        # Résultats détaillés.
        report["advanced_research"] = {
            "enabled": True,

            "engine": getattr(
                pipeline.passage_index,
                "backend",
                "unknown",
            ),

            "web_requested": enable_web,
            "web_configured": is_web_search_configured(),

            "results": advanced_report,
        }

        # Meilleur score sémantique local.
        max_semantic_score = max(
            (
                float(match.get("score", 0.0))
                for match in local_matches
            ),
            default=0.0,
        )

        # Résumé séparé des moteurs.
        report["similarity_summary"] = {

            "classic": {
                "score": report.get(
                    "adjusted_similarity",
                    0.0,
                ),

                "flagged": report.get(
                    "flagged",
                    False,
                ),

                "threshold": report.get(
                    "threshold",
                    None,
                ),
            },

            "semantic": {
                "match_count": len(local_matches),

                "max_score": max_semantic_score,

                "engine": getattr(
                    pipeline.passage_index,
                    "backend",
                    "unknown",
                ),
            },

            "web": {
                "enabled": web_active,

                "match_count": len(web_matches),

                "source_count": advanced_report.get(
                    "web_source_count",
                    0,
                ),

                "error": advanced_report.get(
                    "web_error",
                ),
            },

            "requires_manual_review": bool(
                local_matches or web_matches
            ),
        }

        if enable_web and not web_active:
            report["advanced_research"]["web_message"] = (
                "La recherche Web est demandée mais désactivée "
                "ou non configurée sur le serveur."
            )

    except Exception:

        logger.exception(
            "Erreur pendant l'analyse avancée."
        )

        report["advanced_research"] = {
            "enabled": False,
            "message": (
                "Recherche avancée temporairement indisponible."
            ),
        }

    return report


# ---------------------------------------------------------
# MODÈLE DE REQUÊTE TEXTUELLE
# ---------------------------------------------------------

class TextRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=200_000,
    )

    threshold: float = Field(
        default=0.55,
        ge=0.05,
        le=0.99,
    )

    enable_web: bool = False


# ---------------------------------------------------------
# VALIDATION DES FICHIERS
# ---------------------------------------------------------

def validate_filename(filename: str | None):
    """
    Vérifie le nom et l'extension du fichier.
    """

    if not filename:

        raise HTTPException(
            status_code=400,
            detail="Nom de fichier manquant.",
        )

    safe_name = Path(
        filename.replace("\\", "/")
    ).name

    extension = Path(safe_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=415,
            detail=(
                "Format non accepté. "
                "Formats autorisés : TXT, PDF, DOCX, ODT."
            ),
        )

    return safe_name


# ---------------------------------------------------------
# ROUTE HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "2.1.0",
        "research_pipeline": RESEARCH_PIPELINE_AVAILABLE,
        "web_search_configured": is_web_search_configured(),
    }


# ---------------------------------------------------------
# ANALYSE D'UN TEXTE
# ---------------------------------------------------------

@app.post("/analyze/text")
def analyze_text_endpoint(
    payload: TextRequest,
    user=Depends(require_role("analyst")),
):
    """
    Analyse un texte avec :
    - Détecteur classique.
    - Recherche sémantique locale.
    - Recherche Web facultative.
    """

    # Vérification du quota.
    allowed, remaining = consume_quota(
        user["sub"]
    )

    if not allowed:

        raise HTTPException(
            status_code=429,
            detail="Quota quotidien atteint.",
        )

    try:

        # 1. Analyse classique.
        report = analyze_text(
            payload.text,
            "text-input.txt",
            payload.threshold,
        )

        # 2. Analyse avancée.
        report = add_research_analysis(
            report,
            payload.text,
            enable_web=payload.enable_web,
        )

        # 3. Enregistrement.
        save_analysis(
            user["sub"],
            "text-input.txt",
            payload.text.encode("utf-8"),
            report.get(
                "adjusted_similarity",
                0.0,
            ),
            report,
        )

        # 4. Audit.
        audit(
            user["sub"],
            "analyze_text",
        )

        return {
            "remaining_quota": remaining,
            "report": report,
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Erreur pendant l'analyse du texte."
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant l'analyse.",
        )


# ---------------------------------------------------------
# ANALYSE D'UN FICHIER
# ---------------------------------------------------------

@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    threshold: float = 0.55,
    enable_web: bool = False,
    user=Depends(require_role("analyst")),
):
    """
    Analyse un fichier TXT, PDF, DOCX ou ODT.
    """

    filename = validate_filename(
        file.filename
    )

    if not 0.05 <= threshold <= 0.99:

        raise HTTPException(
            status_code=400,
            detail="Seuil invalide.",
        )

    try:

        # Lecture limitée.
        content = await file.read(
            MAX_UPLOAD_BYTES + 1
        )

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Le fichier est vide.",
            )

        if len(content) > MAX_UPLOAD_BYTES:

            raise HTTPException(
                status_code=413,
                detail="Fichier trop volumineux.",
            )

        # Vérification du quota.
        allowed, remaining = consume_quota(
            user["sub"]
        )

        if not allowed:

            raise HTTPException(
                status_code=429,
                detail="Quota quotidien atteint.",
            )

        # 1. Analyse classique et extraction.
        extracted_text, report = analyze_bytes(
            filename,
            content,
            threshold,
        )

        # 2. Analyse avancée.
        if extracted_text and extracted_text.strip():

            report = add_research_analysis(
                report,
                extracted_text,
                enable_web=enable_web,
            )

        # 3. Enregistrement.
        save_analysis(
            user["sub"],
            filename,
            content,
            report.get(
                "adjusted_similarity",
                0.0,
            ),
            report,
        )

        # 4. Audit.
        audit(
            user["sub"],
            "analyze_file",
        )

        return {
            "remaining_quota": remaining,
            "report": report,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Erreur pendant l'analyse du fichier."
        )

        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant l'analyse du fichier.",
        )

    finally:

        await file.close()


# ---------------------------------------------------------
# RECHARGEMENT DE L'INDEX - ADMINISTRATEUR
# ---------------------------------------------------------

@app.post("/admin/reload-index")
def reload_research_index(
    user=Depends(require_role("admin")),
):
    """
    Recharge l'index de recherche documentaire.
    Accessible uniquement aux administrateurs.
    """

    if not RESEARCH_PIPELINE_AVAILABLE:

        raise HTTPException(
            status_code=503,
            detail="Pipeline de recherche non disponible.",
        )

    try:

        pipeline = reload_research_pipeline()

        if pipeline is None:

            raise HTTPException(
                status_code=503,
                detail="Impossible de charger le nouvel index.",
            )

        audit(
            user["sub"],
            "reload_research_index",
        )

        return {
            "status": "success",

            "message": "Index de recherche rechargé.",

            "engine": getattr(
                pipeline.passage_index,
                "backend",
                "unknown",
            ),

            "passage_count": len(
                pipeline.passage_index.passages
            ),
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Erreur pendant le rechargement de l'index."
        )

        raise HTTPException(
            status_code=500,
            detail="Impossible de recharger l'index.",
        )
