"""
Create the default Firestore database via the Firestore Admin API when it is missing.

This avoids the Firebase Console wizard. The service account in GOOGLE_APPLICATION_CREDENTIALS
must be allowed to create databases (e.g. roles/datastore.owner or datastore.databases.create).
"""
from __future__ import annotations

import logging
import os

from claimguard.config import load_environment

logger = logging.getLogger(__name__)

_DEFAULT_DB_ID = "(default)"


def _auto_provision_enabled() -> bool:
    load_environment()
    raw = os.getenv("FIRESTORE_AUTO_PROVISION", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _firestore_admin_client():
    from google.cloud.firestore_admin_v1 import FirestoreAdminClient
    from google.oauth2 import service_account

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if cred_path:
        creds = service_account.Credentials.from_service_account_file(cred_path)
        return FirestoreAdminClient(credentials=creds)
    return FirestoreAdminClient()


def _default_db_exists(client, parent: str) -> bool:
    response = client.list_databases(parent=parent)
    for db in response.databases:
        if db.name.endswith(f"/databases/{_DEFAULT_DB_ID}"):
            return True
    return False


def ensure_default_firestore_database() -> None:
    """
    If not in test mode and FIRESTORE_AUTO_PROVISION is on, ensure `(default)` exists
    (Native mode). No-op if the database is already there or auto-provision is disabled.
    """
    from claimguard.firebase_config import is_test_environment

    load_environment()
    if is_test_environment() or not _auto_provision_enabled():
        return

    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is required when Firebase is enabled")

    location_id = os.getenv("FIRESTORE_LOCATION", "eur3").strip() or "eur3"
    timeout_s = float(os.getenv("FIRESTORE_CREATE_TIMEOUT_SECONDS", "600"))

    from google.api_core import exceptions as gexc
    from google.cloud.firestore_admin_v1.types import Database

    parent = f"projects/{project_id}"
    client = _firestore_admin_client()

    try:
        if _default_db_exists(client, parent):
            logger.info("Firestore database %s already exists for project %s", _DEFAULT_DB_ID, project_id)
            return
    except gexc.GoogleAPIError as exc:
        logger.warning("Could not list Firestore databases (%s); attempting create anyway.", exc)

    db = Database(
        location_id=location_id,
        type_=Database.DatabaseType.FIRESTORE_NATIVE,
    )
    try:
        operation = client.create_database(
            parent=parent,
            database_id=_DEFAULT_DB_ID,
            database=db,
        )
        logger.info(
            "Creating Firestore database %s in %s (this can take several minutes)...",
            _DEFAULT_DB_ID,
            location_id,
        )
        operation.result(timeout=timeout_s)
        logger.info("Firestore database %s is ready.", _DEFAULT_DB_ID)
    except gexc.AlreadyExists:
        logger.info("Firestore database %s already exists (race or concurrent create).", _DEFAULT_DB_ID)
    except gexc.PermissionDenied as exc:
        raise RuntimeError(
            "Firestore auto-provisioning was denied. Grant this service account permission to "
            "create databases (e.g. IAM role roles/datastore.owner on the project), or create "
            "the database in the console and set FIRESTORE_AUTO_PROVISION=0."
        ) from exc
