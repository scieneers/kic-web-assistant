# speichere z.B. als check_qdrant_collection.py im Repo-Root

from qdrant_client.http.exceptions import UnexpectedResponse
from src.vectordb.qdrant import VectorDBQdrant

COLLECTION = "web_assistant_hybrid"


def check(version: str) -> None:
    try:
        db = VectorDBQdrant(mode=version)
        info = db.client.get_collection(collection_name=COLLECTION)
        print(f"[{version}] Collection '{COLLECTION}' EXISTIERT.")
        print(f"[{version}] Details: {info}")
    except UnexpectedResponse:
        print(f"[{version}] Collection '{COLLECTION}' existiert NICHT.")
    except Exception as e:
        print(f"[{version}] Fehler beim Prüfen: {e}")

def check_payload(version: str) -> None:
    try:
        db = VectorDBQdrant(mode=version)
        points = db.client.scroll(
            collection_name=COLLECTION,
            limit=1
        )
        print(points)
    except Exception as e:
        print(f"[{version}] Fehler beim Abrufen der Payload: {e}")

if __name__ == "__main__":
    check("dev_remote")
    check("prod_remote")
    #check_payload("dev_remote")
    #check_payload("prod_remote")