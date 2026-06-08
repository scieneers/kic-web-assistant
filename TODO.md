# TODO

- [ ] ACR-Authentifizierung von Admin Account auf Service Principal umstellen
  - GitHub Secrets `ACR_USERNAME` / `ACR_PASSWORD` durch Service Principal Credentials oder OIDC Federated Credentials ersetzen
  - ACR Admin Account danach deaktivieren
- [ ] Langfuse austauschen?
- [ ] Check which model the frontends are using and migrate them from `LLAMA3`/`gemma-4-31b-it` to the `gemma4` enum — once done, remove the compatibility mapping in `src/llm/objects/LLMs.py`
- [ ] Loader run-log Blob unter Managed Identity reparieren (Known Issue, geparkt)
  - `src/loaders/run_logger.py` baut den `BlobServiceClient` aus einem **Connection String** (`RUN_LOGS_BLOB_CONNECTION_STRING` bzw. `AzureWebJobsStorage`)
  - Die Data-Loader Function App nutzt `storage_uses_managed_identity = true` → es existiert **kein** Connection String (nur `AzureWebJobsStorage__accountName` + Identity) → das SAS-Run-Log-URL-Feature läuft leer
  - stdout-Logging ist davon nicht betroffen und funktioniert weiter
  - Fix: auf `BlobServiceClient(account_url="https://<acct>.blob.core.windows.net", credential=DefaultAzureCredential())` umstellen; UAI braucht dafür `Storage Blob Data Contributor` (in der IaC nun in dev + prod gesetzt)