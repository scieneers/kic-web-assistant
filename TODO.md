# TODO

- [ ] ACR-Authentifizierung von Admin Account auf Service Principal umstellen
  - GitHub Secrets `ACR_USERNAME` / `ACR_PASSWORD` durch Service Principal Credentials oder OIDC Federated Credentials ersetzen
  - ACR Admin Account danach deaktivieren
- [ ] Langfuse austauschen?
- [ ] Check which model the frontends are using and migrate them from `LLAMA3`/`gemma-4-31b-it` to the `gemma4` enum — once done, remove the compatibility mapping in `src/llm/objects/LLMs.py`