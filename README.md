# NovelForge Writer Server

Bootstrap simples para rodar o Qwen3 1.7B GGUF com llama.cpp em um container Python genérico.

## Enzonic

Use o template Python e configure:

- `GIT_ADDRESS`: `https://github.com/karlosantonio2299/novelforge-writer-server`
- `BRANCH`: `main`
- `PY_FILE`: `app.py`
- `REQUIREMENTS_FILE`: `requirements.txt`
- `AUTO_UPDATE`: `1`
- `USER_UPLOAD`: `0`
- `PY_PACKAGES`: deixe vazio
- `ACCESS_TOKEN`: deixe vazio (repo público)
- `USERNAME`: deixe vazio

O script detecta a porta usando, nesta ordem: `PORT`, `SERVER_PORT`, `APP_PORT`, `P_SERVER_PORT`, com fallback para `8080`.

## Ajustes opcionais

- `N_CTX` (padrão `1024`)
- `LLAMA_THREADS` (padrão `2`)
- `LLAMA_THREADS_BATCH` (padrão `2`)
- `LLAMA_CPP_RELEASE` (padrão `b6162`)

O modelo é baixado automaticamente de:

`exebr/novelforge-qwen3-1.7b-q3/Qwen3-1.7B-Q3_K_L.gguf`

> Observação: o repositório do modelo no Hugging Face precisa estar acessível sem autenticação para esse bootstrap funcionar como está.
