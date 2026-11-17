# Guia de execução e passagem para a IA de treino

Todos os comandos de treino abaixo rodam no **terminal Ubuntu/WSL**, na raiz do
repositório. Substitua caminhos entre aspas pelos caminhos reais dos runs. Não
copie literalmente `CAMINHO_DO_RUN`. Não inicie campanha antes do portão de smoke.

Se usar a candidata L40S em Linux/container, começar pelo roteiro específico
[VALIDAR_L40S.md](VALIDAR_L40S.md), que inclui diagnóstico de cgroups e o caminho
alternativo sem acesso ao Docker. A mudança de máquina ainda depende das medições.

## 1. Preparar Windows 11 e Docker

No PowerShell como administrador, caso o WSL ainda não esteja instalado:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Reiniciar se solicitado. Instalar Docker Desktop, selecionar backend WSL2 e
habilitar integração com Ubuntu. Abrir Ubuntu, clonar o repositório dentro do
filesystem Linux (por exemplo `~/projetos/rl-copa-trabalho-final`), para reduzir
tráfego de arquivos pelo mount do Windows. Usar checkout que inclua os arquivos
novos deste projeto; uma cópia só do commit antigo não contém este pipeline.

O suporte oficial de GPU do Docker Desktop no Windows usa WSL2 e GPU NVIDIA.
Não é requisito do caminho inicial em CPU. Fontes:
[Microsoft WSL](https://learn.microsoft.com/en-us/windows/wsl/install),
[Docker GPU](https://docs.docker.com/desktop/features/gpu/).

Na raiz do repositório, criar `.env` se não existir:

```bash
test -f .env || cp .env.example .env
export GID="$(id -g)"
export UID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
docker compose build gfootball
docker image inspect t-zero-gfootball:latest --format '{{.Id}}'
docker compose run --rm gfootball python -c 'import torch, gymnasium, gfootball, gfootball_engine; print(torch.__version__); print(gfootball_engine.__file__)'
```

Os exemplos adiante passam limites de threads explicitamente ao container;
exportar no host sozinho não passa todas as variáveis ao Compose. Em shells
onde `UID` não existe, definir `export UID="$(id -u)"`. Em Bash/WSL, `UID` já
existe e é somente leitura: apenas exportá-lo, como acima.

Não é necessário preencher W&B: configs do projeto usam `track: false`. Não
publicar chaves do `.env`. Guardar ID da imagem e commit do motor, que esta
imagem mantém em `/opt/football`:

```bash
mkdir -p results/setup
docker image inspect t-zero-gfootball:latest --format '{{.Id}}' > results/setup/image-id.txt
docker compose run --rm gfootball git -c safe.directory=/opt/football -C /opt/football rev-parse HEAD > results/setup/football-commit.txt
docker compose run --rm gfootball python -m pip freeze > results/setup/packages.txt
```

Reutilizar essa imagem durante a campanha. Guardar também o checkout exato e,
se precisar transferir a imagem, exportá-la com `docker image save` para backup.
O Dockerfile base usa master; reconstruir em outra data não é garantia de mesma
versão. O treino registra pacotes por sessão, mas o ID da imagem vem do host.

## 2. Testes e smoke: obrigatório antes dos runs longos

Depois de construir a imagem, a cadeia completa também pode ser executada com
um comando (sempre usar um identificador novo):

```bash
bash scripts/football_smoke.sh smoke-validacao-01
```

O script grava testes, treino de 4096 passos, exportação, checker e avaliação
com quatro jogos completos (lados alternados) em `results/smoke-validacao-01/`, e o agente em
`submissions/smoke-validacao-01/`. Ele para se qualquer etapa falhar e não
sobrescreve evidências antigas. Os comandos individuais estão abaixo.

```bash
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python -m pytest tests/test_ppo.py tests/test_football_project.py tests/test_configs_load.py tests/test_config_roundtrip.py tests/test_wrappers.py tests/test_checkpoint.py
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python -m pytest tests/test_gfootball.py tests/test_arena.py
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --config ppo_football_sparse --override total_timesteps=4096 num_envs=2 ppo.num_steps=128 checkpoint_every=512
```

O runner imprime `PROJECT_RUN_DIR=...`. Usar exatamente esse caminho em seguida:

```bash
docker compose run --rm gfootball python scripts/export_submission.py CAMINHO_DO_RUN --out submissions/smoke --team smoke-e00
docker compose run --rm gfootball python -m arena.check submissions/smoke
```

Esperar `PASS`; um modelo quase aleatório pode passar. Isso verifica interface,
não força de jogo. O exportador grava probe a partir da política de treino e
verifica conversão de observações + TorchScript. O checker reproduz o probe e
executa jogos. Preservar a saída do checker com `tee` se quiser evidência:

```bash
set -o pipefail
docker compose run --rm gfootball python -m arena.check submissions/smoke 2>&1 | tee results/setup/check-smoke.txt
```

O checker precisa rodar em CPU na imagem original. Não alterar `arena/rules.py`,
o checker, limites de tempo ou o ambiente para obter PASS. Se falhar, corrigir
agente/exportação/dependências; registrar erro e comando antes de tentar de novo.

## 3. Benchmark e orçamento

Rodar E01 inicialmente com 65.536 passos, anotando `session_sps`, RAM e tempo
total. Esse é benchmark operacional, excluído das alegações de desempenho:

```bash
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --config ppo_football_sparse --override total_timesteps=65536
```

O orçamento configurado conta chamadas somadas dos ambientes; inclui os passos
de autoreset ignorados pelo motor no contrato Gymnasium. A atualização PPO
descarta essas transições. `completed.json` informa passos efetivamente feitos,
arredondados para baixo em múltiplos do rollout. Comparar orçamentos na mesma
convenção. O treino ainda realiza avaliação genérica ao salvar o modelo; ela
não substitui a arena e usa a configuração de recompensa daquele run.

Para benchmark de mais workers, manter batch constante: comparar 4×128 com
8×64; escolher pelo tempo e memória. Registrar essa decisão. Não mudar batch
apenas em uma condição experimental. A agenda é `[passos de fase]/SPS` mais
avaliações; não há previsão de desempenho garantida para o hardware informado.

## 4. Executar E01 e E02

Rodar seeds sequencialmente antes de tentar concorrência. O comando abaixo
para na primeira falha e grava também stderr. Preservar logs, métricas e runs:

```bash
mkdir -p results/train-logs
set -euo pipefail
for config in ppo_football_sparse ppo_football_checkpoints; do
  for seed in 1 2 3; do
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball \
      python scripts/football_train.py --config "$config" --override "seed=$seed" \
      2>&1 | tee "results/train-logs/${config}_s${seed}_${stamp}.txt"
  done
done
```

As configs usam 1.048.576 passos. Se o piloto justificar orçamento diferente,
fixá-lo antes dos runs comparativos e adicionar o mesmo `total_timesteps=...`
a ambas as condições. Não aumentar apenas o método que parece melhor. Preencher
`experiments/runs.csv` com os caminhos impressos, inclusive runs que falharem.

## 5. Retomar interrupção e iniciar fase nova

Interrupção: perderá o trabalho após o último checkpoint; o simulador reinicia.
Retomar sem trocar ambiente, representação ou recompensas:

```bash
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --resume CAMINHO_DO_RUN
```

Novas sessões acrescentam arquivos `session_*.json` e linhas nas métricas. Após
rollback de checkpoint pode haver passos duplicados; o plot remove a cauda
abandonada. O JSONL bruto permanece intacto. Não comparar uma curva interrompida
como se fosse trajetória idêntica a treino ininterrupto.

O runner aceita em retomadas apenas overrides de `total_timesteps`, `cuda` e
`track`. `config.yml` é a configuração inicial; cada `session_*.json` inclui a
configuração resolvida daquela sessão, inclusive eventual extensão de orçamento.
Alterar o teto com annealing muda a trajetória da agenda: declarar a extensão,
não tratá-la como treino contínuo com orçamento fixo desde o início.

Run concluído: `model.pt` guarda só os pesos; a base apaga os checkpoints.
Para extensão ou nova fase, usar outro run e registrar a linhagem:

```bash
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --config ppo_football_academy --override seed=1
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --config ppo_football_checkpoints --init-from CAMINHO_DO_RUN_ACADEMY --override seed=1 total_timesteps=786432
```

Isso transfere ator e crítico, exige mesma rede/kwargs/representação e reinicia
Adam e a agenda de learning rate. A fase tem seus próprios passos a partir de
zero. Somar 262.144 do pai aos 786.432 do filho; não chamar o total de 786.432.
O runner registra caminho e SHA256 do pai. Validar compatibilidade semântica
dos estados além de shapes antes de propor uma nova representação.

Self-play histórico opcional: primeiro exportar um modelo anterior com nome de
equipe único e passar o caminho Linux relativo ao repo como adversário:

```bash
docker compose run --rm gfootball python scripts/export_submission.py RUN_ANTERIOR --out submissions/old-v1 --team old-v1
docker compose run --rm gfootball python -m arena.check submissions/old-v1
docker compose run --rm -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball python scripts/football_train.py --config ppo_football_checkpoints --init-from RUN_PAI --override env_kwargs.opponent=submissions/old-v1
```

Não sobrescrever `old-v1` durante o treino. Fazer o controle com o mesmo pai,
budget, seed, nova agenda e `env_kwargs.opponent=builtin`. Este runner não faz
pool/mistura dinâmica de adversários. Preservar adversários junto com runs.

## 6. Exportar e avaliar cada candidato

Nomes únicos em manifest por condição, seed e versão evitam confundir adversários:

```bash
docker compose run --rm gfootball python scripts/export_submission.py CAMINHO_DO_RUN --out submissions/e02-s1-v1 --team e02-s1-v1
docker compose run --rm gfootball python -m arena.check submissions/e02-s1-v1
docker compose run --rm gfootball python scripts/football_evaluate.py submissions/e02-s1-v1 --opponents random builtin --matches 20 --seed-start 100000 --out results/dev-e02-s1-v1
```

Repetir para cada run com outro `--out`, nunca reaproveitar pasta de resultados.
Também avaliar snapshots anteriores, usando `--opponents random builtin
submissions/old-v1`. O script usa a arena oficial com subprocessos, limites de
tempo e jogos completos. Escreve protocolo/hashes, partidas brutas e resumo.
Se cair no meio, preservar pasta parcial e repetir em nova pasta identificada;
não há retomada automática da avaliação. Protocolos completos têm `summary.csv`.

Depois da seleção no desenvolvimento, congelar o agente e rodar teste final:

```bash
docker compose run --rm gfootball python scripts/football_evaluate.py submissions/CANDIDATO --opponents random builtin submissions/old-v1 --matches 100 --seed-start 2000000 --out results/final-CANDIDATO
```

Usar `old-v1` apenas se tal snapshot existir. Sem versões anteriores, exportar
um baseline E01 válido como referência, com nome distinto. Manter as mesmas
versões para todos os candidatos comparados. Ver limites de interpretação dos
intervalos e regras de seleção em [EXPERIMENTOS.md](EXPERIMENTOS.md).

## 7. Figuras e pacote de retorno

O gráfico usa matplotlib, que não é dependência do agente. Instalar em um
ambiente de análise (Python/venv) separado ou em imagem de análise derivada,
preservando a imagem original para o checker:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/football_plot.py --runs CAMINHO_DO_RUN --evaluations results/dev-e02-s1-v1 --out report/figures
```

O script aceita vários caminhos. Identifica runs e não inventa curvas de treino;
as barras de erro são das partidas, não das seeds de treinamento. Não misturar
fases no eixo de passos sem contabilizar os pais. Para vídeos, escolher poucos
jogos diagnósticos (renderizar é lento), usando `python -m arena.match ...
--video DIRETORIO`; são exemplos qualitativos, não substitutos das partidas
medidas sem renderização. A CLI de vídeo da arena sorteia seeds novas: guardar
as seeds impressas; não chamar isso de replay exato da avaliação reservada.

Devolver ao orquestrador:

- código e commit usados, inventários CSV e diário de decisões;
- pastas completas de runs, snapshots de submissão e adversários, com hashes;
- `results/setup/`, logs e resultados brutos de desenvolvimento/teste;
- figuras, vídeos diagnósticos identificados e prompts reais de IA;
- falhas/limitações e o comando seguinte recomendado, se ficou algo pendente.

Além da cópia no disco de treino, fazer backup em outro dispositivo/local. Git
ignora os resultados grandes. Nenhuma conclusão depende só do texto gerado pela IA.

## 8. Checklist de entrega

Preencher `members` e nome final em `manifest.yml`; não são inferíveis da base.
Pasta final contém agente, pesos, conversão, probe e dependências locais
necessárias; sem importar módulos do checkout nem instalar pacotes em runtime.
Na imagem original CPU, rodar o checker depois de qualquer alteração da pasta.
Guardar `PASS`, tamanho (até 100 MB), tempos e hash final. Entregar só a pasta
solicitada, junto do PDF, conforme o canal definido pelo professor.

No PDF: até 6 páginas, duas colunas, quatro seções exigidas, fontes, figuras
legíveis, IDs de runs e declaração/anexo de IA. Revisão humana de conclusões e
conferência dos números com os arquivos brutos são parte da finalização.
