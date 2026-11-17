# Verificação da preparação — 17/11/2026

## Ferramentas para a candidata L40S

Preparados `football_machine.py`, `football_benchmark.py`, o modo `local` do
smoke e um overlay opcional de shared memory. O roteiro remoto está em
[VALIDAR_L40S.md](VALIDAR_L40S.md) e o texto de passagem em
[PROMPT_IA_L40S.txt](PROMPT_IA_L40S.txt).

Validação **local**, na imagem CPU/ARM64 deste Mac:

- 33 testes de diagnóstico, métricas e PPO passaram na rodada focada;
- ambos os smokes completos, modo local e Docker com overlay, tiveram
  **146 testes aprovados e um skip CUDA**, checker **PASS**, probe 600/600 e
  seis jogos completos sem forfeits por execução;
- coletor executado no container: confirmou CPU Torch e 1.073.741.824 bytes de
  `/dev/shm` no overlay;
- benchmark exercitado com 4×128 e 8×64, **somente 1024 passos por condição**,
  warmup 512 e uma repetição. Produziu protocolo, logs, snapshots e resumos;
  classificou corretamente como `insufficient_repetitions`;
- sintaxe Bash/Python, configuração Compose e links locais verificados.

Evidências: [ferramentas de infraestrutura](../../results/infra-tools-validation-20261117/),
[smoke local](../../results/smoke-local-mode-20261117/),
[smoke Docker com overlay](../../results/smoke-docker-overlay-20261117/).
O benchmark curto foi apenas teste funcional, não um resultado de desempenho
para selecionar workers. A parte CUDA não foi exercitada com GPU NVIDIA aqui.
**Não foram executados comandos na L40S**, nem o protocolo completo de três
pares de 65.536 passos, nem decidido trocar a máquina.

## Integração com Football: PASS

A cadeia completa foi executada em 17/11/2026, incluindo o motor real, treino,
exportação, checker oficial e avaliador. O agente gerado é um **smoke técnico**,
não o candidato final do campeonato.

| Item | Resultado observado |
|---|---|
| Imagem | `t-zero-gfootball:latest`, Linux ARM64, CPU |
| Python / Torch | 3.10.12 / 2.9.1+cpu, uma thread Torch |
| Football | 2.10.3, commit `3d9e754720a95621bba6475c4d3b0d56fe919014` |
| Testes focados com motor | **140 passaram**, um skip de CUDA |
| Treino | PPO, `5_vs_5`, scoring, um jogador, dois ambientes, **4096 passos** |
| Exportação | TorchScript + manifest + config + probe de 600 estados |
| Checker oficial | **PASS**, sem alterar nenhum arquivo de `arena/` |
| Probe | **600/600 ações**, 100% reproduzidas |
| Carga | 0,68 s (limite 30 s) |
| Inferência no checker | média **0,13 ms**, máximo 18,5 ms |
| Pasta | aproximadamente **0,7 MB** (limite 100 MB) |
| Jogos do checker | dois completos, contra random e builtin, sem forfeit |
| Avaliador do projeto | quatro jogos completos, dois por adversário, lados alternados, sem forfeit |

O checker emitiu **WARN de comportamento**: a ação 8 apareceu em todas as ações
dos seus dois jogos. O probe de 100% confirma que a exportação reproduz a
política de treino; não há evidência de troca de pesos ou conversão divergente.
O treino curto não produziu uma política competitiva. Não usar E00 como baseline
treinado nem como justificativa para declarar superioridade de um método.

O cenário tem duração nominal de 3000 passos; nesta versão o motor encerrou
naturalmente as partidas após 3001 chamadas a `step`. Os jogos foram rodados
**sem `--max-steps`**, preservando a condição de fim do ambiente e as regras.

Reproduzir após construir a imagem:

```bash
bash scripts/football_smoke.sh OUTRO_IDENTIFICADOR_UNICO
```

Evidências locais (pastas de resultados são ignoradas pelo Git; incluir no backup):

- [build e execução](../../results/integration-20261117/);
- [relatório JSON do checker](../../results/smoke-20261117-validation/check.json),
  [log do checker](../../results/smoke-20261117-validation/check.log);
- [testes com motor](../../results/smoke-20261117-validation/tests.log);
- [protocolo, partidas e resumo do avaliador](../../results/smoke-20261117-validation/evaluation/);
- [versão do motor](../../results/smoke-20261117-validation/football-commit.txt),
  [pacotes](../../results/smoke-20261117-validation/packages.txt),
  [hashes dos arquivos oficiais inalterados](../../results/smoke-20261117-validation/arena-source-check.json);
- [agente exportado](../../submissions/smoke-20261117-validation/).

Run: `runs/ppo_football_sparse/GFootball-5_vs_5-v0__a1ff80e299ee70c4__1__20261117_130047`.
SHA256 de `model.pt`:
`4ee54cde634f780f86683f8eccd467a6353d4b6312c89505f2557505ecb35679`.
ID da imagem:
`sha256:37867a54d603dc1b3fa4ca159374649b5788000ec80b4614a1bf4402afaafdea`.

Os avisos de manutenção do Gym legado foram mantidos nos logs e não causaram
falha nesta execução. O adaptador Gymnasium do repositório foi exercitado pelos
testes. Não foram relaxados timeouts, tamanho de pasta, probe ou limites de ações.

Antes dos treinos longos, repetir esse comando no Windows/WSL do colega
(arquitetura x86_64), medir SPS/memória e fixar orçamento. Esta verificação ARM64
não é benchmark nem garantia de tempo no outro hardware. Ainda não foram
executadas as campanhas E01/E02, currículo ou self-play.

## Verificação inicial sem motor no host

Ambiente local: macOS ARM64, Python 3.11.17 em venv temporário, PyTorch 2.9.1,
NumPy 2.2.6, Gymnasium 1.2.0, pytest 8.4.1. Não foi alterado o Python principal.
O hardware do colega e WSL/Docker precisam de sua própria validação.

Comando da rodada principal:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_ppo.py tests/test_football_project.py \
  tests/test_configs_load.py tests/test_config_roundtrip.py \
  tests/test_wrappers.py tests/test_checkpoint.py tests/test_arena.py \
  tests/test_gfootball.py tests/test_agent_network.py \
  tests/test_config_reference.py -q -rs
```

Resultado: **86 testes passaram, 3 skips, sem warnings**. Os skips foram os
módulos de arena/Football (motor `gfootball` ausente) e um teste de RNG CUDA.
Separadamente, `python -m pytest tests/test_a2c.py -k agent -q` passou os cinco
testes da política categórica (nove testes de outras partes foram excluídos).
Não foi executada nem declarada aprovada a suíte inteira dos outros algoritmos.

Verificado:

- GAE, objetivo clipped, gradientes e diagnósticos de PPO nos testes da base;
- política categórica, ações determinísticas e equivalência de ações após
  TorchScript em um lote de 32 observações vetoriais de teste;
- configs novas válidas e E01/E02 diferentes apenas em recompensa;
- treino curto real em CartPole, interrupção simulada após atualizações,
  retomada com checkpoint e nova fase com pesos do pai;
- registros locais, hashes de origem e orçamento resolvido de cada sessão;
- sumarização de placares nos dois lados, empates, derrotas por falha e IC;
- gravação de agenda/seeds/partidas e proteção contra sobrescrita do avaliador,
  usando partidas substitutas no teste unitário, sem engine;
- remoção da cauda de métricas abandonada ao retomar um checkpoint;
- compilação dos arquivos Python alterados, links locais da documentação e
  `git diff --check`.

O portão que faltava nessa primeira rodada foi concluído na integração acima.
Nenhum treino longo de futebol nem PDF de resultados foi produzido. E00 está
registrado nos inventários como validação, separado das campanhas comparativas.
Os comandos completos estão em [EXECUCAO.md](EXECUCAO.md).
