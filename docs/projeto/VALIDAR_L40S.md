# Validar a candidata L40S antes de mudar a campanha

Executar em **Bash, na raiz do repositório**, na máquina candidata. Não há acesso
remoto a ela nesta preparação. Os números do anexo são uma fotografia, não uma
reserva de recursos: KVM com 12 CPUs visíveis, 62 GiB de RAM/21 GiB disponíveis,
L40S com aproximadamente 30 GiB livres, `/dev/shm` de 64 MiB/12 MiB livres.
A alternativa ao Windows ainda **não foi escolhida**.

## 1. Diagnóstico do ambiente em que vocês têm acesso

```bash
infra_tag="$(date -u +%Y%m%dT%H%M%SZ)"
infra_dir="results/l40s-$infra_tag"
mkdir -p "$infra_dir"
python3 scripts/football_machine.py --out "$infra_dir/current-runtime.json"
```

O coletor usa a biblioteca padrão e testa Torch em subprocesso: mesmo um erro de
importação ou instrução ilegal fica registrado. Se o Python correto estiver em
um venv já existente, usar esse interpretador no lugar de `python3`.
O diagnóstico não instala pacotes, altera quotas, enumera variáveis secretas,
reinicia serviços ou libera memória de outros processos.

Examinar `current-runtime.json`:

- `resources.affinity` e `limits.cpu_quota_cores_min`: CPUs permitidas ao processo
  versus quota de tempo de CPU. O menor limite visível é mais útil que apenas
  `lscpu`. Os ancestrais visíveis também são consultados (cgroups v1 e v2).
- `limits.memory_limit_bytes_min` e `memory_headroom_bytes_min`: teto/headroom dos
  cgroups visíveis. `meminfo` é adicional e pode refletir memória compartilhada.
- `resources.cgroups`: dados brutos de quota, throttling, pressão e eventos de
  memória. `null` não significa recurso ilimitado: pode significar indisponível.
- `shm_bytes`: total/usado/livre de memória compartilhada. Não apagar objetos de
  `/dev/shm` de terceiros para dar espaço ao teste.
- `gpu`: nome, VRAM ocupada e utilização instantânea; não pressupor exclusividade
  porque uma amostra mostra 0% ou o namespace não lista processos.
- `torch.returncode`: deve ser zero no interpretador que executará o trabalho.
  O probe faz uma operação CPU com backward e uma pequena operação CUDA se ela
  estiver disponível. `cuda_test`/`cuda_error` registram separadamente uma falha
  de GPU: ela não transforma sucesso de CPU em falha de CPU. Isso não substitui
  o smoke completo do Football.
- `docker`: separa presença do cliente de acesso efetivo ao daemon.

As quotas ocultas acima do namespace e a disputa do host não são demonstradas
ausentes por esse relatório. Confirmar com o responsável pela instância: acesso
durante as duas semanas, recursos compartilhados/dedicados, persistência de
`/docker_vol`, custo e política de desligamento. Não parar a instância para
testar persistência; conferir o volume no provedor e fazer backup dos artefatos.

## 2A. Se há acesso ao Docker e controle de um container próprio

```bash
test -f .env || cp .env.example .env
export UID
export GID="$(id -g)"
export COMPOSE_FILE="docker-compose.yml:docker/compose.football-validation.yml"
docker info --format '{{.ServerVersion}} {{.OSType}}/{{.Architecture}}'
docker compose version
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose build gfootball
docker image inspect t-zero-gfootball:latest --format '{{.Id}}' > "$infra_dir/image-id.txt"
docker compose run --rm -T gfootball python -c 'from pathlib import Path; assert Path("scripts/football_machine.py").is_file(), "O daemon nao montou este checkout"'
docker compose run --rm -T gfootball python scripts/football_machine.py --out "$infra_dir/training-runtime.json"
```

O overlay configura **1 GiB de `/dev/shm` para os novos containers próprios**,
igual nas duas condições. Não modifica o container compartilhado já em uso.
É uma margem operacional proposta, não consumo medido nem requisito universal.
Sem permissão para isso, usar os limites existentes, registrá-los e avaliar as
falhas. Não remountar `/dev/shm` nem mudar cgroups do host.

Se o daemon for externo ao container atual, o bind mount precisa apontar para o
mesmo checkout no host do daemon. O assert acima detecta montagem errada. Resolver
o caminho com o responsável ou seguir 2B; não inicializar Docker privilegiado
dentro de outro container para contornar o acesso.

É esperado que a **imagem CPU** reporte CUDA indisponível, mesmo em um host L40S.
O diagnóstico do ambiente original pode verificar CUDA se Torch já a suportar.
Não construir uma imagem CUDA nem mudar drivers para esta primeira comparação.

## 2B. Se só há terminal no container atual

Usar ambiente que já tenha as dependências do projeto e o motor compilado.
Não executar `pip install --upgrade` no Python compartilhado. Se faltarem pacotes
ou o engine, devolver o erro com proposta de imagem/venv isolado; os comandos de
build existentes estão no Dockerfile, mas dependências nativas podem exigir
permissões que essa instância não oferece.

```bash
export FOOTBALL_PYTHON="$(command -v python3)"
"$FOOTBALL_PYTHON" scripts/football_machine.py --out "$infra_dir/training-runtime.json"
"$FOOTBALL_PYTHON" -c 'import torch, gymnasium, gfootball, gfootball_engine, pytest; print(torch.__version__); print(gfootball_engine.__file__)'
```

Se estiver em outro venv, ajustar `FOOTBALL_PYTHON` para o caminho absoluto dele.
O campo `compiled_cuda` é a versão usada pelo Torch, não a versão exibida pelo
driver em `nvidia-smi`. Falta de CUDA não bloqueia nosso benchmark CPU.

## 3. Smoke completo, antes do benchmark

Escolher **um** comando conforme o caminho acima:

```bash
# 2A: imagem CPU, com o mesmo COMPOSE_FILE exportado
bash scripts/football_smoke.sh "l40s-smoke-$infra_tag"

# 2B: Python do ambiente atual; o script oculta CUDA só dos seus subprocessos
bash scripts/football_smoke.sh "l40s-smoke-$infra_tag" local
```

Não executar ambos com o mesmo ID. O script exige pasta nova, testa o motor,
roda os testes focados, treina 4096 passos, exporta, verifica o probe, executa o
checker oficial em CPU com jogos completos e mais quatro jogos do avaliador.
Não relaxar limites nem suprimir o aviso de política constante de um smoke.

Em modo `local`, o código oficial do checker é executado, mas **isso não certifica
a imagem oficial**: pacotes e bibliotecas do container podem ser diferentes.
Registrar essa limitação e repetir na imagem CPU da disciplina antes da entrega.
Se o engine source não estiver em `/opt/football`, indicar seu checkout com
`GFOOTBALL_SOURCE=/caminho/football`; se desconhecido, registrar a lacuna.

O portão é: imports reais funcionando, testes de Football/arena sem skip por
ausência do motor, checker `PASS`, probe reproduzido e nenhum forfeit por erro.
Se falhar, parar antes dos seis runs e devolver log/erro. Não inferir desempenho
do agente a partir da qualidade esportiva de E00.

## 4. Benchmark controlado de CPU: seis runs sequenciais

No mesmo Bash e mesmo ambiente da etapa anterior, escolher **um** comando:

```bash
# 2A
set -o pipefail
docker compose run --rm -T -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 gfootball \
  python scripts/football_benchmark.py --out "$infra_dir/benchmark" \
  --steps 65536 --warmup 8192 --repeats 3 \
  2>&1 | tee "$infra_dir/benchmark-console.log"

# 2B
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$FOOTBALL_PYTHON" \
  scripts/football_benchmark.py --out "$infra_dir/benchmark" \
  --steps 65536 --warmup 8192 --repeats 3 \
  2>&1 | tee "$infra_dir/benchmark-console.log"
```

Protocolo implementado:

| Fator | Condições |
|---|---|
| Cenário/recompensa/adversário | mesmos `5_vs_5`, scoring, builtin |
| Rede, PPO, inferência do treino | mesmos parâmetros, CPU, uma thread Torch |
| Configuração A | 4 ambientes × 128 passos = batch 512 |
| Configuração B | 8 ambientes × 64 passos = batch 512 |
| Passos globais por run | 65.536 = 128 iterações PPO |
| Aquecimento excluído da taxa | primeiros 8192 passos = 16 iterações |
| Repetições | seeds 1/2/3, pareadas entre A e B |
| Ordem | A/B, B/A, A/B, sem runs concorrentes |
| Vídeo/checkpoints periódicos | desligados igualmente; modelo final é salvo |

O runner normal ainda realiza avaliação genérica ao final. A coluna
`training_sps` usa timestamps das métricas, excluindo inicialização, aquecimento
e essa avaliação. `end_to_end_seconds` inclui todo o subprocesso. Não dividir os
passos de treino pelo tempo que inclui jogos de avaliação e chamar isso de SPS
do PPO. Orçamento total deste benchmark: **393.216 passos de treino**, mais
avaliações; esses runs são infraestrutura, não E01/E02 para o relatório.

O programa grava antes/depois das quotas, deltas de throttling/OOM e amostras a
cada 500 ms do RSS da árvore de processos e uso dos cgroups. RSS pode contar
páginas compartilhadas mais de uma vez; cgroups podem incluir outros usuários.
Esses máximos amostrados não são medições exatas do pico físico exclusivo do run.
O benchmark para na primeira falha e preserva resultados parciais.

## 5. O que devolver e como decidir

Devolver **inteiras** as pastas `results/l40s-.../`,
`results/l40s-smoke-.../`, a submissão do smoke e os caminhos dos runs listados
em `summary.csv`. O benchmark produz:

- `protocol.json`, `machine-before.json`, `machine-after.json`;
- `summary.csv`, `runs.jsonl`, `comparison.json`;
- log e snapshots de recursos por condição/repetição.

Não selecionar pelos retornos desses runs. Nossa regra operacional prévia é
considerar oito ambientes se a mediana de SPS crescer **pelo menos 10%**, vencer
a maioria dos pares (ao menos 2 de 3), sem falhas/OOM e com recursos estáveis.
O limiar de 10% é margem prática, não teste estatístico. Throttling elevado,
carga variável, mudança de quota ou um resultado isolado exigem revisão/repetição.
Caso inconclusivo, manter quatro ambientes até investigar.

Batch igual **não torna as trajetórias idênticas**: reduzir rollout por ambiente
altera o contexto de GAE/bootstrapping. A comparação mede velocidade operacional,
não prova mesma qualidade de aprendizado. Se escolher 8×64 para a campanha,
aplicar essa escolha a E01 e E02 e registrá-la antes de comparar recompensas.

Para decidir **trocar de máquina**, repetir o mesmo protocolo no Core Ultra 9
(WSL), com mesmo código, motor, pacotes, orçamento e threads. Comparar SPS e
custo total, estabilidade, horas realmente disponíveis e custo da instância.
Somente medir a L40S não prova que ela seja mais rápida que a máquina anterior.
Registrar a conclusão em `experiments/decisions.csv`; ainda não há resultados
da candidata L40S nesta preparação.

Referências para interpretar limites:
[Docker/cgroups](https://docs.docker.com/engine/containers/resource_constraints/),
[Docker e shared memory](https://docs.docker.com/engine/containers/run/),
[hierarquia cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html).
