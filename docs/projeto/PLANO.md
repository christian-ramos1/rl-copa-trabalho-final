# Plano do trabalho final

## Comece aqui

Este repositório prepara treino, avaliação, evidências e submissão. O objetivo é
produzir um agente válido e um argumento experimental reproduzível; a força do
agente só será conhecida após os runs comparativos. O smoke E00 com o motor
Football, exportação e checker foi validado; suas evidências são técnicas e
ficam separadas dos experimentos de desempenho.

Leitura para quem vai executar, inclusive uma nova IA:

1. Este plano e `AGENTS.md`: objetivos, decisões e limites.
2. [EXECUCAO.md](EXECUCAO.md): instalação, smoke, comandos e recuperação.
3. [EXPERIMENTOS.md](EXPERIMENTOS.md): comparações, métricas e seleção.
4. [../../report/roteiro.md](../../report/roteiro.md): material para o PDF.

As escolhas iniciais, a origem dos hiperparâmetros e o que permanece pendente
estão reunidos em [DECISOES.md](DECISOES.md).
Para a candidata L40S, usar [VALIDAR_L40S.md](VALIDAR_L40S.md): diagnóstico do
container, smoke e comparação de throughput antes de decidir a troca.

O estado dos testes e o portão de integração concluído estão em
[VERIFICACAO.md](VERIFICACAO.md).

## O que as instruções exigem

| Entrega | Evidência necessária | Critério de conclusão |
|---|---|---|
| MDP | observação, ações, recompensa, término, desconto, jogadores | documentar a modelagem realmente treinada |
| Pipeline | algoritmo, rede, fases, adversários, orçamento, transições | comandos e arquivos permitem repetir o procedimento |
| Experimentos | baselines, versões anteriores, hipóteses e mudanças | curvas e partidas brutas identificadas por run |
| Discussão | interpretar êxitos, falhas, limites e próximos testes | afirmações sustentadas pelas evidências |
| Código | pasta com `Agent`, manifest, pesos e probe | `python -m arena.check PASTA` retorna PASS |
| PDF | máximo 6 páginas em duas colunas | inspeção do PDF final, declaração de IA e prompts |

O enunciado fornecido no chat acrescenta limite de páginas, colunas e declaração
de IA que não aparecem integralmente no arquivo original da disciplina. Vale
o enunciado do usuário. Não pressupor que o anexo está fora das 6 páginas:
planejar inicialmente tudo dentro do limite, ou confirmar a regra com o professor
antes de gerar a versão final. Não omitir prompts para caber.

## Auditoria da base e decisões iniciais

Base inspecionada: commit `6729aa1`. `arena/rules.py` e o checker são a referência
executável da interface; não serão adaptados para favorecer nosso agente.

| Constatação no código | Consequência |
|---|---|
| PPO discreto e rede tinham funções incompletas | implementadas as três funções de PPO e a política categórica; validar testes antes de treinar |
| exportador suporta um jogador, `simple115v2`, observação inalterada | começar com essa combinação para testar a cadeia até a arena desde o primeiro dia |
| PPO padrão só envia curvas ao W&B | runner do projeto grava `metrics.jsonl`, mesmo sem conta externa |
| não havia entrada para transferir pesos entre fases | runner aceita `--init-from`; reinício do Adam e da agenda é explícito |
| finalização remove checkpoints de retomada | retomar interrupção com `--resume`; usar pesos finais em nova fase com `--init-from` |
| engine não vem instalado no Python do host | imagem CPU construída e smoke completo validado; repetir no WSL antes da campanha |
| imagem busca `GFOOTBALL_REF=master` e há dependências sem pin | registrar commit do motor, `pip freeze` e ID da imagem; reutilizar a mesma imagem na campanha |

Não confundir checkpoint de treino (`checkpoint_gs*.pt`) com a recompensa
`scoring,checkpoints`: esta acrescenta shaping por avanço com a bola.

## Modelagem de partida

Baseline proposto: PPO categórico, um jogador de linha controlado, observação
`simple115v2` de 115 valores, 19 ações e duas MLPs separadas com duas camadas
de 64 unidades Tanh (ator e crítico). Desconto inicial 0,99, GAE 0,95, demais
valores em `configs/ppo_football_sparse.yml`.

A escolha inicial tem motivo de engenharia observado: aproveita as interfaces
e o exportador existentes e cabe em inferência pequena. Não é uma alegação de
superioridade esportiva. O goleiro e os jogadores não controlados ficam com a IA
do motor. O índice do jogador ativo pode mudar; a política recebe a representação
do jogador atualmente controlado.

O estado completo do simulador define a dinâmica; o vetor é uma **observação**,
não prova de estado Markoviano completo. `simple115v2` não contém toda a memória
de ações persistentes nem toda a informação do relógio. Descrever essa limitação
e a política sem memória no relatório, em vez de afirmar observabilidade completa.
No jogo de horizonte finito, o placar final é a tarefa; o vetor simplificado é
uma aproximação usada pelo agente. O shim distingue `terminated` de `truncated`,
mas o PPO base corta bootstrap em ambos. Essa convenção será mantida e declarada
nas comparações iniciais. O horizonte de 3000 define o fim da partida-alvo;
cenários academy têm seus próprios critérios de fim.

Recompensa de referência: `scoring` (gols pró/contra). A primeira ablação troca
somente a recompensa por `scoring,checkpoints`. A avaliação da arena sempre usa
placares de jogos completos; retorno com shaping não é taxa de vitória.

## Duas semanas, com orçamento ajustável

Máquina informada: Windows 11, Core Ultra 9, RTX 5070, RAM 32 GB / VRAM 8 GB.
Não estimar milhões de passos por dia a partir desses nomes: medir execução real.

| Janela relativa | Trabalho | Portão para avançar |
|---|---|---|
| Dias 1–2 | WSL/Docker, testes, smoke até checker, benchmark | exportação válida, arquivos íntegros, SPS e custo de partidas medidos |
| Dias 3–5 | E01 vs E02, seeds 1/2/3, orçamento igual | curvas + avaliação de desenvolvimento + diário |
| Dias 6–8 | E03 currículo, apenas se o diagnóstico justificar | controle com custo total equivalente e avaliação em `5_vs_5` |
| Dias 9–10 | E04 adversário antigo, se houver snapshot válido e tempo | controle contra builtin com mesma origem e orçamento |
| Dia 11 | repetir comparação mais informativa, escolher candidato | protocolo de seleção aplicado sem olhar teste final |
| Dias 12–13 | congelar candidato, avaliação final, checker CPU | resultados preservados, nenhuma violação ou timeout |
| Dia 14 | PDF, anexo de IA, empacotamento e revisão | até 6 páginas/duas colunas e submissão conferida |

As janelas são limites de gestão, não justificativas científicas. Se só couberem
E01/E02 bem avaliados, entregar esses experimentos e explicar os testes seguintes
como trabalho futuro. Não disparar extensões mal medidas para preencher seções.

Começar com piloto de 1.048.576 passos por condição. Medir SPS no início e após
alguns minutos; o custo inclui atualizações PPO. Para um run:
`horas ≈ passos / SPS / 3600`, acrescido de exportação e partidas. Reservar
aproximadamente 25% do tempo disponível para avaliação/relatório e uma margem
para falhas. O orçamento definitivo é registrado antes das comparações; pilotos
podem motivar novo orçamento, aplicado igualmente aos controles.

`num_envs=4` é ponto de partida. O vectorizador já usa subprocessos. Primeiro
comparar velocidade com o mesmo batch total (por exemplo 4×128 vs 8×64), sem
interpretar benchmark como comparação de qualidade. Evitar vários runs
simultâneos antes de medir RAM, temperatura e SPS; 32 GB também sustentam WSL e
Docker. CPU é o primeiro benchmark; CUDA só passa a padrão se o tempo medido
melhorar sem tornar a configuração incompatível com inferência CPU.

## Divisão do trabalho e artefatos

O orquestrador mantém código, protocolo, testes, documentação e revisão das
evidências. O colega e a IA dele executam na máquina de treino, registram falhas,
congelam snapshots e devolvem os dados. Decisões dependentes de comportamento
voltam acompanhadas de arquivo, run e métrica, não apenas de uma conclusão verbal.

Manter código/configs/documentação no Git. `runs/`, `results/`, pesos e vídeos
precisam de cópia externa identificada por hash: a cópia do repositório por si só
não inclui esses artefatos. Nunca copiar `.env` com chaves de API entre equipes.

Referências técnicas: [artigo do PPO](https://arxiv.org/abs/1707.06347),
[observações oficiais do GRF](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md).
Essas referências descrevem métodos; não são resultados deste trabalho.
