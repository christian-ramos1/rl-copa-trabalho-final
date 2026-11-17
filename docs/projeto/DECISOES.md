# Registro das escolhas de preparação e treino

Este documento reúne as escolhas feitas até 17/11/2026 e explicita sua origem.
As configurações YAML são a fonte dos valores executáveis; o código define os
detalhes que não são parâmetros de configuração. Uma escolha documentada não
é necessariamente uma escolha demonstrada como melhor por experimentos.

## Situação das decisões

- **Verificado:** integração de E00 com o motor e a arena. Evidências em
  [VERIFICACAO.md](VERIFICACAO.md) e no diário `experiments/decisions.csv`.
- **Escolha inicial:** configuração do baseline e desenho dos experimentos.
  Seus motivos abaixo são de compatibilidade, controle experimental e orçamento.
- **Pendente de dados:** orçamento definitivo, alterações que melhorem a política,
  execução de currículo/self-play e seleção do agente final. Não foram decididos
  nem apresentados como resultados positivos.

## Algoritmo, modelagem e arquitetura

| ID | Escolha | Motivo registrado e limite da evidência |
|---|---|---|
| D01 | PPO discreto como referência | A base oferece harness, testes, checkpoint e exportação para essa interface. Implementamos suas lacunas e validamos a execução. Não comparamos PPO com DQN/A2C/SAC; não há evidência de que PPO seja o melhor algoritmo para o torneio. |
| D02 | Um jogador controlado | É o caminho suportado pelo exportador genérico e pela pilha discreta. Permite validar o deploy desde o começo; não demonstra superioridade sobre controlar mais jogadores. Goleiro e demais jogadores permanecem com o motor. |
| D03 | `simple115v2`, 115 valores | Conversão de raw já disponível no GRF e reproduzida pelo exportador; probe confirmou igualdade de ações. Representação tem limitações de memória/observabilidade, descritas no plano. Não foram comparados SMM, RNN ou outra representação. |
| D04 | 19 ações padrão, sem máscara | Preservar a interface da arena e o espaço de ações da implementação base. Não foi testada máscara de ações. |
| D05 | Ator e crítico separados, duas camadas de 64 unidades Tanh cada | Arquitetura e valores padrão da rede da base. O smoke confirmou execução e inferência dentro do limite, não capacidade suficiente para bom futebol. |
| D06 | Treino estocástico; exportação/arena por argmax | O PPO coleta ações da distribuição categórica; o exportador existente produz a política determinística. Essa convenção deve ser mantida ao comparar candidatos. Não foi comparada submissão estocástica. A avaliação genérica ao fim do treino usa o padrão estocástico da base e não substitui a arena determinística. |
| D07 | Sem normalização adicional, frame stack ou memória | A rede discreta base não implementa normalização; manter observações inalteradas permite exportação genérica. Não é conclusão de que normalização ou memória prejudicam o aprendizado. |
| D08 | Scoring/builtin no treino direto inicial | Scoring mantém sinal alinhado aos gols; builtin fornece o adversário inicial já implementado. A utilidade desse sinal esparso será medida em E01/E02. |

## Hiperparâmetros: origem dos números

**Todos os valores da seção `ppo:` das três configs foram herdados dos defaults
de `PPOConfig` em `algorithms/ppo.py`. Não houve tuning de hiperparâmetros em
Football.** Mantê-los iguais permite uma primeira comparação que muda apenas a
recompensa. A documentação da base atribui esses defaults ao PPO discreto do
CleanRL; isso não os torna ótimos para `5_vs_5`.

| Parâmetro | Valor inicial | Papel no treino |
|---|---|---|
| `learning_rate` | 0,00025 | passo do Adam |
| `anneal_lr` | true | decaimento linear por iteração dentro de cada fase |
| `num_steps` | 128 | rollout por ambiente antes da atualização |
| `gamma` | 0,99 | desconto usado em GAE/alvos |
| `gae_lambda` | 0,95 | mistura temporal da estimativa de vantagem |
| `num_minibatches` | 4 | subdivisão do batch por época |
| `update_epochs` | 4 | passagens pelos dados de cada rollout |
| `norm_adv` | true | normalização das vantagens em cada minibatch |
| `clip_coef` | 0,2 | faixa do objetivo clipped e do value clipping |
| `clip_vloss` | true | clipping da atualização do crítico |
| `ent_coef` | 0,01 | peso do incentivo de entropia |
| `vf_coef` | 0,5 | peso da perda de valor |
| `max_grad_norm` | 0,5 | limite da norma de gradiente |
| `target_kl` | null | sem parada antecipada por limiar de KL |

Detalhes herdados do código, preservados: Adam com `eps=1e-5`, inicialização
ortogonal dos pesos (ganho √2 nas camadas ocultas, 0,01 na saída do ator e 1 na
saída do crítico), biases zero, GAE e objetivo clipped. O PPO corta bootstrap
em término e truncamento, e descarta transições de autoreset da atualização.
Essas convenções estão implementadas e testadas, mas não foram objeto de ablação.

## Orçamento, operação e fases

| ID | Escolha | Origem, critério e status |
|---|---|---|
| D09 | Piloto de 1.048.576 passos; quatro ambientes | Orçamento operacional inicial, múltiplo exato do batch 4×128=512: 2048 atualizações. Não foi escolhido por curva de convergência. Deve ser reavaliado após benchmark no WSL e aplicado igualmente às condições comparadas. |
| D10 | Checkpoint a cada 65.536 passos, reter três | Cadência operacional alinhada ao batch: 128 atualizações entre checkpoints nas configs iniciais. Limitar perda após interrupção e armazenamento; não é hiperparâmetro otimizado para qualidade. A base remove os checkpoints ao salvar o modelo final. |
| D11 | CPU primeiro; uma thread Torch no guia; vídeos desligados | Inferência do torneio exige CPU; medir antes de escolher CUDA para a pequena MLP. Evitar custo de renderização durante treino. CPU versus GPU ainda não foi comparado na máquina de destino. |
| D12 | Métricas locais, W&B desligado, salvar pesos | Curvas e proveniência precisam sobreviver sem conta externa; `save_model: true` é necessário para exportar. Não alteram o objetivo de aprendizagem. |
| D13 | Smoke com 4096 passos e dois ambientes | Reduzir custo da validação técnica: 16 atualizações de batch 256. Esses overrides são exclusivos de E00; não mudaram as configs de piloto de E01/E02. |
| D14 | Currículo opcional: 262.144 academy + 786.432 alvo | Proposta de divisão 25%/75% do piloto, mantendo total de 1.048.576. Não há evidência para essa proporção ou esse cenário como currículo ideal. Exige diagnóstico, diário e controle; ainda não executado. |
| D15 | Transferir pesos com novo Adam e nova agenda de LR | Suporte explícito a nova fase usando `model.pt`, sem fingir que ele contém estado do otimizador. A mecânica foi testada em CartPole. A ablação de currículo precisa de controle com o mesmo reinício para isolar o cenário. |
| D16 | Self-play contra snapshot congelado, opcional | Utiliza o carregamento de adversários que a base oferece e permite identificar exatamente o oponente. Exige hash, controle com mesmo pai e avaliação externa ao adversário de treino. Ainda não executado; nenhum snapshot competitivo foi escolhido. |

O cronograma de duas semanas e a reserva aproximada de 25% do tempo para
avaliação/relatório são escolhas de gestão, não resultados científicos. Orçamento
final, número de workers e benefício de GPU dependem das medições na máquina do
colega. Não extrapolar os aproximadamente 563 passos/s do final do smoke ARM64
para a campanha em outro computador nem confundir esse número com throughput
incluindo avaliações, build e exportação.

## Comparação, avaliação e seleção

| ID | Escolha | Motivo e limite |
|---|---|---|
| D17 | E01 scoring versus E02 scoring+checkpoints | Uma primeira ablação de recompensa com demais fatores iguais. A possibilidade de ajudar exploração é hipótese. Não executar ambos e depois alegar que a mudança foi motivada por um comportamento nunca registrado. |
| D18 | Três seeds de treino: 1, 2, 3 | Três repetições iniciais sob orçamento de duas semanas; os números identificam execuções, não foram escolhidos por desempenho. Variabilidade entre seeds continua incerta. |
| D19 | 20 partidas de desenvolvimento; ampliar para 100 nos candidatos | Triagem com custo limitado seguida de confirmação. Contagens são orçamento pré-planejado, não cálculo de poder estatístico. Usar protocolos iguais entre comparações. |
| D20 | Separar seeds de desenvolvimento e teste final | Reduzir seleção pelo próprio teste. Faixas 100000… no desenvolvimento e 2000000… no final são rótulos de partição; não são sementes privilegiadas. E00 usou 900000/900001 nos quatro jogos extras. |
| D21 | Jogos completos, lados alternados, adversários fixos | Aproximar condições da arena e evitar comparar políticas com adversários ou horizontes diferentes. O checker oficial e seu comportamento de fim de episódio foram preservados. |
| D22 | Selecionar por pontuação contra builtin no desenvolvimento | Média de `vitória + 0,5×empate` entre seeds de treino; verificar regressões contra random/snapshots. Regra de seleção planejada, sem alegação de que represente todos os futuros adversários do campeonato. |
| D23 | W/D/L, saldo, forfeits e Wilson para taxa de vitória | Preservar o resultado completo e explicitar incerteza das partidas. Wilson não mede dispersão entre treinamentos nem prova diferença significativa entre dois métodos. Não excluir derrotas por falha. |

Os detalhes, controles e limitações estão em [EXPERIMENTOS.md](EXPERIMENTOS.md).
Critérios numéricos de promoção por habilidade, limiares para recompensas novas,
arquitetura alternativa e seleção de adversários de uma liga **ainda não foram
definidos**. Se essas extensões forem propostas, registrar critérios antes dos
respectivos runs. O plano atual usa tetos de passos para as fases opcionais.

## Decisão após o smoke: evidência disponível

Entrada `D-E00-001` no [diário](../../experiments/decisions.csv): o checker deu
PASS, reproduziu 600/600 ações e registrou ação 8 em 100% das ações nos dois jogos.
Decidimos preservar a evidência e seguir para benchmark e experimentos controlados,
sem alterar a política ou o checker para apagar o aviso.

Essa observação não identifica a causa da política constante. Recompensa esparsa,
exploração, horizonte ou outros fatores são hipóteses para investigação; não
foram diagnosticados pelo smoke. Não há decisão baseada em desempenho que
justifique trocar arquitetura, aumentar gamma ou ativar self-play neste momento.

O registro de uma nova decisão deve informar: observação e run, arquivo de
evidência, hipótese, controle, fator alterado, orçamento/seeds, critério prévio e
resultado posterior. Manter decisões negativas e propostas rejeitadas no diário.

## D24: protocolo da candidata L40S (decisão de infraestrutura pendente)

O usuário apresentou outra instância Ubuntu/KVM com L40S; os recursos visíveis
e o `/dev/shm` parcialmente ocupado motivaram verificar limites reais antes da
troca. [VALIDAR_L40S.md](VALIDAR_L40S.md) define diagnóstico, smoke e seis runs
CPU com batch 512: 4×128 versus 8×64, 65.536 passos/run, aquecimento de 8192,
seeds 1/2/3, ordem A/B, B/A, A/B. São escolhas operacionais prévias, sem tuning
de qualidade. O limiar de ganho de 10% é margem prática, não significância
estatística. Variar rollout pode afetar GAE e precisa ser mantido igual entre
E01/E02 depois de selecionar a infraestrutura.

Nenhum comando foi executado remotamente na L40S nesta preparação; a decisão de
máquina depende dos relatórios coletados e da disponibilidade/custo da instância.
