# Protocolo experimental e decisões

## Matriz mínima e extensões condicionais

| ID | Hipótese a testar | Comparação e controle | Config / execução |
|---|---|---|---|
| E00 | o pipeline preserva a política no deploy | smoke, exportação, probe e checker | concluído: 600/600 no probe e PASS; ver `VERIFICACAO.md`; não é evidência de aprendizado |
| E01 | PPO padrão estabelece uma referência | treino direto `5_vs_5`, scoring, builtin | `ppo_football_sparse` |
| E02 | shaping melhora a aprendizagem sob mesmo orçamento | E01; só mudar recompensas | `ppo_football_checkpoints` |
| E03 | treino prévio de finalização ajuda no alvo | treino direto com mesmo total de passos e controle de reinício | `ppo_football_academy` → alvo com `--init-from` |
| E04 | adversário congelado reduz fraqueza contra agentes aprendidos | mesma origem e nova fase de mesmo tamanho contra builtin | config vencedora + `env_kwargs.opponent=SUBMISSAO_ANTIGA` |

E02 está pré-planejado por requisito de comparar além do PPO padrão; só afirmar
que a mudança foi motivada por um comportamento se ele realmente foi observado.
E03/E04 só iniciam após preencher a entrada correspondente em
`experiments/decisions.csv`. Não é obrigatório executá-los. Vários jogadores,
RNN, SMM ou recompensa própria exigem nova implementação e novo teste de export;
não estão habilitados neste pipeline inicial.

Para E03, pré-treino de 262.144 + alvo de 786.432 somam 1.048.576. O controle
direto deve gastar o mesmo total. Como `--init-from` reinicia Adam/LR, uma ablação
que queira atribuir ganho **somente ao cenário** precisa também de um controle
direto em duas fases de 262.144 + 786.432, com o mesmo reinício, recompensa e
hiperparâmetros. Sem esse controle, descrever a comparação como pipeline de
currículo incluindo reinício, reconhecendo o fator de confusão.

Não usar desempenho no academy como prova de força em `5_vs_5`. Definir antes
do run o teto de passos de cada fase. No piloto, avaliar a fase ao atingir o
teto; se não houver evidência de aquisição da habilidade ou transferência,
registrar resultado negativo. Se testar promoção por limiar, especificar a
métrica, limiar, quantidade de episódios e frequência **antes** desse novo
experimento; não escolher o limiar depois de ver o teste final.

E04 é uma fase de treino contra snapshot congelado, uma forma simples de
self-play histórico. Não é uma liga dinâmica nem um adversário atualizado a
cada passo. A versão do adversário deve ser imutável e ter hash. Um modelo
treinado contra um único snapshot pode sobreajustar: avaliar também contra
builtin, random e outras versões. A fase-controle contra builtin parte do
mesmo modelo e reinicia o otimizador da mesma maneira. Arquivar todos os pais.

## Sementes, partidas e seleção

- Treino: seeds 1, 2 e 3, idênticas entre condições. Explorações maiores devem
  acrescentar seeds, não escolher a seed com melhor teste.
- Desenvolvimento: seeds de partidas 100000–100019 inicialmente (20 por
  adversário), alternando lados. Para confirmar candidatos, ampliar a faixa
  reservada de desenvolvimento, por exemplo até 100099 (100 partidas).
- Teste final: faixa 2000000–2000099, 100 partidas por adversário, somente depois
  de escolher e congelar o agente. Não sobrepor com desenvolvimento.
- É uma agenda de avaliação reproduzível, não garantia de simulação idêntica
  entre máquinas. Identificar imagem, versão do motor e pacotes.
- Usar mesmos adversários, seeds e quantidade de partidas ao comparar políticas.
  Jogos pares e ímpares têm lados opostos. Não cortar partidas com `max_steps`.

`random` da arena controla um jogador com ações aleatórias; os demais continuam
com o motor. `builtin` controla zero jogadores externos: toda a equipe usa a
IA do motor. Não descrever `random` como cinco jogadores aleatórios.

Definir antes da campanha: seleção por média entre seeds de treino da pontuação
contra builtin (`win + 0.5 draw`), com taxa de vitória e saldo como diagnósticos.
Usar random e snapshots anteriores para detectar regressões. Se os resultados
forem inconclusivos, ampliar desenvolvimento ou preferir o pipeline mais simples;
não declarar vitória estatística a partir de poucas partidas. Para selecionar
uma seed final, aplicar a mesma regra no desenvolvimento e declarar essa seleção
no relatório. O teste final não altera pesos, seed escolhida ou regras.

O script gera IC de Wilson de 95% para a **taxa de vitória** por política e
adversário. Empate conta como ausência de vitória, e aparece também separado.
Esse intervalo não cobre variação entre treinamentos nem é um teste de
significância de duas políticas. Relatar média e dispersão entre seeds à parte;
com três seeds, explicitar a incerteza. Não tratar 300 partidas de três políticas
como 300 treinamentos independentes.

## O que guardar

| Evidência | Arquivo / registro | Uso |
|---|---|---|
| passos e hiperparâmetros | `config.yml`, `completed.json` | contabilizar custo efetivo (rollouts inteiros) |
| origem e código | `session_*.json`, `source_*.patch`, commit | identificar implementação, retomadas e pais |
| ambiente de execução | `packages_*.txt`, ID da imagem e hardware | repetir configuração |
| curva e estabilidade | `metrics.jsonl` | retorno, entropia, KL, clipfrac, perdas, gradiente, SPS |
| desempenho competitivo | `protocol.json`, `matches.jsonl`, `summary.csv` | placares, seeds, lados, W/D/L, falhas e hashes |
| comportamento | replay, vídeo ou contagem de ações com run/seed | sustentar a motivação observada |
| decisão | `experiments/decisions.csv` | encadear observação → hipótese → teste → conclusão |
| uso de IA | `report/uso_ia.md` e prompts reais | declaração e anexo |

Retorno de scoring e retorno de shaping têm escalas distintas. Mostrar curvas
rotuladas, sem compará-las como força de jogo. Os últimos 100 episódios são uma
média móvel, não 100 observações independentes em cada ponto. A arena é a
referência final, com observações brutas convertidas dentro do `Agent`.

`matches.jsonl` inclui distribuição de ações, tempos e forfeits. A frequência de
idle não demonstra por si só que o jogador está parado: direções e sprint são
ações persistentes. Para afirmar "parado com posse", verificar posições, posse
e deslocamento em replay ou acrescentar uma métrica antes da comparação. Para
afirmar "chuta pouco", declarar se contou ação de chute ou chute efetivo do
simulador; esses conceitos diferem. Não há instrumentação automática de posse
ou finalizações efetivas neste primeiro pipeline.

## Diário e inventário

Preencher `experiments/runs.csv` imediatamente após iniciar cada run, com caminho,
condição, seed, orçamento total e pai. Atualizar ao terminar/falhar. A entrada de
decisão deve apontar para um arquivo que exista e para o trecho/linha/seed útil.
Conservar resultados negativos e falhas. Não alterar o arquivo bruto de partidas
para remover derrotas por crash: elas são desempenho real da submissão.

Dois exemplos de raciocínio, **sem resultados alegados**:

- Se scoring não produzir gols, verificar curva e replays; E02 testa se sinal
  mais denso ajuda. Se houver shaping alto mas saldo ruim, ele pode estar
  otimizando aproximação ao gol sem converter; documentar essa dissociação.
- Se o agente vencer builtin mas perder para um snapshot, E04 pode testar essa
  vulnerabilidade. Se piorar contra outros oponentes, registrar sobreajuste.
