# Trabalho final de Aprendizado por Reforço

Leia primeiro `docs/projeto/PLANO.md`, depois `docs/projeto/EXECUCAO.md` e
`docs/projeto/EXPERIMENTOS.md`. O ponto de entrada do projeto é
`scripts/football_train.py`; `train.py` continua sendo a entrada geral da base.
Consulte `docs/projeto/DECISOES.md` para a origem de cada escolha inicial:
defaults e hipóteses não devem ser apresentados como melhorias demonstradas.

## Objetivo e contexto

- Duas semanas de treino. Máquina de execução: Windows 11, Core Ultra 9,
  RTX 5070, 32 GB RAM, 8 GB VRAM, informados pelo usuário. Usar WSL2 + Docker.
- Alternativa em avaliação: instância Ubuntu/KVM com L40S. A troca ainda não
  foi decidida. Seguir `docs/projeto/VALIDAR_L40S.md` para limites, smoke e
  benchmark 4×128 versus 8×64; não iniciar a campanha a partir da ficha técnica.
- Entregar agente autossuficiente para a arena `5_vs_5` e relatório PDF de
  **até 6 páginas, duas colunas**. Declarar IA e incluir prompts em anexo.
- Resultados e motivos observados precisam de arquivos de evidência e IDs.
  Uma hipótese planejada não é um resultado. Não preencher números ausentes.

## Antes de gastar horas de treino

1. Validar dependências, testes focados, smoke de futebol e exportação/probe.
2. Medir SPS na máquina de execução. Reservar tempo para partidas de avaliação.
3. Registrar commit, configuração resolvida, hardware e orçamento. Versionar
   os arquivos novos antes da campanha: `git diff` não inclui arquivos untracked.
4. Não alterar regras, checker ou pontuação da arena para fazer um agente passar.
5. Não iniciar dezenas de runs ou instalar/buildar CUDA sem medir necessidade.

## Protocolo

- Começar E01/E02 com seeds 1, 2, 3 e mesmo orçamento. Um piloto de uma seed
  serve para diagnóstico, não sustenta alegação geral de superioridade.
- Currículo, self-play e outras extensões exigem registro no diário de decisão.
  Mudar um fator por comparação, ou declarar explicitamente um pacote de mudanças.
- Separar avaliação de desenvolvimento (seeds a partir de 100000) do teste final
  reservado (a partir de 2000000). Não escolher modelo pelo teste final.
- Preservar snapshots dos adversários, `matches.jsonl`, `summary.csv`, configs,
  métricas locais, logs e hashes. Não sobrescrever resultados ou versões de agentes.
- `--resume`: retomar run interrompido, mesmo ambiente/modelagem. `--init-from`:
  nova fase, pesos apenas, Adam e agenda de LR novos. Contabilizar passos do pai.
- O treino final nativo apaga checkpoints de retomada; `model.pt` é só pesos.
  Não prometer retomada exata do simulador nem jogo idêntico entre máquinas.
- Usar CPU como referência de inferência: o campeonato usa CPU, uma thread,
  média de `act` até 20 ms, pasta até 100 MB. Rodar o checker oficial em CPU.
- Não adicionar dependência exclusiva do checkout ao `agent.py` exportado.
- Registrar os prompts reais e a identificação da IA em
  `report/uso_ia.md`; o texto inicial desse arquivo ainda não é declaração final.

## Verificação

Consulte os comandos no guia. Os exercícios de A2C/DQN/SAC fora do escopo podem
conter lacunas da base; não declarar a suíte inteira aprovada se só os testes de
PPO foram executados. Smoke sem futebol não valida o motor nem a arena.
