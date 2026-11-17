# Roteiro do relatório — resultados ainda não executados

Este arquivo planeja o PDF; não é a entrega final. O limite informado é
**6 páginas, duas colunas**, incluindo inicialmente referências e anexo até
confirmação do professor. Não gerar resultados fictícios para ocupar tabelas.

| Parte | Espaço inicial | Conteúdo e evidência |
|---|---|---|
| Título, resumo e contexto | 0,3 página | problema, tarefa do torneio e contribuição realmente testada |
| 1. Modelagem do MDP | 0,8 | estado do simulador versus observação, 115 entradas, 19 ações, recompensa, término/horizonte, gamma, um jogador e limites de observabilidade |
| 2. Pipeline final | 1,0 | PPO, duas MLPs 2×64 Tanh, tabela de hiperparâmetros, fases reais, orçamento, adversários, seeds, origem dos pesos e critério de transição |
| 3. Experimentos e resultados | 1,8 | E01/E02 e extensões efetivamente rodadas; hipóteses, evidências anteriores/posteriores, negativos, comparação final |
| 4. Discussão | 0,7 | interpretação, incerteza entre seeds, overfitting, inferência CPU e limites; próximos testes |
| Referências e anexo de IA | 1,4 | métodos consultados, declaração e prompts reais; ajustar sem violar limite |

Essa distribuição é um ponto de partida, não substitui renderização e contagem
de páginas. Se os prompts excederem o espaço, esclarecer a regra do anexo com o
professor; não comprimir texto até ficar ilegível nem simplesmente excluir prompts.

Figuras/tabelas essenciais:

1. Uma tabela de modelagem e hiperparâmetros efetivamente usados.
2. Curvas com eixo de passos, recompensa rotulada, seeds e IDs de runs. Não
   comparar retorno de shaping e scoring como a mesma métrica.
3. Tabela de ablações: controle, alteração, passos totais (incluindo pais),
   sementes, W/D/L ou score rate contra builtin, e evidência da decisão.
4. Avaliação final contra random, builtin e versões anteriores, com número de
   partidas, lados alternados, seeds, intervalo de vitória e contagem de falhas.

Para cada afirmação, manter uma ligação auditável entre célula/figura e arquivos
brutos. Relatar tanto os runs que aprenderam quanto os negativos; não trocar
motivos planejados por supostas observações. Distinguir dispersão entre seeds de
treinamento da incerteza das partidas de uma única política.

Texto de modelagem precisa refletir o código final, especialmente se mudar
representação, desconto, número de jogadores, recompensa ou tratamento de término.
Na discussão, reconhecer que limites de tempo e torneio com equipes desconhecidas
podem diferir dos adversários de treino. Não alegar que PASS implica competitividade.
