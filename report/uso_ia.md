# Registro de uso de IA — preencher ao longo do trabalho

Status: registro inicial, não é ainda a declaração completa nem o anexo final.

Nesta preparação foi usado **OpenAI Codex (agente baseado em GPT-6, conforme
identificação da sessão)** para inspecionar a base, propor o protocolo,
implementar PPO/política e ferramentas de execução/evidência, preparar
documentação e verificar testes. O modelo específico da IA do colega ainda
precisa ser registrado. Não atribuir a ela um nome antes de identificá-la.

Para a declaração final, especificar separadamente: auxílio em código,
planejamento experimental, interpretação dos dados, redação, revisão e figuras.
Se ainda não houve auxílio na redação do relatório, não alegar que houve.
Registrar versão/nome visível da ferramenta, data, seção/arquivo afetado e
revisão humana realizada. Resultados vêm dos runs, não da geração de texto.

## Prompts reais para anexar

As solicitações iniciais estão em `prompts/001-preparacao.txt` e
`prompts/002-hardware-prazo.txt`. A primeira transcrição preserva o texto com
normalização da marcação/espaçamento do chat; conferir com a exportação original
ao montar o anexo. A segunda preserva o texto da resposta de hardware/prazo.
Continuar guardando as solicitações posteriores, sem substituí-las por resumos.
Este registro não replica integralmente a conversa e, sozinho, não atende à
exigência de inserir os prompts no anexo do PDF.

| Data | Ferramenta/modelo | Parte do trabalho | Arquivo do prompt exato | Validação humana |
|---|---|---|---|---|
| 2026-11-17 | OpenAI Codex / GPT-6 (identificação da sessão) | preparação do pipeline e documentação | `prompts/001-preparacao.txt`, `prompts/002-hardware-prazo.txt` | pendente: conferir transcrição e alterações |
| 2026-11-17 | OpenAI Codex / GPT-6 (identificação da sessão) | validação com motor Football, smoke, exportação e checker oficial | `prompts/003-validacao-football.txt` | conferir evidências em `docs/projeto/VERIFICACAO.md` e logs |
| 2026-11-17 | OpenAI Codex / GPT-6 (identificação da sessão) | auditoria e explicitação da origem das escolhas de treino | `prompts/004-documentacao-decisoes.txt` | conferir `docs/projeto/DECISOES.md` |
| 2026-11-17 | OpenAI Codex / GPT-6 (identificação da sessão) | análise da candidata L40S e preparação de diagnóstico/benchmark | `prompts/005-candidata-l40s.txt`, `prompts/006-protocolo-l40s.txt` | conferir `docs/projeto/VALIDAR_L40S.md`; resultados remotos pendentes |

Não copiar instruções internas da ferramenta nem credenciais para o anexo.
Somente prompts e interações relevantes do usuário e a descrição transparente
do uso. Guardar também os prompts de eventuais outras IAs usadas pelo grupo.
