# Gestão de Fraude em Negócios Digitais
### um survey sobre práticas, métricas e controles

**TCC — MBA em Digital Business | USP/ESALQ — 2026**  
Autor: Breno Mendes Moura  
Orientadora: Gisela Consolmagno Pelegrini

---

## Sobre o trabalho

Este repositório contém o código Python que sustenta a análise quantitativa do Trabalho de Conclusão de Curso apresentado ao MBA em Digital Business da USP/ESALQ.

O estudo integrou duas fontes de dados:

- **Survey** aplicado a 57 profissionais de tecnologia, dados, risco e operações
- **Base transacional pública** de cartão de crédito (Credit Card Fraud, Europa/2013)

O objetivo foi demonstrar, de forma replicável, como um framework analítico baseado em Machine Learning pode identificar padrões de fraude e orientar decisões operacionais de controle — com foco no trade-off entre mitigação de risco e custo operacional associado ao volume de alertas gerados.

---

## Principais resultados

| Modelo | PR-AUC (teste) | ROC-AUC (teste) |
|---|---|---|
| Regressão Logística (baseline) | 0,6826 | 0,9633 |
| **Random Forest** | **0,7426** | **0,9749** |

**Threshold recomendado: 0,30**

| Limiar | Alertas | Recall | Precisão | FN | Custo FN estimado |
|---|---|---|---|---|---|
| 0,50 | 87 | 77,0% | 65,5% | 17 | R$ 2.078 |
| **0,30** | **166** | **82,4%** | **36,7%** | **13** | **R$ 1.589** |
| 0,20 | 291 | 86,5% | 22,0% | 10 | R$ 1.222 |
| 0,10 | 1.226 | 91,9% | 5,5% | 6 | R$ 733 |

As 4 variáveis mais preditivas (V14, V10, V17, V4) respondem por **56,1%** da capacidade do modelo.

---

## Estrutura do repositório

```
.
├── fraud_detection_tcc.py     # Script principal — pipeline completo
├── requirements.txt           # Dependências Python
├── README.md                  # Este arquivo
└── figures/                   # Gráficos gerados (após execução)
    ├── fig1_abordagens_survey.png
    ├── fig2_prioridade_survey.png
    ├── fig3_pr_curve.png
    ├── fig4_threshold_impact.png
    └── fig5_feature_importance.png
```

---

## Como reproduzir

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/tcc-fraude-digital.git
cd tcc-fraude-digital
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe a base de dados

Acesse o Kaggle e baixe o arquivo `creditcard.csv`:  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Coloque o arquivo na raiz do repositório (mesma pasta do script).

> A base não está incluída no repositório por conta do tamanho (~144 MB) e das restrições de licença do Kaggle.

### 4. Execute o script

```bash
# Execução padrão (base na mesma pasta)
python fraud_detection_tcc.py

# Especificando caminho da base e pasta de saída dos gráficos
python fraud_detection_tcc.py --data caminho/creditcard.csv --output figures/
```

---

## O que o script produz

O pipeline completo executa as seguintes etapas e gera as saídas correspondentes:

| Etapa | Saída |
|---|---|
| Inspeção exploratória da base | Print no terminal |
| Pré-processamento (sem leakage) | Split 70/15/15 estratificado |
| Treinamento dos modelos | Regressão Logística + Random Forest |
| Avaliação por PR-AUC | Tabela comparativa no terminal |
| Simulação de thresholds | Tabela com custo por cenário |
| Feature importance | Top 10 variáveis + barras visuais |
| Gráficos do TCC | 5 arquivos PNG em `figures/` |
| Síntese gerencial | Recomendações operacionais no terminal |

---

## Notas metodológicas

**Por que PR-AUC em vez de ROC-AUC?**  
A base apresenta forte desbalanceamento (0,17% de fraudes). Um modelo que classifique todas as transações como legítimas atingiria ROC-AUC próxima de 0,50, mas PR-AUC próxima de zero — tornando a PR-AUC a métrica mais informativa neste contexto (Saito; Rehmsmeier, 2015).

**Por que `class_weight='balanced'` em vez de SMOTE?**  
O ajuste de pesos por classe evita a necessidade de oversampling e, consequentemente, elimina o risco de data leakage durante a validação cruzada.

**Por que normalizar Amount e Time após o split?**  
O `StandardScaler` é ajustado exclusivamente no conjunto de treino e aplicado via `transform` nos conjuntos de validação e teste. Isso evita que informações estatísticas dos dados de teste contaminem o treinamento.

---

## Dependências

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
```

---

## Referências

- Bolton, R.J.; Hand, D.J. 2002. Statistical fraud detection: a review. *Statistical Science*, 17(3): 235–255.
- Carcillo, F. et al. 2021. Scarff: a scalable framework for streaming credit card fraud detection with sliding windows. *IEEE Transactions on Neural Networks and Learning Systems*, 32(10): 1–15.
- Dal Pozzolo, A. 2015. Calibrando probabilidades com undersampling para classes desbalanceadas. In: *IEEE SSCI*. IEEE.
- Dal Pozzo, A. et al. 2020. Adaptive machine learning for fraud detection. *Journal of Financial Crime*, 27(4): 1–15.
- Gil, A.C. 2008. *Métodos e técnicas de pesquisa social*. 6. ed. Atlas, São Paulo, SP, Brasil.
- Kaggle. 2025. Credit Card Fraud Detection Dataset. Disponível em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Saito, T.; Rehmsmeier, M. 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3): e0118432.

---

## Licença

Este código é disponibilizado para fins acadêmicos e de reprodutibilidade científica.  
A base de dados `creditcard.csv` está sujeita à licença do Kaggle — consulte os termos de uso antes de qualquer utilização comercial.
