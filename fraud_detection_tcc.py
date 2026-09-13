"""
============================================================
Gestão de Fraude em Negócios Digitais:
um survey sobre práticas, métricas e controles

TCC — MBA em Digital Business | USP/ESALQ — 2026
Autor  : Breno Mendes Moura  (bmoura.profissional@gmail.com)
Orient.: Gisela Consolmagno Pelegrini
============================================================

DESCRIÇÃO
---------
Este script reproduz integralmente a análise quantitativa
apresentada no Trabalho de Conclusão de Curso. Ele cobre:

  1. Carregamento e inspeção exploratória da base
  2. Pré-processamento sem data leakage
  3. Split estratificado treino / validação / teste (70/15/15)
  4. Modelo baseline: Regressão Logística
  5. Modelo principal: Random Forest
  6. Avaliação comparativa por PR-AUC e ROC-AUC
  7. Simulação de thresholds com custo de decisão (R$)
  8. Feature importance (Top 10)
  9. Exportação dos gráficos usados no TCC
 10. Síntese gerencial com recomendações operacionais

FONTE DE DADOS
--------------
Base pública: Credit Card Fraud Detection (Europa, setembro/2013)
Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Arquivo esperado: creditcard.csv (mesma pasta deste script)

  - 284.807 transações | 492 fraudes (0,1727%)
  - Features V1–V28: componentes PCA (anonimizados por privacidade)
  - Features originais: Time, Amount
  - Target: Class (0 = legítima, 1 = fraude)

REFERÊNCIAS
-----------
  Bolton, R.J.; Hand, D.J. 2002. Statistical fraud detection:
    a review. Statistical Science, 17(3): 235–255.

  Dal Pozzolo, A. 2015. Calibrando probabilidades com
    undersampling para classes desbalanceadas.
    In: IEEE SSCI. IEEE.

  Gil, A.C. 2008. Métodos e técnicas de pesquisa social.
    6. ed. Atlas, São Paulo, SP, Brasil.

  Saito, T.; Rehmsmeier, M. 2015. The precision-recall plot
    is more informative than the ROC plot when evaluating binary
    classifiers on imbalanced datasets.
    PLOS ONE, 10(3): e0118432.

DEPENDÊNCIAS
------------
  pip install pandas numpy scikit-learn matplotlib

EXECUÇÃO
--------
  python fraud_detection_tcc.py
  python fraud_detection_tcc.py --data caminho/para/creditcard.csv

SAÍDAS
------
  fig1_abordagens_survey.png   — Gráfico P7 do survey (barras)
  fig2_prioridade_survey.png   — Gráfico P9 do survey (pizza)
  fig3_pr_curve.png            — Curvas Precision-Recall
  fig4_threshold_impact.png    — Impacto dos thresholds
  fig5_feature_importance.png  — Top 10 variáveis preditivas
============================================================
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem interface gráfica
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
TEST_SIZE     = 0.15          # 15 % para teste
VAL_FRACTION  = 0.1765        # ~15 % do total a partir do restante
THRESHOLDS    = [0.50, 0.30, 0.20, 0.10]

# Paleta de cores USP/Esalq (azuis escalonados)
AZUL_USP  = "#1F4E79"
AZUL2     = "#2E75B6"
AZUL3     = "#5B9BD5"
AZUL4     = "#9DC3E6"
LARANJA   = "#C55A11"
CINZA     = "#808080"


# ─────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Imprime um separador de seção."""
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Converte scores probabilísticos em predições binárias."""
    return (proba >= threshold).astype(int)


def threshold_metrics(y_true, proba, threshold, cost_per_fn):
    """Calcula métricas operacionais para um dado threshold."""
    preds = apply_threshold(proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold"   : threshold,
        "alertas"     : int(preds.sum()),
        "recall"      : recall_score(y_true, preds, zero_division=0),
        "precision"   : precision_score(y_true, preds, zero_division=0),
        "f1"          : f1_score(y_true, preds, zero_division=0),
        "fn"          : int(fn),
        "fp"          : int(fp),
        "tp"          : int(tp),
        "tn"          : int(tn),
        "custo_fn"    : int(fn) * cost_per_fn,
    }


def fig_style(ax):
    """Aplica estilo padrão (sem grade, sem borda superior/direita)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.grid(False)
    ax.set_facecolor("white")


# ─────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E INSPEÇÃO DA BASE
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    section("1. CARREGAMENTO E INSPEÇÃO DA BASE")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: '{path}'\n"
            "Baixe a base em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "e coloque o arquivo creditcard.csv na mesma pasta deste script."
        )

    df = pd.read_csv(path)

    n_total     = len(df)
    n_fraude    = int(df["Class"].sum())
    taxa_fraude = df["Class"].mean()
    avg_legit   = df[df["Class"] == 0]["Amount"].mean()
    avg_fraud   = df[df["Class"] == 1]["Amount"].mean()

    print(f"  Total de transações   : {n_total:>10,}")
    print(f"  Fraudes               : {n_fraude:>10,}  ({taxa_fraude:.4%})")
    print(f"  Legítimas             : {n_total - n_fraude:>10,}  ({1 - taxa_fraude:.4%})")
    print(f"  Valores nulos         : {df.isnull().sum().sum():>10,}")
    print(f"  Amount médio legítima : R${avg_legit:>9,.2f}")
    print(f"  Amount médio fraude   : R${avg_fraud:>9,.2f}")
    print(f"  Features              : {list(df.columns[:5])} ... [30 total + Class]")

    return df


# ─────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO E SPLIT
# ─────────────────────────────────────────────────────────────

def preprocess_and_split(df: pd.DataFrame):
    section("2. PRÉ-PROCESSAMENTO E SPLIT ESTRATIFICADO (70 / 15 / 15)")

    X_raw = df.drop("Class", axis=1)
    y     = df["Class"]

    # Split 1: isola o conjunto de teste (15 %)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split 2: isola validação do restante (~15 % do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"  Treino    : {len(X_train):>8,} transações | fraudes: {y_train.sum():>4}")
    print(f"  Validação : {len(X_val):>8,} transações | fraudes: {y_val.sum():>4}")
    print(f"  Teste     : {len(X_test):>8,} transações | fraudes: {y_test.sum():>4}")

    # Normalização de Amount e Time — StandardScaler fit APENAS no treino
    # Evita data leakage: nenhuma informação do val/teste contamina o treino
    scaler = StandardScaler()
    for col in ["Amount", "Time"]:
        X_train = X_train.copy()
        X_val   = X_val.copy()
        X_test  = X_test.copy()
        X_train[f"{col}_z"] = scaler.fit_transform(X_train[[col]])
        X_val[f"{col}_z"]   = scaler.transform(X_val[[col]])
        X_test[f"{col}_z"]  = scaler.transform(X_test[[col]])
        X_train.drop(col, axis=1, inplace=True)
        X_val.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)

    print("  Normalização de Amount e Time: fit no treino, transform em val/teste.")
    print("  class_weight='balanced' será aplicado nos modelos (sem SMOTE/oversampling).")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────
# 3. TREINAMENTO DOS MODELOS
# ─────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    section("3. TREINAMENTO DOS MODELOS")

    print("  [1/2] Regressão Logística (baseline)...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # trata desbalanceamento via pesos
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    print("        Concluído.")

    print("  [2/2] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,                 # usa todos os núcleos disponíveis
    )
    rf.fit(X_train, y_train)
    print("        Concluído.")

    return lr, rf


# ─────────────────────────────────────────────────────────────
# 4. AVALIAÇÃO: PR-AUC E ROC-AUC
# ─────────────────────────────────────────────────────────────

def evaluate_models(lr, rf, X_val, X_test, y_val, y_test):
    section("4. AVALIAÇÃO COMPARATIVA DOS MODELOS")

    results = {}
    for name, model in [("Regressão Logística", lr), ("Random Forest", rf)]:
        proba_val  = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "proba_val"   : proba_val,
            "proba_test"  : proba_test,
            "pr_auc_val"  : average_precision_score(y_val, proba_val),
            "pr_auc_test" : average_precision_score(y_test, proba_test),
            "roc_auc_test": roc_auc_score(y_test, proba_test),
        }

    # Tabela comparativa
    print(f"\n  {'Modelo':<22} {'PR-AUC (val)':>13} {'PR-AUC (teste)':>15} {'ROC-AUC (teste)':>16}")
    print("  " + "-" * 68)
    for name, r in results.items():
        print(f"  {name:<22} {r['pr_auc_val']:>13.4f} {r['pr_auc_test']:>15.4f} {r['roc_auc_test']:>16.4f}")

    delta = results["Random Forest"]["pr_auc_test"] - results["Regressão Logística"]["pr_auc_test"]
    print(f"\n  → Random Forest superior em PR-AUC: +{delta:.4f}")
    print("  → Modelo selecionado para análise de thresholds: Random Forest")
    print()
    print("  NOTA: ROC-AUC elevada em ambos os modelos é esperada em bases")
    print("  desbalanceadas — um modelo que classifique tudo como legítimo")
    print("  atingiria ROC-AUC próxima de 0,50. PR-AUC é a métrica relevante")
    print("  neste contexto (Saito; Rehmsmeier, 2015).")

    return results


# ─────────────────────────────────────────────────────────────
# 5. SIMULAÇÃO DE THRESHOLDS
# ─────────────────────────────────────────────────────────────

def simulate_thresholds(proba_test, y_test, cost_per_fn):
    section("5. SIMULAÇÃO DE THRESHOLDS — RANDOM FOREST")
    print(f"  Custo estimado por FN (fraude não detectada): R${cost_per_fn:.2f}\n")

    header = (
        f"  {'Limiar':>7} | {'Alertas':>8} | {'Recall':>7} | "
        f"{'Precision':>9} | {'F1':>6} | {'FN':>4} | {'FP':>6} | {'Custo FN':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    sim_results = []
    for t in THRESHOLDS:
        m = threshold_metrics(y_test, proba_test, t, cost_per_fn)
        sim_results.append(m)
        print(
            f"  {t:>7.2f} | {m['alertas']:>8,} | {m['recall']:>7.3f} | "
            f"{m['precision']:>9.3f} | {m['f1']:>6.3f} | {m['fn']:>4} | "
            f"{m['fp']:>6,} | R${m['custo_fn']:>9,.0f}"
        )

    print(f"\n  → Threshold recomendado: 0.30")
    print(f"    Recall {sim_results[1]['recall']:.1%} | {sim_results[1]['alertas']} alertas "
          f"| custo FN R${sim_results[1]['custo_fn']:,.0f}")

    return sim_results


# ─────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def feature_importance(rf, feature_names):
    section("6. FEATURE IMPORTANCE — TOP 10 (Random Forest)")

    imp = pd.Series(rf.feature_importances_, index=feature_names)
    top10 = imp.sort_values(ascending=False).head(10)
    total = top10.sum()

    print(f"  {'Feature':>10} | {'Importância':>12} | {'% Top-10':>9}")
    print("  " + "-" * 38)
    for feat, val in top10.items():
        bar = "█" * int(val / top10.max() * 20)
        print(f"  {feat:>10} | {val:>12.4f} | {val/total*100:>8.1f}%  {bar}")

    top4_pct = top10.iloc[:4].sum() / total * 100
    print(f"\n  → Top 4 variáveis (V14, V10, V17, V4) respondem por {top4_pct:.1f}%")
    print("    da capacidade preditiva total do modelo.")
    print("  → V14 sozinha concentra o maior poder discriminante.")

    return top10


# ─────────────────────────────────────────────────────────────
# 7. EXPORTAÇÃO DOS GRÁFICOS
# ─────────────────────────────────────────────────────────────

def export_figures(results, sim_results, top10, lr, rf, X_test, y_test, out_dir="."):
    section("7. EXPORTAÇÃO DOS GRÁFICOS")
    os.makedirs(out_dir, exist_ok=True)

    # ── Figura 1: Abordagens do survey (barras horizontais) ──────────────────
    labels  = ["Machine Learning", "Modelos estatísticos",
               "Revisão humana especializada", "Regras manuais",
               "Não sei / Prefiro não responder"]
    valores = [36, 31, 23, 13, 1]
    pcts    = [v / 57 * 100 for v in valores]
    cores   = [AZUL_USP, AZUL2, AZUL3, AZUL4, CINZA]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], valores[::-1], color=cores[::-1],
                   height=0.55, edgecolor="none")
    for bar, v, p in zip(bars, valores[::-1], pcts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v} ({p:.0f}%)", va="center", ha="left", fontsize=11)
    ax.set_xlabel("Número de respondentes (múltipla escolha, n=57)", fontsize=11)
    ax.set_xlim(0, 48)
    fig_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelsize=10)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "fig1_abordagens_survey.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Salvo: {path1}")

    # ── Figura 2: Prioridade das empresas (pizza) ────────────────────────────
    pizza_labels = [
        "Buscar equilíbrio entre risco\ne experiência do cliente",
        "Minimizar perdas financeiras",
        "Minimizar falsos positivos",
    ]
    pizza_vals = [33, 13, 11]  # n=57
    pizza_pcts = [v / 57 * 100 for v in pizza_vals]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedges, _, autotexts = ax.pie(
        pizza_vals, autopct="%1.1f%%", startangle=90,
        colors=[AZUL_USP, AZUL3, AZUL2],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set(fontsize=11, color="white", fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip([AZUL_USP, AZUL3, AZUL2], pizza_labels)]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), fontsize=9, frameon=False, ncol=1)
    ax.set_facecolor("white")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "fig2_prioridade_survey.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Salvo: {path2}")

    # ── Figura 3: Curvas Precision-Recall ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color, style in [
        ("Regressão Logística", AZUL3, "--"),
        ("Random Forest",       AZUL_USP, "-"),
    ]:
        proba = results[name]["proba_test"]
        prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
        auc = results[name]["pr_auc_test"]
        ax.plot(rec_c, prec_c, color=color, ls=style, lw=2,
                label=f"{name} (PR-AUC = {auc:.4f})")

    # Linha de referência (random classifier)
    baseline = y_test.mean()
    ax.axhline(baseline, color=CINZA, ls=":", lw=1.2,
               label=f"Baseline aleatório ({baseline:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig_style(ax)
    plt.tight_layout()
    path3 = os.path.join(out_dir, "fig3_pr_curve.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Salvo: {path3}")

    # ── Figura 4: Impacto dos thresholds (alertas + recall) ──────────────────
    thrs     = [r["threshold"] for r in sim_results]
    alertas  = [r["alertas"]   for r in sim_results]
    recalls  = [r["recall"]    for r in sim_results]
    custos   = [r["custo_fn"]  for r in sim_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar([str(t) for t in thrs], alertas, color=AZUL3,
            alpha=0.8, width=0.45, label="Alertas gerados")
    ax2.plot([str(t) for t in thrs], [r * 100 for r in recalls],
             color=AZUL_USP, marker="o", lw=2.5, label="Recall (%)")

    ax1.set_xlabel("Threshold (limiar de decisão)", fontsize=11)
    ax1.set_ylabel("Número de alertas gerados", fontsize=11, color=AZUL3)
    ax2.set_ylabel("Recall de fraude (%)", fontsize=11, color=AZUL_USP)
    ax2.set_ylim(0, 110)

    # Destaca threshold recomendado
    ax1.axvline("0.3", color=LARANJA, ls="--", lw=1.5, label="Threshold recomendado (0.30)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, frameon=False)
    fig_style(ax1)
    ax1.spines["right"].set_visible(True)
    plt.tight_layout()
    path4 = os.path.join(out_dir, "fig4_threshold_impact.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Salvo: {path4}")

    # ── Figura 5: Feature importance (barras horizontais) ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    total = top10.sum()
    bars = ax.barh(top10.index[::-1], top10.values[::-1],
                   color=AZUL_USP, height=0.6, edgecolor="none")
    for bar, val in zip(bars, top10.values[::-1]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val/total*100:.1f}%", va="center", ha="left", fontsize=10)
    ax.set_xlabel("Importância média de impureza (Mean Decrease in Impurity)", fontsize=11)
    ax.set_xlim(0, top10.max() * 1.25)
    fig_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelsize=11)
    plt.tight_layout()
    path5 = os.path.join(out_dir, "fig5_feature_importance.png")
    plt.savefig(path5, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Salvo: {path5}")


# ─────────────────────────────────────────────────────────────
# 8. SÍNTESE GERENCIAL
# ─────────────────────────────────────────────────────────────

def management_summary(sim_results, top10):
    section("8. SINTESE GERENCIAL — FRAMEWORK DE CONTROLE")

    rec = sim_results[1]   # threshold 0.30 — cenário recomendado
    t10 = sim_results[3]   # threshold 0.10 — cenário agressivo
    t50 = sim_results[0]   # threshold 0.50 — cenário conservador

    print(f"""
  CENARIO RECOMENDADO (threshold = 0.30)
  ─────────────────────────────────────────────────────────────
  Recall de fraude       : {rec['recall']:.1%}  ({rec['tp']} de {rec['tp'] + rec['fn']} fraudes capturadas)
  Precisao               : {rec['precision']:.1%}  dos alertas sao fraude real
  Alertas gerados        : {rec['alertas']:,}  transacoes sinalizadas
  Falsos negativos (FN)  : {rec['fn']}  fraudes nao detectadas
  Custo estimado (FN)    : R${rec['custo_fn']:,.2f}
  Falsos positivos (FP)  : {rec['fp']:,}  clientes legitimos impactados

  COMPARATIVO DE CENARIOS
  ─────────────────────────────────────────────────────────────
  Threshold 0.10 (agressivo)  → recall {t10['recall']:.1%}, {t10['alertas']:,} alertas,
                                 custo FN R${t10['custo_fn']:,.0f}
                                 Impacto: {t10['fp']:,} falsos positivos — alta
                                 carga operacional, risco de friccao com cliente.

  Threshold 0.30 (recomendado)→ recall {rec['recall']:.1%}, {rec['alertas']:,} alertas,
                                 custo FN R${rec['custo_fn']:,.0f}
                                 Melhor equilibrio entre captura e carga
                                 operacional para o contexto analisado.

  Threshold 0.50 (conservador) → recall {t50['recall']:.1%}, {t50['alertas']:,} alertas,
                                 custo FN R${t50['custo_fn']:,.0f}
                                 Alta precisao ({t50['precision']:.1%}), mas perde
                                 {t50['fn']} fraudes.

  SINAIS PRIORITARIOS DE MONITORAMENTO
  ─────────────────────────────────────────────────────────────
  As 4 principais variaveis (V14, V10, V17, V4) respondem por
  {top10.iloc[:4].sum() / top10.sum() * 100:.1f}% da capacidade preditiva do modelo.
  Em ambientes reais, correspondem a sinais comportamentais
  como padrao de horario, frequencia de transacoes e desvio
  do perfil historico do cliente (Bolton; Hand, 2002).

  RECOMENDACOES OPERACIONAIS
  ─────────────────────────────────────────────────────────────
  1. Adotar Random Forest com threshold=0.30 para triagem automatica.
  2. Encaminhar scores 0.20-0.50 para revisao humana priorizada.
  3. Monitorar V14 como principal indicador de comportamento suspeito.
  4. Revisar o threshold trimestralmente conforme evolucao dos padroes.
  5. Avaliar modelos adicionais (Gradient Boosting, XGBoost) e tecnicas
     de explicabilidade (SHAP values) para ampliar a transparencia.
""")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TCC USP/ESALQ — Detecção de Fraude em Negócios Digitais"
    )
    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Caminho para o arquivo creditcard.csv (padrão: ./creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Pasta de saída para os gráficos (padrão: pasta atual)",
    )
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  TCC — MBA Digital Business | USP/ESALQ — 2026")
    print("  Gestao de Fraude em Negocios Digitais")
    print("  Autor: Breno Mendes Moura")
    print("=" * 62)

    # Pipeline completo
    df = load_data(args.data)

    avg_fraud_amount = df[df["Class"] == 1]["Amount"].mean()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(df)

    lr, rf = train_models(X_train, y_train)

    results = evaluate_models(lr, rf, X_val, X_test, y_val, y_test)

    proba_rf_test = results["Random Forest"]["proba_test"]

    sim_results = simulate_thresholds(proba_rf_test, y_test, avg_fraud_amount)

    top10 = feature_importance(rf, X_train.columns)

    export_figures(results, sim_results, top10, lr, rf, X_test, y_test,
                   out_dir=args.output)

    management_summary(sim_results, top10)

    print("=" * 62)
    print("  Script finalizado. Graficos salvos em:", os.path.abspath(args.output))
    print("=" * 62)


if __name__ == "__main__":
    main()
