"""
Mean-reverting bivariate SDE for the inoperability vector q(t)
================================================================

Model simulated by this app:

    dq(t) = K c* dt - K (I - A*) q(t) dt + Sigma dW(t)

where:
    q(t)  in R^2   inoperability vector
    K              resilience / mean-reversion (diagonal) matrix
    A*             normalized interdependency matrix (Lian & Haimes 2006, Eq. 6)
    c*             normalized final-demand shift (Lian & Haimes 2006, Eq. 5) --
                    a constant forcing term, user-editable, default = the
                    published deterministic examples of Sec. 6.1/6.2
    Sigma          diffusion matrix
    W(t)           2-D (possibly correlated) Brownian motion

NOTE ON THE MODEL EQUATION (read this if you compare with other sources)
--------------------------------------------------------------------------
The Dynamic Inoperability Input-Output Model (DIIM) of Lian & Haimes (2006),
Eq. (21), is

    q_dot(t) = K [ A* q(t) + c*(t) - q(t) ] = -K(I - A*) q(t) + K c*(t)

This application now simulates exactly this drift structure (same functional
form as Eq. 21), replacing the paper's deterministic/short-run-uncertainty
Wiener term with a general correlated diffusion Sigma dW(t):

    dq(t) = K c* dt - K(I - A*) q(t) dt + Sigma dW(t)

With Sigma = 0 and c* held constant, this reduces exactly to Lian & Haimes'
Eq. (21)/(23), whose deterministic equilibrium (Eq. 25) is

    q(infinity) = (I - A*)^{-1} c*

This closed-form equilibrium is used below as a numerical consistency check
against the paper's own Figures 1 and 2 (Sec. 6.1/6.2). A* is computed
exactly as defined in Lian & Haimes (2006), Eq. (6); only their equations
for A* and for this drift structure are imported, not their (deterministic)
Wiener-noise specification for Sigma.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(page_title="Mean Reverting Multivariato", layout="wide")
st.title("Dynamic Inoperability Input-Output Model (DIIM) - modello mean-reverting")
st.markdown(r"""
Interfaccia interattiva per la simulazione stocastica bivariata:

$$dq(t) = K c^*\, dt -K(I - A^*) q(t)\, dt + \Sigma\, dW(t)$$

dove $q(t)\in\mathbb{R}^2$ è il vettore di inoperabilità, $K$ è la matrice di
resilienza (mean-reversion), $A^*$ è la matrice di interdipendenza normalizzata
definita da Lian & Haimes (2006), Eq. (6), $c^*$ è lo spostamento di domanda
normalizzato (Eq. 5), $\Sigma$ è la matrice di diffusione e $W(t)$ è un moto
Browniano bidimensionale (eventualmente correlato).
""")

with st.expander("ℹ️ Nota sul modello: relazione con il DIIM di Lian & Haimes (2006)", expanded=False):
    st.markdown(r"""
Il modello dinamico di Lian & Haimes (2006), Eq. (21), è

$$\dot q(t) = K\big[A^* q(t) + c^*(t) - q(t)\big] = -K(I-A^*)q(t) + K c^*(t)$$

Questa app implementa **esattamente questa struttura di drift**, sostituendo
al termine di rumore deterministico/a breve termine del paper una diffusione
$\Sigma\,dW(t)$ generale ed eventualmente correlata:

$$dq(t) = K c^*\,dt - K(I-A^*)q(t)\,dt + \Sigma\,dW(t)$$

Con $\Sigma=0$ e $c^*$ costante, questa equazione coincide esattamente con
l'Eq. (21)/(23) del paper, il cui equilibrio deterministico (Eq. 25) è

$$q(\infty) = (I-A^*)^{-1} c^*.$$

Questa formula in forma chiusa viene usata più sotto come controllo numerico
di coerenza rispetto alle Figure 1 e 2 del paper (Sez. 6.1/6.2). Dal paper si
importano quindi **sia** la definizione di $A^*$ (Eq. 6) **sia** questa
struttura di drift (Eq. 21) — non si importa invece la specificazione del
rumore Browniano deterministico del paper per $\Sigma$.
""")

# ============================================================
# VALORI PUBBLICATI DI RIFERIMENTO — Lian & Haimes (2006), Sezione 6
# (Two-industry economic system, Table I ed Eq. 40-43)
# Questi valori NON sono usati direttamente nella simulazione: sono
# tenuti come riferimento/check per la matrice A e A* calcolate.
# ============================================================
Z_REF = np.array([[150.0, 500.0],
                   [200.0, 100.0]])          # Eq. (40), Table I (milioni di $)
X_REF = np.array([1000.0, 2000.0])           # Table I, Total Output (x-hat)
C_REF = np.array([350.0, 1700.0])            # Table I, Final Demand
A_REF = np.array([[0.15, 0.25],
                   [0.20, 0.05]])            # Eq. (41)
A_STAR_REF = np.array([[0.15, 0.5],
                        [0.1, 0.05]])        # Eq. (42)
# --- Errore identificato e CORRETTO nel paper (Eq. 43) ---
# Eq. (31) definisce theta_i = 1 - a*_ii. Applicata letteralmente alla A* di
# Eq. (42) (diagonale a*_ii=0.15, a*_jj=0.05) dà theta = (0.85, 0.95): è il
# valore corretto, ed è quello che questa app calcola e usa (THETA_CORRECT
# sotto, calcolato da A_star a runtime). Il paper, nella sua Eq. (43), STAMPA
# invece theta_i=1-a*_ii=0.15 e theta_j=1-a*_jj=0.05 -- numeri che coincidono
# con a*_ii e a*_jj stessi, non con 1-a*_ii: un errore aritmetico nella fonte,
# non una diversa convenzione (lo stesso valore errato viene riusato come
# "1-a*_ii" anche in Eq. 45, Sez. 6.3). Non abbiamo verificato se questo
# errore sia stato notato/corretto altrove (errata corrige, letteratura
# successiva): la correzione qui applicata è dedotta unicamente dalla coerenza
# interna delle equazioni del paper stesso (Eq. 6, 31, 42).
THETA_PRINTED_IN_PAPER = np.array([0.15, 0.05])   # Eq. (43), come stampato (ERRATO)

# Equilibrio deterministico pubblicato per la Figura 1 (Sez. 6.1, solo
# riduzione di domanda, c*=(0,0.10), q(0)=(0,0)): il testo del paper afferma
# "stabilizes at around 11.2%" (settore j) e "6.6%" (settore i) -- Eq. (25).
# NON dipende da theta (vedi Eq. 24/25: solo A*, K, c*), quindi NON e' affetto
# dall'errore in Eq. (43) -- verificato più sotto confrontando col calcolo.
Q_EQ_FIG1_PAPER = np.array([0.066, 0.112])   # valori citati nel testo, Sez. 6.1

C_STAR_FIG1_REF = np.array([0.0, 0.10])      # Sez. 6.1 (Fig. 1): riduzione domanda 10% su j
C_STAR_FIG2_REF = np.array([0.0, 0.03])      # Sez. 6.2 (Fig. 2): riduzione domanda 3% su j
K_DET_REF = np.array([0.20, 0.20])           # Sez. 6.1/6.2: k_i = k_j = 0.2 (esempio deterministico)
Q0_FIG1_REF = np.array([0.0, 0.0])           # Sez. 6.1: nessun attacco, solo riduzione di domanda
Q0_FIG2_REF = np.array([0.0, 0.10])          # Sez. 6.2: attacco al settore j, q_i(0)=0, q_j(0)=10%

TOL = 1e-9  # tolleranza numerica per i controlli di coerenza

# ============================================================
# PANNELLO LATERALE: PARAMETRI
# ============================================================
st.sidebar.header("Parametri del Modello")

# --- 1. Dati Input-Output: matrice delle transazioni Z e output totale x_hat ---
st.sidebar.subheader("Dati Input-Output (Sez. 6, Lian & Haimes 2006)")
st.sidebar.caption(
    "Valori pubblicati come default (Tabella I). Da qui si CALCOLANO A e A* "
    "(Eq. 6) — non sono inseriti a mano."
)

z11 = st.sidebar.number_input("Z[i,i] (transazione i→i)", value=float(Z_REF[0, 0]), step=10.0, format="%.1f")
z12 = st.sidebar.number_input("Z[i,j] (transazione i→j)", value=float(Z_REF[0, 1]), step=10.0, format="%.1f")
z21 = st.sidebar.number_input("Z[j,i] (transazione j→i)", value=float(Z_REF[1, 0]), step=10.0, format="%.1f")
z22 = st.sidebar.number_input("Z[j,j] (transazione j→j)", value=float(Z_REF[1, 1]), step=10.0, format="%.1f")
Z = np.array([[z11, z12], [z21, z22]])

x1_hat = st.sidebar.number_input("x̂ (output totale) — settore i", value=float(X_REF[0]), step=100.0, format="%.1f")
x2_hat = st.sidebar.number_input("x̂ (output totale) — settore j", value=float(X_REF[1]), step=100.0, format="%.1f")
x_hat = np.array([x1_hat, x2_hat])

if np.any(x_hat <= 0):
    st.sidebar.error("x̂ deve avere componenti strettamente positive (compare al denominatore in A e A*).")
    st.stop()

# --- 2. Coefficienti di resilienza K ---
st.sidebar.subheader("Coefficienti di Resilienza (K)")

scenario_K = st.sidebar.selectbox(
    "Scenario per K:",
    [
        "Pubblicato (Lian & Haimes 2006, Sez. 6.1-6.2): k₁ = k₂ = 0.20",
        "Personalizzato",
        "Simmetrici (k₁ = k₂ = 0.10)",
        "Settore 1 fragile (k₁=0.02, k₂=0.10)",
        "Settore 2 fragile (k₁=0.10, k₂=0.02)",
        "Entrambi fragili (k₁=0.02, k₂=0.02)",
        "Entrambi resilienti (k₁=0.30, k₂=0.30)",
    ],
)

if scenario_K.startswith("Pubblicato"):
    k1, k2 = float(K_DET_REF[0]), float(K_DET_REF[1])
elif scenario_K == "Simmetrici (k₁ = k₂ = 0.10)":
    k1, k2 = 0.10, 0.10
elif scenario_K == "Settore 1 fragile (k₁=0.02, k₂=0.10)":
    k1, k2 = 0.02, 0.10
elif scenario_K == "Settore 2 fragile (k₁=0.10, k₂=0.02)":
    k1, k2 = 0.10, 0.02
elif scenario_K == "Entrambi fragili (k₁=0.02, k₂=0.02)":
    k1, k2 = 0.02, 0.02
elif scenario_K == "Entrambi resilienti (k₁=0.30, k₂=0.30)":
    k1, k2 = 0.30, 0.30
else:
    k1 = st.sidebar.slider("k₁", 0.01, 0.50, 0.20, 0.01)
    k2 = st.sidebar.slider("k₂", 0.01, 0.50, 0.20, 0.01)

K = np.array([[k1, 0.0],
              [0.0, k2]])

# --- 3. Condizioni iniziali q(0) e riduzione di domanda c* (termine K c*) ---
# q0 e c* sono raggruppati in un unico scenario per evitare combinazioni
# incoerenti: le Fig. 1 e Fig. 2 del paper corrispondono ciascuna a UNA
# coppia specifica (q0, c*), entrambe richieste per riprodurle esattamente.
st.sidebar.subheader("Scenario di domanda e condizioni iniziali (Sez. 6)")
st.sidebar.caption(
    "Imposta insieme q(0) e c* (lo spostamento di domanda normalizzato, Eq. 5, "
    "che entra nel drift come termine costante K c*). I due scenari pubblicati "
    "riproducono rispettivamente la Figura 1 e la Figura 2 del paper."
)

scenario_domanda = st.sidebar.selectbox(
    "Scenario:",
    [
        "Pubblicato — Fig. 1 (Sez. 6.1): solo riduzione domanda, q(0)=(0,0), c*=(0,0.10)",
        "Pubblicato — Fig. 2 (Sez. 6.2): shock iniziale + riduzione domanda, q(0)=(0,0.10), c*=(0,0.03)",
        "Solo shock iniziale: c*=(0,0)",
        "Personalizzato",
    ],
)

if scenario_domanda.startswith("Pubblicato — Fig. 1"):
    q1_0, q2_0 = float(Q0_FIG1_REF[0]), float(Q0_FIG1_REF[1])
    c1_star, c2_star = float(C_STAR_FIG1_REF[0]), float(C_STAR_FIG1_REF[1])
elif scenario_domanda.startswith("Pubblicato — Fig. 2"):
    q1_0, q2_0 = float(Q0_FIG2_REF[0]), float(Q0_FIG2_REF[1])
    c1_star, c2_star = float(C_STAR_FIG2_REF[0]), float(C_STAR_FIG2_REF[1])
elif scenario_domanda.startswith("Solo shock"):
    q1_0 = st.sidebar.slider("q₁(0)", 0.0, 1.0, 0.0, 0.05)
    q2_0 = st.sidebar.slider("q₂(0)", 0.0, 1.0, 0.10, 0.05)
    c1_star, c2_star = 0.0, 0.0
else:
    st.sidebar.markdown("**Condizioni iniziali q(0)**")
    q1_0 = st.sidebar.slider("q₁(0)", 0.0, 1.0, 0.0, 0.05)
    q2_0 = st.sidebar.slider("q₂(0)", 0.0, 1.0, 0.10, 0.05)
    st.sidebar.markdown("**Riduzione di domanda normalizzata c\* (Eq. 5)**")
    c1_star = st.sidebar.slider("c₁*", 0.0, 1.0, 0.0, 0.01)
    c2_star = st.sidebar.slider("c₂*", 0.0, 1.0, 0.10, 0.01)

q0 = np.array([q1_0, q2_0])
c_star = np.array([c1_star, c2_star])

# --- 4. Matrice di diffusione Sigma: volatilità + correlazione ---
st.sidebar.subheader("Diffusione (Σ): volatilità e correlazione")
st.sidebar.caption(
    "Si impostano le volatilità marginali σ₁, σ₂ e la correlazione ρ fra i "
    "due moti Browniani. Σ viene poi costruita (fattore di Cholesky) in modo "
    "che ΣΣᵀ sia ESATTAMENTE la matrice di covarianza voluta."
)

scenario_corr = st.sidebar.selectbox(
    "Scenario per la correlazione ρ:",
    [
        "Personalizzato",
        "Indipendenti (ρ = 0)",
        "Correlazione bassa (ρ = 0.2)",
        "Correlazione media (ρ = 0.5)",
        "Correlazione alta (ρ = 0.8)",
        "Correlazione negativa (ρ = -0.5)",
    ],
)

sigma1 = st.sidebar.slider("σ₁ (volatilità settore 1)", 0.001, 0.10, 0.01, 0.005)
sigma2 = st.sidebar.slider("σ₂ (volatilità settore 2)", 0.001, 0.10, 0.01, 0.005)

if scenario_corr == "Indipendenti (ρ = 0)":
    rho = 0.0
elif scenario_corr == "Correlazione bassa (ρ = 0.2)":
    rho = 0.2
elif scenario_corr == "Correlazione media (ρ = 0.5)":
    rho = 0.5
elif scenario_corr == "Correlazione alta (ρ = 0.8)":
    rho = 0.8
elif scenario_corr == "Correlazione negativa (ρ = -0.5)":
    rho = -0.5
else:
    rho = st.sidebar.slider("ρ (correlazione, -0.99 ≤ ρ ≤ 0.99)", -0.99, 0.99, 0.0, 0.01)

# --- Parametri simulazione ---
st.sidebar.subheader("Parametri Simulazione")
n_steps = st.sidebar.slider("Numero di step temporali", 10, 200, 20, 5)
n_simulazioni = st.sidebar.slider("Numero di simulazioni", 10, 500, 100, 10)
dt = 1.0
np.random.seed(st.sidebar.number_input("Seed casuale (per riproducibilità)", value=42, step=1))

# ============================================================
# CALCOLI
# ============================================================
I = np.eye(2)

# ---- Controllo dimensioni (punto 4.1) ----
assert Z.shape == (2, 2), "Z deve essere 2x2"
assert x_hat.shape == (2,), "x_hat deve avere dimensione 2"
assert K.shape == (2, 2), "K deve essere 2x2"
assert q0.shape == (2,), "q0 deve avere dimensione 2"
assert c_star.shape == (2,), "c* deve avere dimensione 2"

# ---- 1) Matrice tecnica A, dai dati grezzi Z e x_hat ----
# Leontief technical coefficient: a_ij = z_ij / x_j
# (si divide ogni COLONNA j di Z per l'output totale x_j di quella colonna)
A_calc = Z / x_hat[np.newaxis, :]

# ---- 2) Matrice normalizzata A*, Eq. (6) di Lian & Haimes (2006) ----
#   A* = [diag(x_hat)]^{-1} A [diag(x_hat)]
# Implementata in modo esplicito e trasparente, passo per passo:
D_xhat = np.diag(x_hat)               # diag(x_hat)
D_xhat_inv = np.diag(1.0 / x_hat)     # [diag(x_hat)]^{-1}
A_star = D_xhat_inv @ A_calc @ D_xhat  # Eq. (6)

assert A_star.shape == (2, 2), "A* deve essere 2x2"

# ---- Indici di interdipendenza, Eq. (31): theta_i = 1 - a*_ii ----
# Applicata LETTERALMENTE alla A* calcolata (vedi nota nell'expander sotto:
# il valore che risulta da questa formula NON coincide con quello stampato
# in Eq. (43) del paper, che sembra un errore aritmetico nella fonte).
theta = 1.0 - np.diag(A_star)

# ---- Matrice di diffusione Sigma (fattore "Cholesky" della covarianza target) ----
# Covarianza voluta:  Sigma Sigma^T = [[sigma1^2 , rho*sigma1*sigma2],
#                                       [rho*sigma1*sigma2, sigma2^2]]
L11 = sigma1
L21 = rho * sigma2
L22 = sigma2 * np.sqrt(max(1.0 - rho ** 2, 0.0))
Sigma = np.array([[L11, 0.0],
                   [L21, L22]])

Cov_target = np.array([[sigma1 ** 2, rho * sigma1 * sigma2],
                        [rho * sigma1 * sigma2, sigma2 ** 2]])
Cov_matrix = Sigma @ Sigma.T  # covarianza effettivamente prodotta da Sigma

# ---- Drift lineare (dipendente da q): -K(I - A*) ----
B = K @ (I - A_star)
drift_matrix = -B

# ---- Drift costante (termine forzante): K c* ----
drift_const = K @ c_star

# ---- Autovalori del drift lineare (per il controllo di mean-reversion, punto 4.10) ----
eig_drift = np.linalg.eigvals(drift_matrix)
is_mean_reverting = np.all(eig_drift.real < 0)

# ---- Equilibrio deterministico, Eq. (25): q(inf) = (I-A*)^{-1} c* ----
# Indipendente da K (K compare in entrambi i termini del drift e si semplifica
# nella condizione di equilibrio K[c* - (I-A*)q] = 0); dipende solo da A* e c*.
try:
    q_eq = np.linalg.solve(I - A_star, c_star)
    q_eq_ok = True
except np.linalg.LinAlgError:
    q_eq = np.full(2, np.nan)
    q_eq_ok = False

# ============================================================
# SIMULAZIONE (Euler-Maruyama)
# dq = [K c* + drift_matrix @ q] * dt + Sigma @ dW,   dW ~ N(0, dt*I)
# ============================================================
risultati = np.zeros((n_simulazioni, 2, n_steps + 1))

for i in range(n_simulazioni):
    q_corrente = q0.copy()
    risultati[i, :, 0] = q_corrente

    for t in range(n_steps):
        dW = np.random.standard_normal(2) * np.sqrt(dt)   # incrementi Browniani indipendenti, varianza dt
        dq = (drift_const + drift_matrix @ q_corrente) * dt + Sigma @ dW  # Sigma correla i due shock
        q_corrente = q_corrente + dq
        risultati[i, :, t + 1] = q_corrente

# Estrazione componenti
q1_sim = risultati[:, 0, :]
q2_sim = risultati[:, 1, :]
media_q1 = np.mean(q1_sim, axis=0)
media_q2 = np.mean(q2_sim, axis=0)

# TRAIETTORIA DETERMINISTICA (Sigma = 0), stessa discretizzazione Euler
q_det = np.zeros((2, n_steps + 1))
q_det[:, 0] = q0.copy()
for t in range(n_steps):
    q_det[:, t + 1] = q_det[:, t] + (drift_const + drift_matrix @ q_det[:, t]) * dt

det_q1 = q_det[0, :]
det_q2 = q_det[1, :]

# ============================================================
# VISUALIZZAZIONE: 1. DATI INPUT-OUTPUT E CALCOLO DI A*
# ============================================================
st.header("1. Dati Input-Output e calcolo di A*")

st.markdown(
    "**Legenda:** 🟦 *pubblicato* = valori riportati in Lian & Haimes (2006), Sez. 6 "
    "&nbsp;&nbsp;|&nbsp;&nbsp; 🟩 *calcolato* = derivato nell'app dai dati Input-Output "
    "&nbsp;&nbsp;|&nbsp;&nbsp; 🟧 *modificabile* = parametro impostato dall'utente nella sidebar"
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🟧 Z — matrice delle transazioni (input utente, default = Tabella I)**")
    st.write(pd.DataFrame(Z, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
    st.markdown("**🟧 x̂ — output totale (input utente, default = Tabella I)**")
    st.write(pd.DataFrame(x_hat.reshape(1, 2), columns=["x̂₁", "x̂₂"], index=["valore"]))

with col2:
    st.markdown(r"**🟩 A — calcolata come $a_{ij}=z_{ij}/x_j$**")
    st.write(pd.DataFrame(A_calc, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
    st.markdown(r"**🟩 A$^*$ — Eq. (6): $A^*=[\mathrm{diag}(\hat x)]^{-1}A\,[\mathrm{diag}(\hat x)]$**")
    st.write(pd.DataFrame(A_star, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

#with st.expander("Mostra il calcolo passo-passo di A* (Eq. 6)"):
#    st.markdown(r"$\mathrm{diag}(\hat x)^{-1}$:")
#    st.write(pd.DataFrame(D_xhat_inv, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
#    st.markdown(r"$A$:")
#    st.write(pd.DataFrame(A_calc, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
#    st.markdown(r"$\mathrm{diag}(\hat x)$:")
#    st.write(pd.DataFrame(D_xhat, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
#    st.markdown(r"$A^* = \mathrm{diag}(\hat x)^{-1}\;A\;\mathrm{diag}(\hat x) =$")
#    st.write(pd.DataFrame(A_star, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

# ------------------------------------------------------------------
# Sezione "Controllo di coerenza con i valori pubblicati (Lian & Haimes
# 2006, Sez. 6)" — SU RICHIESTA, resa non visibile nell'app (commentata).
# check_A e check_Astar restano calcolati (servono più sotto: verifica
# dell'equilibrio Fig. 1 in §2 e riepilogo verifiche in §5).
# ------------------------------------------------------------------
# st.subheader("Controllo di coerenza con i valori pubblicati (Lian & Haimes 2006, Sez. 6)")
# st.caption(
#     "Il confronto ha senso quando Z e x̂ sono lasciati ai valori pubblicati di default; "
#     "se li modificate, uno scostamento da qui in poi è atteso."
# )

check_A = np.allclose(A_calc, A_REF, atol=1e-6) if np.allclose(Z, Z_REF) and np.allclose(x_hat, X_REF) else None
check_Astar = np.allclose(A_star, A_STAR_REF, atol=1e-6) if np.allclose(Z, Z_REF) and np.allclose(x_hat, X_REF) else None

# df_check = pd.DataFrame({
#     "Grandezza": ["A", "A*"],
#     "Calcolato": [str(np.round(A_calc, 4).tolist()), str(np.round(A_star, 4).tolist())],
#     "Pubblicato (Eq. 41-42)": [str(A_REF.tolist()), str(A_STAR_REF.tolist())],
# })
# st.write(df_check)
#
# if np.allclose(Z, Z_REF) and np.allclose(x_hat, X_REF):
#     if check_A and check_Astar:
#         st.success("✅ A e A* calcolate coincidono con i valori pubblicati (Eq. 41-42).")
#     else:
#         st.error("❌ Discrepanza fra A/A* calcolate e quelle pubblicate: verificare la formula.")
# else:
#     st.info("ℹ️ Z e/o x̂ sono stati modificati rispetto ai valori pubblicati: il confronto diretto con Eq. 41-42 non è applicabile.")

#with st.expander("🛠️ θ (indice di interdipendenza, Eq. 31) — errore nel paper, CORRETTO in questa app", expanded=True):
#    st.markdown(r"""
#**Errore individuato nella fonte.** L'Eq. (31) del paper definisce
#$\theta_i = 1-a^*_{ii}$. Applicata alla $A^*$ dell'Eq. (42) (diagonale
#$a^*_{ii}=0.15$, $a^*_{jj}=0.05$), l'unico risultato aritmeticamente
#corretto è $\theta_i=0.85$, $\theta_j=0.95$. Il paper, nella sua Eq. (43), # type: ignore
#stampa invece $\theta_i=1-a^*_{ii}=0.15$ e $\theta_j=1-a^*_{jj}=0.05$ — # type: ignore
#numeri che coincidono con $a^*_{ii}$ e $a^*_{jj}$ stessi, non con
#$1-a^*_{ii}$: un errore aritmetico nella fonte (lo stesso errore si ripropaga
#in Eq. 45, Sez. 6.3, dove gli stessi 0.15/0.05 sono riusati come "$1-a^*_{ii}$"
#per stimare $k_i, k_j$ nello scenario stocastico). Non abbiamo verificato se
#sia un errore noto/corretto altrove (errata corrige o letteratura successiva);
#la correzione qui è dedotta unicamente dalla coerenza interna di Eq. (6),
#(31) e (42) del paper stesso.

#**Correzione implementata.** Questa app calcola $\theta$ applicando l'Eq.
#(31) **correttamente** alla $A^*$ effettiva, cioè $\theta = \mathbf{1} -
#\mathrm{diag}(A^*)$, senza forzare il risultato a coincidere con i numeri
#(errati) stampati in Eq. (43). θ non entra comunque nel drift della SDE
#simulata: è mostrato solo a titolo informativo/di verifica.

#**Le Figure 1 e 2 del paper NON sono affette da questo errore.** Le Eq.
#(24)/(25), che generano le Figure 1 e 2 (Sez. 6.1/6.2), dipendono solo da
#$A^*$, $K$ e $c^*$ — mai da $\theta$. La Sez. 6.1 usa $k_i=k_j=0.2$ assegnato
#direttamente, non derivato da $\theta$. Il controllo più sotto (§2, "Verifica
#dell'equilibrio deterministico") riproduce numericamente l'equilibrio
#$(6.6\%,\,11.2\%)$ citato nel testo per la Figura 1, confermando che $A^*$ e
#la struttura di drift sono corrette a prescindere dall'errore in $\theta$.
#L'unico punto del paper che **userebbe** $\theta$ per calcolare parametri
#numerici è l'Eq. (45) (Sez. 6.3, scenario stocastico, Figura 3): lì
#l'errore in $\theta$ altera sensibilmente $k_i,k_j$ (con la correzione,
#$k_i$ scende da 0.0904 a circa 0.016 e $k_j$ da 0.0808 a circa 0.004, cioè un
#recupero molto più lento) — ma quello scenario stocastico non è
#riprodotto da questa app (si simula la SDE mean-reverting, non il DIIM
#esatto di Sez. 6.3).
#""")
 #   st.write(pd.DataFrame({
 #       "θ CORRETTO (Eq. 31 applicata correttamente, usato da questa app)": np.round(theta, 4),
 #       "θ stampato in Eq. (43) del paper (errato)": THETA_PRINTED_IN_PAPER,
 #   }, index=["i", "j"]))

st.caption(
    f"Rapporto di interdipendenza (Eq. 32), corretto: θ_i/θ_j = {theta[0] / theta[1]:.3f} "
    f"— rapporto citato nel testo del paper (con i valori errati di Eq. 43): "
    f"{THETA_PRINTED_IN_PAPER[0] / THETA_PRINTED_IN_PAPER[1]:.2f}. "
    f"Nota: con θ corretto, l'affermazione del paper \"Industry i is three times more "
    f"interdependent than Industry j\" non è più supportata dai numeri (il rapporto "
    f"corretto è ≈{theta[0] / theta[1]:.2f}, non 3)."
)

# ============================================================
# VISUALIZZAZIONE: 2. MATRICI DEL MODELLO SDE
# ============================================================
st.header("2. Matrici del modello e verifica di mean-reversion")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🟧 Matrice K**")
    st.write(pd.DataFrame(K, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

with col2:
    st.markdown("**🟩 I − A\***")
    st.write(pd.DataFrame(I - A_star, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

with col3:
    st.markdown("**🟩 Drift lineare: −K(I−A\*)**")
    st.write(pd.DataFrame(drift_matrix, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

with col4:
    st.markdown("**🟩 Autovalori del drift lineare**")
    st.write(pd.DataFrame({"Re(λ)": eig_drift.real, "Im(λ)": eig_drift.imag}, index=["λ₁", "λ₂"]))

if is_mean_reverting:
    st.success("✅ Tutti gli autovalori del drift lineare hanno parte reale negativa: il processo è mean-reverting.")
else:
    st.error("❌ Almeno un autovalore del drift ha parte reale ≥ 0: il processo NON è mean-reverting con questi parametri.")

st.subheader("Termine costante K c* e condizioni iniziali")

colc1, colc2, colc3 = st.columns(3)
with colc1:
    st.markdown("**🟧 c\* — riduzione di domanda normalizzata (Eq. 5)**")
    st.write(pd.DataFrame(c_star.reshape(1, 2), columns=["c₁*", "c₂*"], index=["valore"]))
with colc2:
    st.markdown("**🟩 K c\* — termine costante del drift**")
    st.write(pd.DataFrame(drift_const.reshape(1, 2), columns=["(Kc*)₁", "(Kc*)₂"], index=["valore"]))
with colc3:
    st.markdown("**🟧 q(0) — condizioni iniziali**")
    st.write(pd.DataFrame(q0.reshape(1, 2), columns=["q₁(0)", "q₂(0)"], index=["valore"]))

st.subheader("Verifica dell'equilibrio deterministico — Eq. (25): q(∞) = (I−A\*)⁻¹c\*")
st.caption(
    "Questo equilibrio dipende SOLO da A* e c* (non da θ): è quindi il controllo "
    "più diretto che Figure 1 e 2 del paper non sono affette dall'errore in Eq. (43)."
)

if q_eq_ok:
    colE1, colE2 = st.columns(2)
    with colE1:
        st.markdown("**🟩 q(∞) calcolato con i parametri correnti**")
        st.write(pd.DataFrame(q_eq.reshape(1, 2), columns=["q₁(∞)", "q₂(∞)"], index=["valore"]))
    with colE2:
        if np.allclose(c_star, C_STAR_FIG1_REF) and check_Astar:
            err = np.abs(q_eq - Q_EQ_FIG1_PAPER)
            st.markdown("**Confronto con la Figura 1 del paper (Sez. 6.1)**")
            st.write(pd.DataFrame({
                "Calcolato": np.round(q_eq, 4),
                "Citato nel testo (~6.6%, ~11.2%)": Q_EQ_FIG1_PAPER,
            }, index=["i", "j"]))
            if np.all(err < 0.002):
                st.success(f"✅ Equilibrio riprodotto: scostamento massimo {err.max():.4f} (~{err.max()*100:.2f} punti %).")
            else:
                st.error(f"❌ Scostamento dall'equilibrio pubblicato: {err}")
        else:
            st.info("ℹ️ Seleziona lo scenario 'Pubblicato — Fig. 1' (con A*, x̂, Z ai valori di default) "
                    "per confrontare q(∞) con il valore citato nel testo del paper (Sez. 6.1).")
else:
    st.error("❌ (I − A*) è singolare: l'equilibrio q(∞) non è definito per questa A*.")

st.markdown(f"**Condizioni iniziali:** q₁(0) = {q0[0]:.2f}, q₂(0) = {q0[1]:.2f}  |  "
            f"**c\*:** c₁* = {c_star[0]:.2f}, c₂* = {c_star[1]:.2f}")

st.subheader("Diffusione Σ e struttura di correlazione")

col1, col2 = st.columns(2)
with col1:
    st.markdown(r"🟧 Σ ")
    st.write(pd.DataFrame(Sigma, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
with col2:
    st.markdown(r"**🟩 ΣΣᵀ (covarianza)**")
    st.write(pd.DataFrame(Cov_matrix, columns=["j=1", "j=2"], index=["i=1", "i=2"]))
#with col3:
#    st.markdown("**🟩 Covarianza target**")
#    st.write(pd.DataFrame(Cov_target, columns=["j=1", "j=2"], index=["i=1", "i=2"]))

# Correlazione implicita ricavata da ΣΣ^T
if Cov_matrix[0, 0] > 0 and Cov_matrix[1, 1] > 0:
    rho_implicita = Cov_matrix[0, 1] / np.sqrt(Cov_matrix[0, 0] * Cov_matrix[1, 1])
else:
    rho_implicita = np.nan

eig_cov = np.linalg.eigvalsh(Cov_matrix)
is_psd = np.all(eig_cov >= -1e-10)

#colA, colB, colC = st.columns(3)
#with colA:
#    st.metric("ρ impostato (input utente)", f"{rho:.4f}")
#with colB:
#    st.metric("ρ implicito da ΣΣᵀ", f"{rho_implicita:.4f}")
#with colC:
#    st.metric("Autovalori min. di ΣΣᵀ", f"{eig_cov.min():.6f}")

#if abs(rho - rho_implicita) < 1e-6 and np.allclose(Cov_matrix, Cov_target, atol=1e-9):
#    st.success("✅ Σ è coerente con la covarianza target: ΣΣᵀ riproduce esattamente ρ, σ₁, σ₂ impostati.")
#else:
#    st.error("❌ La correlazione implicita da Σ non coincide con ρ impostato: verificare la costruzione di Σ.")

#if is_psd:
#    st.success("✅ ΣΣᵀ è semidefinita positiva (autovalori ≥ 0) per il ρ scelto.")
#else:
#    st.error("❌ ΣΣᵀ NON è semidefinita positiva: ρ non ammissibile.")

# ============================================================
# VISUALIZZAZIONE: 3. EVOLUZIONE DELLE TRAIETTORIE
# ============================================================
st.header("3. Evoluzione delle Traiettorie")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i in range(n_simulazioni):
    ax1.plot(q1_sim[i, :], color='blue', alpha=0.1)
    ax2.plot(q2_sim[i, :], color='red', alpha=0.1)

# Traiettoria media
ax1.plot(media_q1, color='black', linewidth=2, label="Traiettoria Media")
ax2.plot(media_q2, color='black', linewidth=2, label="Traiettoria Media")

# Traiettoria deterministica (Sigma = 0) e livello di equilibrio q(inf)=(I-A*)^{-1}c*
ax1.plot(det_q1, color='orange', linewidth=2, linestyle='-.', label="Deterministica (no rumore)")
ax2.plot(det_q2, color='orange', linewidth=2, linestyle='-.', label="Deterministica (no rumore)")
if q_eq_ok:
    ax1.axhline(y=q_eq[0], color='green', linestyle='--', linewidth=1.5, label=f"Equilibrio q₁(∞)={q_eq[0]:.3f}")
    ax2.axhline(y=q_eq[1], color='green', linestyle='--', linewidth=1.5, label=f"Equilibrio q₂(∞)={q_eq[1]:.3f}")

ax1.set_title("Evoluzione di q₁(t)")
ax1.set_xlabel("Step Temporali")
ax1.set_ylabel("Inoperabilità")
ax1.legend()

ax2.set_title("Evoluzione di q₂(t)")
ax2.set_xlabel("Step Temporali")
ax2.set_ylabel("Inoperabilità")
ax2.legend()

plt.tight_layout()
st.pyplot(fig1)

# ============================================================
# DATI TABELLARI
# ============================================================
st.header("4. Dati Tabellari (Pandas)")

df_q1 = pd.DataFrame(q1_sim.T, columns=[f"Sim_{i}" for i in range(n_simulazioni)])
df_q2 = pd.DataFrame(q2_sim.T, columns=[f"Sim_{i}" for i in range(n_simulazioni)])

tab1, tab2 = st.tabs(["Componente q₁", "Componente q₂"])

with tab1:
    st.dataframe(df_q1.head(10))

with tab2:
    st.dataframe(df_q2.head(10))

# ============================================================
# RIEPILOGO VERIFICHE (punto 5 - verifica finale)
# ============================================================
#st.header("5. Riepilogo delle verifiche")

# checks = []
# # checks.append(("Dimensioni di Z, x̂, A, A*, K, c*, Σ, q0 sono tutte 2×2 / 2", True))
# # checks.append(("A calcolata coincide con Eq. (41) [con dati pubblicati]", bool(check_A) if check_A is not None else "n/a (Z o x̂ modificati)"))
# checks.append(("A* calcolata coincide con Eq. (42) [con dati pubblicati]", bool(check_Astar) if check_Astar is not None else "n/a (Z o x̂ modificati)"))
# checks.append(("θ = 1-diag(A*): Eq. (31) applicata CORRETTAMENTE (errore in Eq. 43 del paper identificato e corretto, vedi nota §1)", True))
# checks.append(("Termine costante K c* incluso nel drift (dq = Kc* dt - K(I-A*)q dt + ΣdW)", True))
# checks.append(("q(∞)=(I-A*)^{-1}c* riproduce l'equilibrio citato per la Figura 1 (Sez. 6.1)"
#                " [solo con scenario 'Fig. 1' e A* pubblicata]",
#                bool(q_eq_ok and np.allclose(c_star, C_STAR_FIG1_REF) and check_Astar
#                     and np.all(np.abs(q_eq - Q_EQ_FIG1_PAPER) < 0.002))
#                if (q_eq_ok and np.allclose(c_star, C_STAR_FIG1_REF) and check_Astar is not None) else "n/a (altro scenario)"))
# checks.append(("ΣΣᵀ riproduce esattamente [[σ1²,ρσ1σ2],[ρσ1σ2,σ2²]]", bool(np.allclose(Cov_matrix, Cov_target, atol=1e-9))))
# checks.append(("ρ impostato = ρ implicito da ΣΣᵀ", bool(abs(rho - rho_implicita) < 1e-6)))
# checks.append(("ΣΣᵀ è semidefinita positiva", bool(is_psd)))
# checks.append(("Drift lineare -K(I-A*) ha tutti gli autovalori con parte reale < 0 (mean-reversion verso q(∞))", bool(is_mean_reverting)))

# st.write(pd.DataFrame(checks, columns=["Verifica", "Esito"]))

#st.markdown("---")
#st.subheader("Le Figure 1 e 2 di Lian & Haimes (2006) sono affette dall'errore in Eq. (43)?")
#st.markdown(r"""
#**No.** Le Figure 1 e 2 (Sez. 6.1 e 6.2) sono generate dall'Eq. (24)/(25) del
#paper, $q(t)=[I-e^{-K(I-A^*)t}](I-A^*)^{-1}c^*$, che dipende solo da $A^*$,
#$K$ e $c^*$ — **mai** da $\theta$. Sopra (§2) l'equilibrio calcolato con lo
#scenario "Fig. 1" riproduce quasi esattamente i valori citati nel testo
#(6.6% e 11.2%), confermando che $A^*$ e la dinamica non sono toccate
#dall'errore. $K$, in quegli esempi, è assegnato direttamente ($k_i=k_j=0.2$),
#non derivato da $\theta$.

#L'unico punto del paper in cui $\theta$ entra in un calcolo numerico è l'Eq.
#(45) (Sez. 6.3), che stima $k_i,k_j$ dai tempi di recupero per lo scenario
#**stocastico** (Figura 3): lì l'errore in $\theta$ **sì** avrebbe un impatto
#rilevante (con la correzione, $k_i$ scenderebbe da ≈0.090 a ≈0.016 e $k_j$ da
#≈0.081 a ≈0.004, cioè un recupero molto più lento di quello mostrato in
#Figura 3). Questo scenario non è però implementato in questa app, che simula
#la SDE mean-reverting generale, non il DIIM esatto di Sez. 6.3.
#""")
