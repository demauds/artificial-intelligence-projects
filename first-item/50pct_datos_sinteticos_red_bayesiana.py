import pandas as pd
from sklearn.model_selection import train_test_split

# pgmpy nueva API
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, PC
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
# 1️⃣ Cargar dataset original
df = pd.read_csv("first-item/covid_flu_cold_symptoms.csv")  # adapta el path

# 2️⃣ Generar datos sintéticos (50% más)
n_sinteticas = int(len(df) * 0.5)
df_sintetico = df.sample(n=n_sinteticas, replace=True, random_state=42)

# 3️⃣ Combinar dataset original + sintético
df_ampliado = pd.concat([df, df_sintetico], ignore_index=True)

# 4️⃣ Guardar dataset ampliado en un archivo nuevo
df_ampliado.to_csv("first-item/covid_flu_cold_symptoms_ampliado.csv", index=False)

print(f"Tamaño original: {len(df)}, tamaño ampliado: {len(df_ampliado)}")
print("Archivo ampliado creado: covid_flu_cold_symtoms_ampliado.csv")

df = pd.read_csv("first-item/covid_flu_cold_symptoms_ampliado.csv")  # adapta el path

use_cols = ["FEVER", "COUGH", "LOSS_OF_TASTE", "LOSS_OF_SMELL", "SHORTNESS_OF_BREATH", "TYPE"]
df = df[use_cols].copy()

# Convertir síntomas a binario string
for col in df.columns:
    if col != "TYPE":
        df[col] = df[col].fillna(0).astype(int).astype(str)

# Filtrar categorías válidas
df["TYPE"] = df["TYPE"].fillna("Unknown").astype(str)
df = df[df["TYPE"].isin(["FLU", "COLD", "COVID", "ALLERGY"])]

# ---------- 2) SPLIT 70/30 ----------
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["TYPE"])

# ---------- 3) HillClimbSearch ----------
hc = HillClimbSearch(train_df)
score = BIC(train_df)
hc_model = hc.estimate(scoring_method=score, max_indegree=3)
print("HillClimb edges:", hc_model.edges())

# ---------- 4) PC ----------
pc = PC(data=train_df)
pc_dag = pc.estimate(ci_test="chi_square", significance_level=0.01, return_type="dag", n_jobs=1)
print("PC edges:", pc_dag.edges())

# ---------- 5) Ajuste con DiscreteBayesianNetwork ----------
# HillClimb
hc_edges = list(hc_model.edges())
hc_bm = DiscreteBayesianNetwork(hc_edges)
hc_bm.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
hc_infer = VariableElimination(hc_bm)

# PC
pc_edges = list(pc_dag.edges())
pc_bm = DiscreteBayesianNetwork(pc_edges)
pc_bm.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
pc_infer = VariableElimination(pc_bm)

# ---------- Helper: función de consulta ----------
def query_prob(infer, model, query_var, evidence):
    """
    infer: VariableElimination
    model: DiscreteBayesianNetwork (fitted)
    query_var: string
    evidence: dict, e.g. {'SHORTNESS_OF_BREATH':'1'}
    Devuelve dict {estado: probabilidad}
    """
    factor = infer.query([query_var], evidence=evidence, show_progress=False)
    states = factor.state_names[query_var]
    probs = factor.values
    return dict(zip(states, probs))

# ---------- 6) Consultas solicitadas ----------
print("\n--- Inferencias Hill Climbing ---")
hc_queries = [
    ('TYPE', {'SHORTNESS_OF_BREATH':'1'}, 'FLU'),
    ('TYPE', {'LOSS_OF_SMELL':'1'}, 'COLD'),
    ('TYPE', {'LOSS_OF_TASTE':'1'}, 'COVID'),
    ('FEVER', {'TYPE':'ALLERGY'}, '0')
]

for var, evidence, target_state in hc_queries:
    res = query_prob(hc_infer, hc_bm, var, evidence)
    print(f"{var} | {evidence} -> {target_state}: {res.get(target_state, None):.4f}, full dist: {res}")

print("\n--- Inferencias PC ---")
pc_queries = [
    ('TYPE', {'SHORTNESS_OF_BREATH':'1'}, 'ALLERGY'),
    ('LOSS_OF_SMELL', {'TYPE':'FLU'}, '1'),
    ('TYPE', {'LOSS_OF_TASTE':'0'}, 'COVID'),
    ('TYPE', {'FEVER':'1'}, 'COLD')
]

for var, evidence, target_state in pc_queries:
    res = query_prob(pc_infer, pc_bm, var, evidence)
    print(f"{var} | {evidence} -> {target_state}: {res.get(target_state, None):.4f}, full dist: {res}")

    
# ---------- 7) Inferencias sobre test_df ----------
print("\n=== Inferencias usando test_df ===")

# Hill Climb sobre test_df
hc_bm_test = DiscreteBayesianNetwork(hc_edges)
hc_bm_test.fit(test_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
hc_infer_test = VariableElimination(hc_bm_test)

# PC sobre test_df
pc_bm_test = DiscreteBayesianNetwork(pc_edges)
pc_bm_test.fit(test_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
pc_infer_test = VariableElimination(pc_bm_test)

# Consultas Hill Climb con test_df
print("\n--- Inferencias Hill Climbing (test_df) ---")
for var, evidence, target_state in hc_queries:
    res = query_prob(hc_infer_test, hc_bm_test, var, evidence)
    print(f"{var} | {evidence} -> {target_state}: {res.get(target_state, None):.4f}, full dist: {res}")

# Consultas PC con test_df
print("\n--- Inferencias PC (test_df) ---")
for var, evidence, target_state in pc_queries:
    res = query_prob(pc_infer_test, pc_bm_test, var, evidence)
    print(f"{var} | {evidence} -> {target_state}: {res.get(target_state, None):.4f}, full dist: {res}")
