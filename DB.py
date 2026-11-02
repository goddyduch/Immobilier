"""
Exemple complet PostgreSQL + pandas :
connexion, création d'une table depuis un DataFrame, ajout / suppression
d'une ligne, création d'une 2e table liée, lecture.

Prérequis :
    pip install pandas sqlalchemy psycopg2-binary
"""

import getpass

import pandas as pd
from sqlalchemy import create_engine, text

# --------------------------------------------------------------------------
# 1. Connexion
# --------------------------------------------------------------------------
# Avec Homebrew : utilisateur = ton nom macOS (whoami), pas de mot de passe.
USER = getpass.getuser()
DB = "immobilier"
engine = create_engine(f"postgresql+psycopg2://{USER}@localhost:5432/{DB}")

with engine.connect() as conn:
    print("Connecté :", conn.execute(text("SELECT version()")).scalar())


# --------------------------------------------------------------------------
# 2. Créer une table à partir d'un DataFrame
# --------------------------------------------------------------------------
def creer_table_depuis_df(df: pd.DataFrame, nom_table: str = "brute_annonces"):
    df = df.copy()
    df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
    for col in ["pieces", "nbr_chambres", "etage", "annee_construction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # if_exists : "replace" (recrée), "append" (ajoute), "fail" (erreur si existe)
    df.to_sql(nom_table, engine, if_exists="replace", index=False)

    # to_sql ne crée pas de clé primaire : on l'ajoute après
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {nom_table} ADD PRIMARY KEY (id)"))
    print(f"Table '{nom_table}' créée avec {len(df)} lignes.")


# --------------------------------------------------------------------------
# 3. Ajouter une ligne (ignorée si l'id existe déjà)
# --------------------------------------------------------------------------
def ajouter_annonce(annonce: dict):
    sql = text("""
        INSERT INTO annonces (id, titre, ville, code_postal, prix, surface_m2, pieces, url)
        VALUES (:id, :titre, :ville, :code_postal, :prix, :surface_m2, :pieces, :url)
        ON CONFLICT (id) DO NOTHING
    """)
    with engine.begin() as conn:  # begin() = commit automatique à la fin
        res = conn.execute(sql, annonce)
    print("Ligne ajoutée." if res.rowcount else "Id déjà présent, rien ajouté.")


# --------------------------------------------------------------------------
# 4. Supprimer une ligne
# --------------------------------------------------------------------------
def supprimer_annonce(annonce_id: str):
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM annonces WHERE id = :id"), {"id": annonce_id})
    print(f"{res.rowcount} ligne(s) supprimée(s).")


# --------------------------------------------------------------------------
# 5. Créer une autre table (en SQL), liée à la première
# --------------------------------------------------------------------------
def creer_table_favoris():
    sql = text("""
        CREATE TABLE IF NOT EXISTS favoris (
            id          SERIAL PRIMARY KEY,
            annonce_id  TEXT NOT NULL REFERENCES annonces(id) ON DELETE CASCADE,
            note        TEXT,
            cree_le     TIMESTAMPTZ DEFAULT now()
        )
    """)
    with engine.begin() as conn:
        conn.execute(sql)
    print("Table 'favoris' prête.")


def ajouter_favori(annonce_id: str, note: str = ""):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO favoris (annonce_id, note) VALUES (:a, :n)"),
            {"a": annonce_id, "n": note},
        )


# --------------------------------------------------------------------------
# 6. Lire / supprimer une table
# --------------------------------------------------------------------------
def lire(requete: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql(text(requete), engine, params=params)


def supprimer_table(nom_table: str):
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {nom_table} CASCADE"))
    print(f"Table '{nom_table}' supprimée.")


# --------------------------------------------------------------------------
# Démo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("annonces.csv")
    creer_table_depuis_df(df)

    """ajouter_annonce({
        "id": "test-001", "titre": "T2 de test", "ville": "Poitiers",
        "code_postal": "86000", "prix": 90000, "surface_m2": 30,
        "pieces": 2, "url": "https://exemple.fr/test-001",
    })"""

    #creer_table_favoris()
    #ajouter_favori("test-001", "à visiter samedi")

    #print(lire("SELECT id, titre, prix FROM annonces ORDER BY prix LIMIT 5"))
    #print(lire("SELECT * FROM favoris"))

    #supprimer_annonce("test-001")   # supprime aussi le favori lié (ON DELETE CASCADE)
    # supprimer_table("favoris")