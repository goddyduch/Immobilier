"""
Pipeline légère de traitement des photographies immobilières (`Photos.py`).

Conçue pour tourner sur CPU (iMac Intel i5 / 8 Go RAM / espace disque restreint) :
1. Télécharge jusqu'à 4 photos par annonce et les compresse à la volée en JPEG 384x384 (~25-35 Ko/photo).
2. Déduplique via hash MD5 et enregistre les métadonnées dans PostgreSQL (`annonces_photos` / `csv/annonces_photos.csv`).
3. Extrait des features visuelles objectives rapides (Pillow + NumPy, ~10 ms/image) :
   - `vis_nb_photos`   : nombre de photos disponibles sur l'annonce
   - `vis_luminosite`  : luminosité moyenne des pièces (0–255)
   - `vis_contraste`   : contraste moyen des clichés
   - `vis_nettete`     : indice de netteté / qualité de prise de vue
4. Optionnel : si un mini-VLM (`moondream`) est installé dans Ollama, évalue l'état intérieur
   et les travaux visibles en JSON structuré (mis en cache dans `photos_features_cache.json`).

Usage :
    python Photos.py
"""

import hashlib
import io
import json
import os
import numpy as np
import pandas as pd
import requests
from PIL import Image

from src.ingestion import db as DB

DOSSIER_PHOTOS = "photos_cache"
CACHE_VISUEL_PATH = "photos_features_cache.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
_CLIP_STATE: dict | None = None


def _charger_clip():
    """Charge paresseusement OpenCLIP (ViT-B-32) et pré-encode les prompts travaux/rénové."""
    global _CLIP_STATE
    if _CLIP_STATE is not None:
        return _CLIP_STATE
    try:
        import open_clip
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        prompts_5_etats = [
            "very degraded dilapidated room or derelict building needing heavy structural renovation",
            "old dated worn interior from the 1970s needing full renovation",
            "average lived-in interior needing light refresh painting and flooring",
            "clean well maintained apartment or house interior in good condition",
            "brand new modern luxury renovated bright interior",
        ]
        with torch.no_grad():
            tok_5 = tokenizer(prompts_5_etats).to(device)
            emb_5 = model.encode_text(tok_5)
            emb_5 = emb_5 / emb_5.norm(dim=-1, keepdim=True)

        _CLIP_STATE = {
            "ok": True,
            "torch": torch,
            "device": device,
            "model": model,
            "preprocess": preprocess,
            "etats_5_feats": emb_5,
        }
    except Exception as e:
        print(f"[Photos CLIP] Indisponible ({e}), utilisation des embeddings visuels légers uniquement.")
        _CLIP_STATE = {"ok": False}
    return _CLIP_STATE


def _pil_to_clip_tensor(im: Image.Image, torch_mod):
    """Convertit une image PIL en tenseur CLIP 224x224 normalisé sans passer par torch.from_numpy."""
    from PIL import ImageOps
    im224 = ImageOps.fit(im.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
    t = torch_mod.tensor(bytearray(im224.tobytes()), dtype=torch_mod.uint8).view(224, 224, 3).permute(2, 0, 1).float() / 255.0
    mean = torch_mod.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch_mod.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return (t - mean) / std


def scorer_photos_clip(chemins_imgs: list[str]) -> dict | None:
    """Calcule les scores zero-shot CLIP (travaux vs rénové) sur une liste de photos locales."""
    st = _charger_clip()
    if not st or not st.get("ok") or not chemins_imgs:
        return None
    torch = st["torch"]
    tensors = []
    for p in chemins_imgs[:4]:
        try:
            with Image.open(p) as im:
                tensors.append(_pil_to_clip_tensor(im, torch))
        except Exception:
            continue
    if not tensors:
        return None
    try:
        batch = torch.stack(tensors).to(st["device"])
        with torch.no_grad():
            img_feats = st["model"].encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits_5 = (img_feats @ st["etats_5_feats"].T) * 20.0
            probs_5 = torch.softmax(logits_5, dim=-1).mean(dim=0).cpu().tolist()
            etat_5 = sum((idx + 1) * probs_5[idx] for idx in range(5))
            score_trav = probs_5[0] + probs_5[1] + probs_5[2]
            score_renov = probs_5[3] + probs_5[4]

        return {
            "clip_etat_score": round(float(etat_5), 3),
            "clip_score_travaux": round(float(score_trav), 4),
            "clip_score_renove": round(float(score_renov), 4),
        }
    except Exception:
        return None


def _calculer_nettete_numpy(arr_gray: np.ndarray) -> float:
    """Estime la netteté d'une image par variance du Laplacien discret (NumPy pur)."""
    if arr_gray.shape[0] < 3 or arr_gray.shape[1] < 3:
        return 0.0
    lap = (
        arr_gray[:-2, 1:-1]
        + arr_gray[2:, 1:-1]
        + arr_gray[1:-1, :-2]
        + arr_gray[1:-1, 2:]
        - 4.0 * arr_gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def analyser_miniature_locale(chemin_img: str) -> dict | None:
    """Calcule luminosité, contraste, netteté, saturation et un embedding visuel compact 32D (sans VLM)."""
    try:
        with Image.open(chemin_img) as img:
            rgb = img.convert("RGB")
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
            # Embedding spatial léger 4x4 RGB moyen + histogramme teinte/saturation (32D)
            small = np.asarray(rgb.resize((4, 4), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            grid_rgb = small.reshape(-1)  # 48 valeurs -> on réduit par blocs de 3 canaux moyens par quadrant (2x2x3=12)
            q2 = np.asarray(rgb.resize((2, 2), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            hist_s, _ = np.histogram(hsv[:, :, 1].ravel(), bins=4, range=(0, 255), density=True)
            hist_v, _ = np.histogram(hsv[:, :, 2].ravel(), bins=4, range=(0, 255), density=True)
            emb = np.concatenate([q2.reshape(-1), hist_s * 50.0, hist_v * 50.0]).astype(float)
            # Projection déterministe orthogonale vers 5 composantes compactes
            proj = np.array([
                float(np.mean(emb[0:4])),
                float(np.mean(emb[4:8])),
                float(np.mean(emb[8:12])),
                float(np.mean(emb[12:16])),
                float(np.mean(emb[16:20])),
            ])
        return {
            "luminosite": float(np.mean(gray)),
            "contraste": float(np.std(gray)),
            "nettete": _calculer_nettete_numpy(gray),
            "saturation": float(np.mean(hsv[:, :, 1])),
            "emb": [round(x, 4) for x in proj],
        }
    except Exception:
        return None


def telecharger_miniature(
    url: str,
    dest_path: str,
    taille: tuple[int, int] = (384, 384),
    qualite: int = 75,
) -> str | None:
    """Télécharge une photo, la redimensionne en mémoire en JPEG compressé et retourne son hash MD5."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500:
        with open(dest_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        r.raise_for_status()
        with Image.open(io.BytesIO(r.content)) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail(taille, Image.Resampling.LANCZOS)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            img_rgb.save(dest_path, format="JPEG", quality=qualite, optimize=True)
        with open(dest_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def synchroniser_photos(
    df: pd.DataFrame,
    max_annonces: int = 120,
    max_photos_par_annonce: int = 4,
    telecharger: bool = True,
) -> pd.DataFrame:
    """
    Télécharge les miniatures compressées pour les annonces disposant de `photos_urls`
    et sauvegarde la table `annonces_photos` (PostgreSQL + `csv/annonces_photos.csv`).
    """
    os.makedirs(DOSSIER_PHOTOS, exist_ok=True)
    lignes = []
    traitees = 0

    for _, row in df.iterrows():
        ad_id = str(row.get("id", "")).strip()
        raw_urls = str(row.get("photos_urls", "") or "").strip()
        if not ad_id or not raw_urls or raw_urls == "nan":
            continue
        urls = [u.strip() for u in raw_urls.split("|") if u.strip().startswith("http")][:max_photos_par_annonce]
        if not urls:
            continue

        for idx, u in enumerate(urls):
            dest = os.path.join(DOSSIER_PHOTOS, ad_id, f"{idx}.jpg")
            md5 = None
            if os.path.exists(dest) and os.path.getsize(dest) > 500:
                with open(dest, "rb") as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
            elif telecharger and traitees < max_annonces:
                md5 = telecharger_miniature(u, dest)

            if md5:
                lignes.append({
                    "id": f"{ad_id}_{idx}",
                    "annonce_id": ad_id,
                    "ordre": idx,
                    "url": u,
                    "chemin_local": dest,
                    "hash_md5": md5,
                    "taille_ko": round(os.path.getsize(dest) / 1024.0, 1),
                })
        traitees += 1

    df_photos = pd.DataFrame(lignes)
    if not df_photos.empty:
        df_photos.to_csv(DB.CSV_PHOTOS, index=False)
        DB.ecrire_table_sql(df_photos, DB.TABLE_PHOTOS)
    return df_photos


def _vlm_disponible(vlm_model: str = "moondream") -> bool:
    try:
        import ollama
        modeles = [m.get("model", "") for m in ollama.list().get("models", [])]
        return any(vlm_model in m for m in modeles)
    except Exception:
        return False


def extraire_features_visuelles(
    df: pd.DataFrame,
    cache_path: str = CACHE_VISUEL_PATH,
    utiliser_vlm: bool = True,
    vlm_model: str = "moondream",
    max_appels_vlm: int = 30,
) -> pd.DataFrame:
    """
    Enrichit `df` avec les features visuelles issues des miniatures locales :
    - `vis_nb_photos`   : nombre de photos sur l'annonce
    - `vis_luminosite`  : luminosité moyenne (0–255)
    - `vis_contraste`   : contraste moyen
    - `vis_nettete`     : netteté moyenne
    - `vis_etat_vlm`    : note état visuel (1–5) si `moondream` est installé dans Ollama
    """
    df = df.copy()
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    vlm_ok = _vlm_disponible(vlm_model) if utiliser_vlm else False
    appels_vlm = 0
    modifies = 0
    resultats = []

    for _, row in df.iterrows():
        ad_id = str(row.get("id", "")).strip()
        raw_urls = str(row.get("photos_urls", "") or "").strip()
        nb_urls = len([u for u in raw_urls.split("|") if u.strip().startswith("http")]) if raw_urls and raw_urls != "nan" else 0

        dossier_ad = os.path.join(DOSSIER_PHOTOS, ad_id)
        imgs_locales = (
            sorted([os.path.join(dossier_ad, f) for f in os.listdir(dossier_ad) if f.endswith(".jpg")])
            if os.path.isdir(dossier_ad)
            else []
        )

        entree = cache.get(ad_id, {})
        if (not entree or "vis_emb_0" not in entree) and imgs_locales:
            stats = [s for p in imgs_locales if (s := analyser_miniature_locale(p)) is not None]
            if stats:
                embs = np.array([s["emb"] for s in stats if "emb" in s], dtype=float)
                emb_mean = np.mean(embs, axis=0) if len(embs) else np.zeros(5)
                entree.update({
                    "vis_luminosite": round(float(np.mean([s["luminosite"] for s in stats])), 1),
                    "vis_contraste": round(float(np.mean([s["contraste"] for s in stats])), 1),
                    "vis_nettete": round(float(np.mean([s["nettete"] for s in stats])), 1),
                    "vis_saturation": round(float(np.mean([s.get("saturation", 0.0) for s in stats])), 1),
                    "vis_emb_0": round(float(emb_mean[0]), 4),
                    "vis_emb_1": round(float(emb_mean[1]), 4),
                    "vis_emb_2": round(float(emb_mean[2]), 4),
                    "vis_emb_3": round(float(emb_mean[3]), 4),
                    "vis_emb_4": round(float(emb_mean[4]), 4),
                })
                modifies += 1

        if imgs_locales and "clip_etat_score" not in entree:
            clip_res = scorer_photos_clip(imgs_locales)
            if clip_res:
                entree.update(clip_res)
                modifies += 1

        if vlm_ok and imgs_locales and "vis_etat_vlm" not in entree and appels_vlm < max_appels_vlm:
            try:
                import ollama
                resp = ollama.chat(
                    model=vlm_model,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Rate the interior condition of this real estate photo from 1 (needs full renovation) "
                            "to 5 (brand new / modern) and reply ONLY with a JSON: {\"etat\": 3}"
                        ),
                        "images": [imgs_locales[0]],
                    }],
                )
                parsed = json.loads(resp["message"]["content"])
                entree["vis_etat_vlm"] = int(np.clip(int(parsed.get("etat", 3)), 1, 5))
                appels_vlm += 1
                modifies += 1
            except Exception:
                pass

        if entree:
            cache[ad_id] = entree

        resultats.append({
            "vis_nb_photos": nb_urls if nb_urls > 0 else (len(imgs_locales) if imgs_locales else np.nan),
            "vis_luminosite": entree.get("vis_luminosite", np.nan),
            "vis_contraste": entree.get("vis_contraste", np.nan),
            "vis_nettete": entree.get("vis_nettete", np.nan),
            "vis_saturation": entree.get("vis_saturation", np.nan),
            "vis_etat_vlm": entree.get("vis_etat_vlm", np.nan),
            "vis_emb_0": entree.get("vis_emb_0", np.nan),
            "vis_emb_1": entree.get("vis_emb_1", np.nan),
            "vis_emb_2": entree.get("vis_emb_2", np.nan),
            "clip_etat_score": entree.get("clip_etat_score", np.nan),
            "clip_score_travaux": entree.get("clip_score_travaux", np.nan),
            "clip_score_renove": entree.get("clip_score_renove", np.nan),
        })

    if modifies > 0:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    res_df = pd.DataFrame(resultats, index=df.index)
    for col in res_df.columns:
        df[col] = res_df[col]
    return df


if __name__ == "__main__":
    df_hist = DB.lire_table_ou_csv(DB.TABLE_HIST_BRUT, DB.CSV_HIST_BRUT)
    if not df_hist.empty:
        df_p = synchroniser_photos(df_hist, max_annonces=80, telecharger=True)
        print(f"[Photos] {len(df_p)} miniatures indexées dans '{DB.TABLE_PHOTOS}' ({DOSSIER_PHOTOS}/).")
    else:
        print("Aucune annonce trouvée pour la synchronisation photo.")
