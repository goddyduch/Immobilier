import argparse
import asyncio
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from playwright.async_api import async_playwright

from src.ingestion import db as DB
from src.ingestion.territoires import get_territoire

DOSSIER = Path("session_jinka")
DOSSIER.mkdir(exist_ok=True)

reponses_json = []

DOSSIER_HTML = Path("annonces_html")
DOSSIER_HTML.mkdir(exist_ok=True)


def purger_html_anciens(jours_ttl: int = 7) -> int:
    """RGPD & stockage : purge les fichiers HTML bruts de plus de `jours_ttl` jours."""
    seuil = time.time() - jours_ttl * 86400
    nb = 0
    for f in DOSSIER_HTML.glob("*.html"):
        try:
            if f.stat().st_mtime < seuil:
                f.unlink()
                nb += 1
        except Exception:
            pass
    return nb


# --- AJOUT 2 --- lecture des infos visibles sur la carte de la liste
def parser_carte(texte_carte: str) -> dict:
    lignes = [l.strip() for l in texte_carte.split("\n") if l.strip()]
    info = {"source": lignes[0] if lignes else None}

    m = re.search(r"([\d\s\u202f\u00a0]+)\s*€(?!\s*/)", texte_carte)
    info["prix"] = re.sub(r"\D", "", m.group(1)) if m else None

    m = re.search(r"([\d\s\u202f\u00a0]+)\s*€\s*/\s*m²", texte_carte)
    info["prix_m2"] = re.sub(r"\D", "", m.group(1)) if m else None

    m = re.search(r"(\d+)\s*pièces?", texte_carte)
    info["pieces"] = m.group(1) if m else None

    m = re.search(r"(\d+)\s*Ch\.", texte_carte)
    info["chambres"] = m.group(1) if m else None

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m²\s*$", texte_carte.strip())
    info["surface_m2"] = m.group(1) if m else None

    # ligne du quartier / ville : celle qui suit "... - Agence"
    for i, l in enumerate(lignes):
        if "/ m²" in l and i + 1 < len(lignes):
            info["localisation"] = lignes[i + 1]
            break

    m = re.search(r"\b([A-G])\n", texte_carte)  # DPE
    info["dpe"] = m.group(1) if m else None
    return info


# --- AJOUT 2 --- extraction de toutes les caractéristiques d'une page d'annonce
async def extraire_annonce(context, url: str) -> dict:
    page = await context.new_page()
    resultat = {"url": url}
    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1500)

        # id de l'annonce (partie après /ad/)
        m = re.search(r"/ad/([^?]+)", url)
        ad_id = m.group(1) if m else str(random.randint(0, 10**9))
        resultat["id"] = ad_id

        # HTML brut sauvegardé pour analyse ultérieure
        html = await page.content()
        (DOSSIER_HTML / f"{ad_id}.html").write_text(html, encoding="utf-8")

        # titre
        h1 = page.locator("h1")
        resultat["titre"] = (await h1.first.inner_text()).strip() if await h1.count() else None

        # tout le texte visible de la page
        resultat["texte_complet"] = (await page.inner_text("body")).strip()

        # paires libellé / valeur (listes de définitions, tableaux)
        resultat["caracteristiques"] = await page.evaluate("""
            () => {
                const out = {};
                document.querySelectorAll('dt').forEach(dt => {
                    const dd = dt.nextElementSibling;
                    if (dd) out[dt.innerText.trim()] = dd.innerText.trim();
                });
                document.querySelectorAll('tr').forEach(tr => {
                    const c = tr.querySelectorAll('th,td');
                    if (c.length === 2) out[c[0].innerText.trim()] = c[1].innerText.trim();
                });
                return out;
            }
        """)

        # données structurées éventuelles (JSON-LD)
        resultat["json_ld"] = await page.evaluate("""
            () => [...document.querySelectorAll('script[type="application/ld+json"]')]
                    .map(s => { try { return JSON.parse(s.textContent); } catch(e) { return null; } })
                    .filter(Boolean)
        """)

        # photos
        resultat["images"] = await page.eval_on_selector_all(
            "img[src]",
            "els => [...new Set(els.map(e => e.src))].filter(s => s.startsWith('http'))",
        )

        # liens externes (souvent le lien vers l'annonce d'origine)
        resultat["liens_externes"] = await page.eval_on_selector_all(
            "a[href]",
            "els => [...new Set(els.map(e => e.href))].filter(h => h.startsWith('http') && !h.includes('jinka.fr'))",
        )
    except Exception as e:
        resultat["erreur"] = str(e)
    finally:
        await page.close()
    return resultat


def _lire_url_jinka_territoire(code_territoire: str) -> str | None:
    """Lit `jinka_search_url` depuis PostgreSQL `ref_territoires` ou `JINKA_URLS_JSON`."""
    urls_env = os.getenv("JINKA_URLS_JSON")
    if urls_env:
        try:
            mapping = json.loads(urls_env)
            if code_territoire in mapping:
                return mapping[code_territoire]
        except Exception:
            pass
    if DB.db_disponible():
        try:
            with DB.engine.connect() as conn:
                u = conn.execute(
                    DB.text("SELECT jinka_search_url FROM ref_territoires WHERE code_territoire = :c"),
                    {"c": code_territoire},
                ).scalar()
                if u:
                    return str(u)
        except Exception:
            pass
    return None


def _detecter_blocage_page(html: str) -> str | None:
    """Détecte un CAPTCHA, Cloudflare, 403/429 ou blocage anti-bot sans tenter de le contourner."""
    low = (html or "").lower()
    for motif in ("captcha", "cloudflare", "verify you are human", "access denied", "too many requests"):
        if motif in low:
            return f"Blocage détecté ({motif})"
    return None


async def run_jinka(
    code_territoire: str = "poitiers",
    dry_run: bool = False,
    max_annonces: int | None = None,
    headless: bool | None = None,
):
    cfg = get_territoire(code_territoire)
    if not cfg:
        raise ValueError(f"Territoire inconnu : {code_territoire}")

    if not dry_run and DB.est_en_cooldown("jinka", cfg.code_territoire):
        print(f"[Jinka] {cfg.nom} en cooldown (3h) -> passage ignoré.")
        return []

    purger_html_anciens(jours_ttl=7)
    is_headless = (os.getenv("JINKA_HEADLESS", "1") == "1") if headless is None else headless
    search_url = _lire_url_jinka_territoire(cfg.code_territoire)
    if not search_url and cfg.code_territoire != "poitiers":
        print(f"[Jinka] Aucune `jinka_search_url` configurée pour {cfg.nom} -> Jinka ignoré pour cette ville.")
        return []

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(DOSSIER.resolve()),
            headless=is_headless,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        async def on_response(response):
            if response.request.resource_type in ("xhr", "fetch"):
                try:
                    data = await response.json()
                except Exception:
                    return
                reponses_json.append({"url": response.url, "data": data})

        page.on("response", on_response)

        try:
            if search_url:
                await page.goto(search_url, wait_until="domcontentloaded")
            else:
                await page.goto("https://www.jinka.fr/", wait_until="domcontentloaded")
                async with page.expect_navigation():
                    await page.get_by_text("Ma Recherche").click()

            await page.wait_for_load_state("networkidle")
            blocage = _detecter_blocage_page(await page.content())
            if blocage:
                DB.enregistrer_blocage_scrape("jinka", cfg.code_territoire, blocage, cooldown_hours=3)
                await context.close()
                return []

            print(f"[Jinka {cfg.nom}] Navigation réussie vers la page de recherche !")
            max_scrolls = 3 if dry_run else 40
            hauteur_precedente = 0
            for _ in range(max_scrolls):
                await page.mouse.wheel(0, 5000)
                await page.wait_for_timeout(1500)
                hauteur = await page.evaluate("document.documentElement.scrollHeight")
                if hauteur == hauteur_precedente:
                    break
                hauteur_precedente = hauteur

            liens = await page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => ({texte: e.innerText.trim(), href: e.href}))",
            )
            annonces_liens = []
            vus = set()
            for l in liens:
                if "jinka.fr/ad/" in l["href"] and l["href"] not in vus:
                    vus.add(l["href"])
                    annonces_liens.append(l)
            if max_annonces:
                annonces_liens = annonces_liens[:max_annonces]
            print(f"[Jinka {cfg.nom}] {len(annonces_liens)} annonces ciblées.")

            ids_connus_db = DB.get_ids_connus(source="jinka", code_territoire=cfg.code_territoire)
            toutes_annonces = []
            nb_nouvelles = 0

            for i, l in enumerate(annonces_liens, start=1):
                m_id = re.search(r"/ad/([^?]+)", l["href"])
                ad_id = m_id.group(1) if m_id else None
                carte_actuelle = parser_carte(l["texte"])

                if not dry_run and ad_id and ad_id in ids_connus_db:
                    continue

                print(f"[{i}/{len(annonces_liens)}] Extraction Jinka : {l['href']}")
                details = await extraire_annonce(context, l["href"])
                details["carte"] = carte_actuelle
                details["code_territoire"] = cfg.code_territoire
                toutes_annonces.append(details)
                nb_nouvelles += 1
                await asyncio.sleep(random.uniform(2, 4))

            if dry_run:
                print(json.dumps(toutes_annonces, ensure_ascii=False, indent=2)[:1500])
                await context.close()
                return toutes_annonces

            Path("csv").mkdir(exist_ok=True)
            out_json = Path(f"csv/jinka_{cfg.code_territoire}.json")
            out_json.write_text(json.dumps(toutes_annonces, ensure_ascii=False, indent=2), encoding="utf-8")
            DB.enregistrer_succes_scrape("jinka", cfg.code_territoire, len(annonces_liens), scrape_complet=True)
            await context.close()
            return toutes_annonces
        except Exception as e:
            DB.enregistrer_blocage_scrape("jinka", cfg.code_territoire, str(e), cooldown_hours=3)
            await context.close()
            return []


async def main():
    parser = argparse.ArgumentParser(description="Scraper Jinka multi-villes")
    parser.add_argument("--ville", default="poitiers", help="Code territoire cible")
    parser.add_argument("--dry-run", action="store_true", help="Test léger sans écriture DB")
    parser.add_argument("--max-annonces", type=int, default=None, help="Limite d'annonces à extraire")
    parser.add_argument("--visible", action="store_true", help="Afficher la fenêtre Chromium")
    args, _ = parser.parse_known_args()
    await run_jinka(
        code_territoire=args.ville,
        dry_run=args.dry_run,
        max_annonces=args.max_annonces,
        headless=False if args.visible else None,
    )


if __name__ == "__main__":
    asyncio.run(main())