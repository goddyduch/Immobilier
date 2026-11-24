import asyncio
import json
import csv
import re
import random
from pathlib import Path
from playwright.async_api import async_playwright

DOSSIER = Path("session_jinka")
DOSSIER.mkdir(exist_ok=True)

# --- AJOUT --- liste qui reçoit toutes les réponses JSON de la page
reponses_json = []

# --- AJOUT 2 --- dossier pour garder le HTML brut de chaque annonce
DOSSIER_HTML = Path("annonces_html")
DOSSIER_HTML.mkdir(exist_ok=True)


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


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(DOSSIER.resolve()),
            headless=False,
        )

        page = context.pages[0] if context.pages else await context.new_page()

        # --- AJOUT --- écoute des appels réseau (avant la navigation)
        async def on_response(response):
            if response.request.resource_type in ("xhr", "fetch"):
                try:
                    data = await response.json()
                except Exception:
                    return
                print("JSON reçu :", response.url)
                reponses_json.append({"url": response.url, "data": data})

        page.on("response", on_response)

        # Ouvrir Jinka
        await page.goto(
            "https://www.jinka.fr/",
            wait_until="domcontentloaded"
        )

        #await page.wait_for_load_state("networkidle")
        # ou : await page.wait_for_selector("css=.ton-selecteur")

        #html = await page.content()
        #Path("jinka.html").write_text(html, encoding="utf-8")
        async with page.expect_navigation():
            await page.get_by_text("Ma Recherche").click()


        print("Navigation réussie vers la page de recherche !")

        # --- AJOUT --- attendre le chargement des annonces
        await page.wait_for_load_state("networkidle")

        # --- AJOUT --- défilement pour charger toutes les annonces
        hauteur_precedente = 0
        for _ in range(40):
            await page.mouse.wheel(0, 5000)
            await page.wait_for_timeout(1500)
            hauteur = await page.evaluate("document.documentElement.scrollHeight")
            if hauteur == hauteur_precedente:
                break
            hauteur_precedente = hauteur

        html = await page.content()
        Path("jinkarecherche.html").write_text(html, encoding="utf-8")

        # --- AJOUT --- sauvegarde de toutes les réponses JSON
        Path("jinka_reponses.json").write_text(
            json.dumps(reponses_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{len(reponses_json)} réponses JSON sauvegardées dans jinka_reponses.json")

        # --- AJOUT --- extraction générique : tous les liens de la page
        liens = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({texte: e.innerText.trim(), href: e.href}))",
        )
        Path("jinka_liens.json").write_text(
            json.dumps(liens, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{len(liens)} liens sauvegardés dans jinka_liens.json")

        # --- AJOUT 2 --- ne garder que les liens d'annonces (sans doublons)
        annonces_liens = []
        vus = set()
        for l in liens:
            if "jinka.fr/ad/" in l["href"] and l["href"] not in vus:
                vus.add(l["href"])
                annonces_liens.append(l)
        print(f"{len(annonces_liens)} annonces trouvées")
        Path("jinka_liens_annonces.json").write_text(
            json.dumps(annonces_liens, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- AJOUT 2 --- ouvrir uniquement les NOUVELLES annonces non présentes en base/cache
        existantes_par_id = {}
        if Path("jinka_annonces.json").exists():
            try:
                anciennes = json.loads(Path("jinka_annonces.json").read_text(encoding="utf-8"))
                for a in anciennes:
                    if a.get("id"):
                        existantes_par_id[str(a["id"])] = a
            except Exception:
                pass

        try:
            from DB import get_ids_connus
            ids_connus_db = get_ids_connus(source="jinka")
        except Exception:
            ids_connus_db = set()

        toutes_annonces = []
        nb_nouvelles = 0
        nb_reutilisees = 0

        for i, l in enumerate(annonces_liens, start=1):
            m_id = re.search(r"/ad/([^?]+)", l["href"])
            ad_id = m_id.group(1) if m_id else None
            carte_actuelle = parser_carte(l["texte"])

            # Si l'annonce est déjà connue, on ne ré-ouvre pas la page Playwright !
            if ad_id and ad_id in existantes_par_id:
                details = existantes_par_id[ad_id]
                details["carte"] = carte_actuelle  # met à jour le prix/infos de la carte
                toutes_annonces.append(details)
                nb_reutilisees += 1
                print(f"[{i}/{len(annonces_liens)}] Déjà connue (ignorée) : {ad_id}")
                continue

            print(f"[{i}/{len(annonces_liens)}] NOUVELLE annonce -> extraction : {l['href']}")
            details = await extraire_annonce(context, l["href"])
            details["carte"] = carte_actuelle
            toutes_annonces.append(details)
            nb_nouvelles += 1
            await asyncio.sleep(random.uniform(2, 5))  # pause pour ne pas surcharger le site

        print(f"Bilan Jinka : {nb_nouvelles} nouvelles extraites, {nb_reutilisees} déjà connues toujours actives.")

        Path("jinka_annonces.json").write_text(
            json.dumps(toutes_annonces, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- AJOUT 2 --- export CSV (une ligne par annonce)
        lignes_csv = []
        for a in toutes_annonces:
            ligne = {"id": a.get("id"), "url": a["url"], "titre": a.get("titre")}
            ligne.update(a.get("carte", {}))
            for k, v in (a.get("caracteristiques") or {}).items():
                ligne[f"carac_{k}"] = v
            ligne["liens_externes"] = " | ".join(a.get("liens_externes", []))
            ligne["erreur"] = a.get("erreur")
            lignes_csv.append(ligne)

        colonnes = []
        for ligne in lignes_csv:
            for k in ligne:
                if k not in colonnes:
                    colonnes.append(k)
        Path("csv").mkdir(exist_ok=True)
        with open("csv/jinka_annonces.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=colonnes, delimiter=";")
            w.writeheader()
            w.writerows(lignes_csv)
        print(f"{len(toutes_annonces)} annonces sauvegardées (jinka_annonces.json / csv/jinka_annonces.csv)")

        input("Appuyez sur Entrée dans le terminal pour fermer le navigateur...")


if __name__ == "__main__":
    asyncio.run(main())