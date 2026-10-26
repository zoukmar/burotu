<p align="center">
  <img src="docs/assets/onepager.png" alt="burotu.com" width="720">
</p>

# Burotu

**Détection multi-cancers à partir d'une prise de sang**, par la lecture de la méthylation de
l'ADN circulant, avec les modèles Meta Muse.
Projet du Meta Global AI Developer Hackathon 2026, track indie.

> [!IMPORTANT]
> Burotu est un **prototype de recherche**. Ce n'est pas un dispositif médical, il n'a pas de
> marquage CE et ne doit pas servir à un diagnostic. Le site ne vend rien : le paiement est en
> mode test.

## Comment ça marche

1. Une prise de sang est faite chez un préleveur partenaire.
2. Le laboratoire isole l'ADN circulant et séquence un panel ciblé de 1 692 sites CpG.
3. Il renvoie un fichier de comptages au format [`burotu-bmc`](src/burotu/labformat.py).
4. Un classifieur dédié détecte un signal de cancer et son organe d'origine probable.
5. **Muse Spark** (Meta Model API) contrôle la qualité des données et rédige le compte rendu
   pour le médecin.

Détails : [architecture](docs/architecture.md) · [données et méthode](docs/data.md) ·
[déploiement](docs/deployment.md)

## Démarrage rapide

```bash
make install        # Python 3.11+
make check          # lint + tests (comme la CI)
make dev            # http://localhost:8000 (site) et http://admin.localhost:8000 (suivi, mot de passe « dev »)
```

Pipeline de données (≈ 5 Go téléchargés depuis GDC et GEO) :

```bash
make data           # tumeurs TCGA, sang total, plasma réel
make reference      # panel ciblé et découpage train/test
make demo-samples   # fichiers labo simulés pour la démo
```

Toutes les commandes : `make help`.

## Configuration

| Variable | Rôle |
| --- | --- |
| `MODEL_API_KEY` | Clé Meta Model API. Absente : mode mock, tout fonctionne hors ligne |
| `META_MODEL` | Modèle Muse (défaut `muse-spark-1.3`) |
| `BUROTU_ADMIN_USER`, `BUROTU_ADMIN_PASSWORD` | Accès au suivi |
| `BUROTU_SECRET_KEY` | Signature des cookies de session |
| `BUROTU_PUBLIC_HOSTS`, `BUROTU_ADMIN_HOST` | Domaines du site et du suivi |

Voir [`.env.example`](.env.example).

## Structure

```
src/
  burotu/          pipeline : données, panel, simulateur, client Meta, format labo
  burotu_web/      burotu.com (one-pager EN/FR/ES) et admin.burotu.com (suivi)
tests/
  core/            pipeline
  web/             site, réservation, suivi, sécurité
deploy/            compose de production, bloc Caddy, script de déploiement
docs/              architecture, données, déploiement
.github/workflows  CI (lint, tests, image) et déploiement continu
```

## Déploiement

Chaque push sur `main` passe la CI, publie l'image `ghcr.io/zoukmar/burotu-web:<sha>` et la
déploie par SSH sur le serveur, avec retour automatique à la version précédente si le site ne
répond pas. Voir [docs/deployment.md](docs/deployment.md).

## Données utilisées

TCGA (portail GDC, accès ouvert) · Hannum et al., *Mol Cell* 2013 (GSE40279) ·
Moss et al., *Nat Commun* 2018 (GSE122126).
