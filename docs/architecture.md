# Architecture

## Vue d'ensemble

```
 Patient ── prise de sang ──► Préleveur partenaire ──► Laboratoire
                                                         │ séquençage bisulfite ciblé
                                                         ▼
                                              fichier burotu-bmc (comptages par CpG)
                                                         │
                       ┌─────────────────────────────────┴──────────────────────┐
                       │ Burotu                                                  │
                       │  1. Contrôle qualité ......... Muse Spark (Meta)        │
                       │  2. Signal cancer ............ classifieur dédié        │
                       │  3. Organe d'origine ......... classifieur dédié        │
                       │  4. Compte rendu médecin ..... Muse Spark (Meta)        │
                       └─────────────────────────────────┬──────────────────────┘
                                                         ▼
                                                      Médecin
```

Le classifieur est un modèle tabulaire entraîné sur des profils de méthylation : un modèle de
langage n'est pas fait pour classer des centaines de milliers de valeurs numériques. Les modèles
Meta interviennent là où le langage compte : vérifier qu'un fichier de laboratoire est
exploitable, expliquer un résultat à un médecin, répondre à ses questions.

## Composants du dépôt

| Dossier | Rôle |
| --- | --- |
| `src/burotu/data` | Téléchargement TCGA (GDC) et GEO, construction du panel ciblé |
| `src/burotu/simulate` | Simulateur d'ADN circulant à partir de profils réels |
| `src/burotu/llm` | Client Meta Model API (compatible OpenAI), mode mock sans clé |
| `src/burotu/labformat.py` | Format d'échange `burotu-bmc` avec le laboratoire |
| `src/burotu_web` | burotu.com (one-pager, réservation) et admin.burotu.com (suivi) |
| `deploy/` | Compose de production, bloc Caddy, script de déploiement |

## Site et suivi

Un seul processus FastAPI, deux sites aiguillés par le nom de domaine (`starlette.routing.Host`) :

- **burotu.com** : one-pager en anglais, français et espagnol, réservation d'une prise de sang.
- **admin.burotu.com** : réservations et parcours de chaque échantillon
  (demande reçue → rendez-vous confirmé → sang prélevé → au laboratoire → analysé →
  compte rendu envoyé), avec historique horodaté, centres, horaires et fermetures.

Les deux sites ont des cookies de session distincts ; les routes du suivi n'existent pas sur
le domaine public. Formulaires protégés par jeton CSRF, connexion limitée à 8 essais par
quart d'heure, en-têtes de sécurité stricts (CSP sans script externe), export CSV protégé
contre l'injection de formules.

Stockage : SQLite (WAL) dans un volume Docker. Suffisant pour le prototype ; le passage à
PostgreSQL ne touche que `db.py`.
