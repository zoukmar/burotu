# Déploiement

Burotu tourne sur **srv-prod-2** (`193.70.45.138`) comme les autres projets du serveur :
une image GHCR étiquetée par commit, une stack Compose dans `/opt/burotu`, et le **Caddy
système** de l'hôte pour le HTTPS.

```
GitHub (push sur main)
  └─ Actions : lint + tests → image ghcr.io/zoukmar/burotu-web:<sha> → SSH
       └─ srv-prod-2 : /opt/burotu/deploy.sh <sha>
            └─ docker compose up  (127.0.0.1:8120)
                 └─ Caddy : burotu.com · www.burotu.com → burotu.com · admin.burotu.com
```

| Domaine | Rôle |
| --- | --- |
| `burotu.com` | Site public (one-pager EN/FR/ES, réservation) |
| `www.burotu.com` | Redirection permanente vers `burotu.com` |
| `admin.burotu.com` | Suivi : réservations, parcours des échantillons, centres |

Un seul conteneur sert les deux sites ; l'application choisit le site selon le nom de domaine.

## Mise en place (une seule fois)

### 1. DNS

Chez le registrar de `burotu.com`, trois enregistrements A vers `193.70.45.138` :
`@`, `www` et `admin`.

### 2. Serveur

```bash
# Utilisateur dédié au déploiement, membre du groupe docker, sans sudo
sudo adduser --disabled-password --gecos "" deploy-burotu
sudo usermod -aG docker deploy-burotu
sudo install -d -o deploy-burotu -g deploy-burotu -m 750 /opt/burotu

# Secrets de l'application
sudo -u deploy-burotu tee /opt/burotu/.env >/dev/null <<EOF
BUROTU_ADMIN_USER=admin
BUROTU_ADMIN_PASSWORD=$(openssl rand -base64 24)
BUROTU_SECRET_KEY=$(openssl rand -hex 32)
MODEL_API_KEY=
EOF
sudo chmod 600 /opt/burotu/.env
sudo cat /opt/burotu/.env   # noter le mot de passe admin dans le gestionnaire de mots de passe
```

### 3. Clé SSH de déploiement

Sur le serveur :

```bash
sudo -u deploy-burotu ssh-keygen -t ed25519 -N "" -C "github-actions-burotu" -f /home/deploy-burotu/.ssh/id_deploy
sudo -u deploy-burotu sh -c 'cat ~/.ssh/id_deploy.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
sudo cat /home/deploy-burotu/.ssh/id_deploy        # clé privée → secret DEPLOY_SSH_KEY
ssh-keyscan -t ed25519 srv-prod-2.tuvalutech.com   # → secret DEPLOY_KNOWN_HOSTS
sudo rm /home/deploy-burotu/.ssh/id_deploy         # la clé privée ne reste que dans GitHub
```

Dans GitHub, **Settings → Environments → New environment** : `production`, puis ajouter
ces secrets d'environnement :

| Secret | Valeur |
| --- | --- |
| `DEPLOY_HOST` | `srv-prod-2.tuvalutech.com` |
| `DEPLOY_USER` | `deploy-burotu` |
| `DEPLOY_SSH_KEY` | la clé privée générée ci-dessus |
| `DEPLOY_KNOWN_HOSTS` | la sortie de `ssh-keyscan` |

### 4. Caddy

Ajouter le contenu de [`deploy/caddy/burotu.caddy`](../deploy/caddy/burotu.caddy) à la fin de
`/etc/caddy/Caddyfile`, puis :

```bash
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

Caddy obtient les certificats Let's Encrypt dès que le DNS pointe sur le serveur.

## Au quotidien

- **Déployer** : pousser sur `main`. Le workflow *Deploy* fait le reste.
- **Redéployer une version** : Actions → *Deploy* → *Run workflow*, ou sur le serveur
  `bash /opt/burotu/deploy.sh <sha>`.
- **Retour arrière automatique** : si le site ne répond pas après 30 s, `deploy.sh`
  relance la version précédente.
- **Logs** : `docker compose -f /opt/burotu/docker-compose.yml logs -f web`
- **Sauvegarde** de la base SQLite (volume `burotu_data`) :

```bash
docker compose -f /opt/burotu/docker-compose.yml exec web \
  python -c "import sqlite3; sqlite3.connect('/data/site.db').backup(sqlite3.connect('/data/backup.db'))"
```
