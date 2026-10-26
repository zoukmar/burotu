# Deployment

Burotu runs on **srv-prod-2** (`193.70.45.138`): one GHCR image tagged per commit, a Compose
stack in `/opt/burotu`, and the host's **system Caddy** for HTTPS.

```
GitHub (push to main)
  └─ Actions: lint + tests → image ghcr.io/zoukmar/burotu-web:<sha> → SSH
       └─ srv-prod-2: /opt/burotu/deploy.sh <sha>
            └─ docker compose up  (127.0.0.1:8120)
                 └─ Caddy: burotu.com · www.burotu.com → burotu.com · admin.burotu.com
```

| Domain | Role |
| --- | --- |
| `burotu.com` | Public site (one-pager in EN/FR/ES, booking) |
| `www.burotu.com` | Permanent redirect to `burotu.com` |
| `admin.burotu.com` | Tracking: bookings, sample journey, collection sites |

A single container serves both sites; the app picks the site from the host name.

## One-time setup

### 1. DNS

Point `burotu.com`, `www` and `admin` to `193.70.45.138` (A records, or CNAMEs to
`burotu.com` for the subdomains). Remove any parking records and any AAAA records.

### 2. Server

```bash
# Dedicated deploy user, member of the docker group, no sudo
sudo adduser --disabled-password --gecos "" deploy-burotu
sudo usermod -aG docker deploy-burotu
sudo install -d -o deploy-burotu -g deploy-burotu -m 750 /opt/burotu

# Application secrets
sudo -u deploy-burotu tee /opt/burotu/.env >/dev/null <<EOF
BUROTU_ADMIN_USER=admin
BUROTU_ADMIN_PASSWORD=$(openssl rand -base64 24)
BUROTU_SECRET_KEY=$(openssl rand -hex 32)
MODEL_API_KEY=
EOF
sudo chmod 600 /opt/burotu/.env
sudo grep ADMIN_PASSWORD /opt/burotu/.env   # store it in your password manager
```

### 3. Deploy SSH key

On the server:

```bash
sudo -u deploy-burotu install -d -m 700 /home/deploy-burotu/.ssh
sudo -u deploy-burotu ssh-keygen -t ed25519 -N "" -C "github-actions-burotu" -f /home/deploy-burotu/.ssh/id_deploy
sudo -u deploy-burotu sh -c 'cat ~/.ssh/id_deploy.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
sudo cat /home/deploy-burotu/.ssh/id_deploy        # private key → DEPLOY_SSH_KEY secret
ssh-keyscan -t ed25519 srv-prod-2.tuvalutech.com   # → DEPLOY_KNOWN_HOSTS secret
sudo rm /home/deploy-burotu/.ssh/id_deploy         # the private key now only lives in GitHub
```

In GitHub, go to **Settings → Environments → New environment**, name it `production`, and add
these environment secrets:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | `srv-prod-2.tuvalutech.com` |
| `DEPLOY_USER` | `deploy-burotu` |
| `DEPLOY_SSH_KEY` | the private key generated above |
| `DEPLOY_KNOWN_HOSTS` | the `ssh-keyscan` output |

### 4. Caddy

Append [`deploy/caddy/burotu.caddy`](../deploy/caddy/burotu.caddy) to `/etc/caddy/Caddyfile`,
then:

```bash
sudo install -d -o caddy -g caddy /var/log/caddy
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo chown -R caddy:caddy /var/log/caddy   # validate runs as root and may create the log files
sudo systemctl reload caddy
```

Caddy gets Let's Encrypt certificates as soon as DNS points to the server.

## Day to day

- **Deploy**: push to `main`. The *Deploy* workflow does the rest.
- **Redeploy a version**: Actions → *Deploy* → *Run workflow*, or on the server
  `bash /opt/burotu/deploy.sh <sha>`.
- **Automatic rollback**: if the site does not answer within 30 s, `deploy.sh` restarts the
  previous version.
- **Logs**: `docker compose -f /opt/burotu/docker-compose.yml logs -f web`
- **Back up** the SQLite database (volume `burotu_data`):

```bash
docker compose -f /opt/burotu/docker-compose.yml exec web \
  python -c "import sqlite3; sqlite3.connect('/data/site.db').backup(sqlite3.connect('/data/backup.db'))"
```
