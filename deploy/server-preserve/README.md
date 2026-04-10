# Server Preserve Notes

This folder captures the current server wiring before replacing the application.

Keep these pieces when deploying the new CCTV version:

- Domain: `dev-cctv.hio.ai.kr`
- TLS: existing Certbot certificate under `/etc/letsencrypt/live/dev-cctv.hio.ai.kr/`
- Reverse proxy entrypoint: `nginx` listens on `80/443`
- Current app upstream port: `127.0.0.1:8002`
- Service user: `vlmapp`
- Timezone: `Asia/Seoul`

Recommended cutover sequence:

1. Preserve this repo state in GitHub before deletion.
2. Stop and disable old `vlm-*` systemd services.
3. Deploy the new CCTV app in a new directory.
4. Update only the upstream target in `nginx-live-vlm-cctv.conf` if the new app uses a different localhost port.
5. Recreate a single systemd service for the new app under the `vlmapp` user if desired.
6. Run `nginx -t` and restart the new service.

If the new app still serves HTTP locally, the minimum reusable asset is:

- `nginx-live-vlm-cctv.conf`

If the new app should also inherit the same service account and boot behavior, also reuse:

- `vlm-frontend.service`
- `service-override.conf`

Server inventory at snapshot time:

- Enabled nginx site: `/etc/nginx/sites-available/vlm-cctv`
- Enabled services: `vlm-model.service`, `vlm-db.service`, `vlm-frontend.service`
- Active symlink: `/etc/nginx/sites-enabled/vlm-cctv`
