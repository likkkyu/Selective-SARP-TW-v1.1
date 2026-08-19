# Sync Endpoints (Cloud ↔ Local)

- Cloud SSH: `ssh -p 30479 root@connect.westb.seetacloud.com`
- Cloud project path: `/root/autodl-tmp/project-vrp-v6-4-main`
- Local project path: `/Users/bytedance/Downloads/project-vrp-v6-4-main`

## Rsync template (must include custom port)

```bash
rsync -azv -e "ssh -p 30479" \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/project-vrp-v6-4-main/<remote-path> \
  /Users/bytedance/Downloads/project-vrp-v6-4-main/<local-path>
```
