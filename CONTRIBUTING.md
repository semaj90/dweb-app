Please do not commit large binaries, installers, or generated database payloads to this repository.

Guidelines
- Avoid committing: *.exe, *.msi, *.zip, and large database payloads (qdrant payloads, MinIO data, Neo4j distributions).
- If you need to distribute large binaries, use a release artifact (GitHub Releases) or an external storage service.
- Use the existing .gitignore rules. If a file is accidentally added, run `git rm --cached <file>` and open a PR with the cleanup.

If you must include binary assets, request an exception and we'll add Git LFS for selective files.
