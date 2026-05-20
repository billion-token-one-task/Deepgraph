param(
  [string]$DeployRoot = "H:\\deepgraph-full-deploy"
)

$ErrorActionPreference = "Stop"

$deepgraph = Join-Path $DeployRoot "Deepgraph"
$composePostgres = Join-Path $deepgraph "docker-compose.postgres.yml"
$composeGrobid = Join-Path $deepgraph "docker-compose.grobid.yml"
$dumpDir = Join-Path $DeployRoot "deploy_bundle\\postgres"

if (-not (Test-Path $composePostgres)) {
  throw "Missing $composePostgres (check DeployRoot)"
}
if (-not (Test-Path (Join-Path $dumpDir "deepgraph_pg.dump"))) {
  throw "Missing pg dump at $dumpDir\\deepgraph_pg.dump"
}

docker version | Out-Null

Push-Location $deepgraph
try {
  docker compose -f $composePostgres up -d

  $dumpWinPath = (Resolve-Path $dumpDir).Path
  docker run --rm `
    -v "${dumpWinPath}:/dump:ro" `
    -e PGPASSWORD=deepgraph `
    pgvector/pgvector:pg16 `
    pg_restore -h host.docker.internal -p 5433 -U deepgraph -d deepgraph `
      --clean --if-exists --no-owner --no-acl `
      /dump/deepgraph_pg.dump

  # Optional: better PDF parse quality (large image)
  if (Test-Path $composeGrobid) {
    docker compose -f $composeGrobid up -d
  }
} finally {
  Pop-Location
}

Write-Host "Postgres restore finished. Next: cd `"$deepgraph`" then create venv + pip install -r requirements.txt + python main.py"
