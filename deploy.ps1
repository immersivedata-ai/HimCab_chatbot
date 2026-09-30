# deploy.ps1 — Travel Assistant Cloud Run Deploy Script
# Usage: .\deploy.ps1
# Run from the project root: d:\Immersivedata\travel_assistant

$PROJECT = "himcab-tourist-assistant"
$IMAGE = "us-central1-docker.pkg.dev/$PROJECT/travel-assistant-repo/travel-assistant-api:latest"
$SERVICE = "travel-assistant-api"
$REGION = "us-central1"

# The browser test clients ship INSIDE the image (Dockerfile COPY frontend/) and are
# served by FastAPI on the same origin, so there is nothing separate to upload. What
# can go wrong is silent: a missing page still builds, still deploys, and returns 404
# to whoever opens it - so they are checked here instead of in a browser afterwards.
$FRONTEND_PAGES = @("index.html", "local.html", "rental.html")

Write-Host "Checking test client pages..." -ForegroundColor Cyan
foreach ($page in $FRONTEND_PAGES) {
    $path = Join-Path $PSScriptRoot "frontend\$page"
    if (-not (Test-Path $path)) {
        Write-Host "Missing frontend\$page - that workflow would 404. Aborting." -ForegroundColor Red
        exit 1
    }
    Write-Host "  frontend\$page" -ForegroundColor DarkGray
}

Write-Host "Building and pushing container..." -ForegroundColor Cyan
gcloud builds submit --tag $IMAGE --project=$PROJECT
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }

Write-Host "Deploying to Cloud Run..." -ForegroundColor Cyan

# --min-instances=1 is what removes the cold start, and it is the only thing that
#   does. Importing this application takes ~7 seconds of Python before it can answer
#   anything, and the browser cannot even fetch the test pages until it has, because
#   this container serves them. With no warm instance the first visitor after an idle
#   period waited ~40 seconds. This keeps one instance alive, which is BILLABLE around
#   the clock - that is the trade, and it is the only real fix for a first request.
# --cpu-boost gives extra CPU during startup. Free, and an import-heavy Python app is
#   exactly the workload it exists for.
# --cpu=2 so the import is not serialised onto a single core.
# --concurrency=80 because the app is async: one instance handles many requests on its
#   event loop, so Cloud Run should fill an instance before starting another.
gcloud run deploy $SERVICE `
    --image=$IMAGE `
    --region=$REGION `
    --platform=managed `
    --port=8080 `
    --allow-unauthenticated `
    --memory=2Gi `
    --cpu=2 `
    --cpu-boost `
    --min-instances=1 `
    --max-instances=10 `
    --concurrency=80 `
    --timeout=300 `
    --project=$PROJECT `
    --set-env-vars="GCP_PROJECT_ID=himcab-tourist-assistant,GCP_REGION=us-central1,VERTEX_AI_LOCATION=us-central1,GEMINI_PRO_MODEL=gemini-2.5-flash-lite,GEMINI_FLASH_MODEL=gemini-2.5-flash-lite,FIREBASE_PROJECT_ID=himcab-tourist-assistant,FIRESTORE_DATABASE_ID=(default),FIRESTORE_SESSIONS_COLLECTION=trip_sessions,FIRESTORE_USERS_COLLECTION=user_profiles,REDIS_HOST=localhost,REDIS_PORT=6379,REDIS_DB=0,APP_ENV=development,APP_DEBUG=true,APP_HOST=0.0.0.0,APP_PORT=8080,LOG_LEVEL=INFO"

if ($LASTEXITCODE -ne 0) { Write-Host "Deploy failed!" -ForegroundColor Red; exit 1 }

$URL = "https://travel-assistant-api-42345441632.us-central1.run.app"
Write-Host "Deployed! $URL" -ForegroundColor Green
Write-Host ""
Write-Host "API docs - share these with an integrating team:" -ForegroundColor Cyan
Write-Host "  Swagger UI   : $URL/docs"          -ForegroundColor Green
Write-Host "  Rental only  : $URL/docs#/Rental"  -ForegroundColor Green
Write-Host "  ReDoc        : $URL/redoc"         -ForegroundColor DarkGray
Write-Host "  OpenAPI JSON : $URL/openapi.json"  -ForegroundColor DarkGray
Write-Host "  Health       : $URL/v1/health"     -ForegroundColor DarkGray
Write-Host ""
Write-Host "Test clients - paste a real Him Cab rider JWT into each to get live fares:" -ForegroundColor Cyan
Write-Host "  Outstation   : $URL/outstation" -ForegroundColor Green
Write-Host "  Local        : $URL/local"      -ForegroundColor Green
Write-Host "  Rental       : $URL/rental"     -ForegroundColor Green
Write-Host ""
Write-Host "Every /v1/chat/* call needs: Authorization: Bearer <Him Cab rider JWT>" -ForegroundColor DarkGray
Write-Host "The test clients are served because APP_ENV is development. Set it to" -ForegroundColor DarkGray
Write-Host "production to serve the API alone, with no rebuild." -ForegroundColor DarkGray
