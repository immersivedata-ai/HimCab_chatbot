# deploy.ps1 — Travel Assistant Cloud Run Deploy Script
# Usage: .\deploy.ps1
# Run from the project root: d:\Immersivedata\travel_assistant

$PROJECT = "himcab-tourist-assistant"
$IMAGE = "us-central1-docker.pkg.dev/$PROJECT/travel-assistant-repo/travel-assistant-api:latest"
$SERVICE = "travel-assistant-api"
$REGION = "us-central1"

# This service deploys as an API only. The browser test clients under frontend/ are
# excluded from the image (.dockerignore) and no longer served, so there are no pages
# to check for here - what ships is app/ and nothing else.

Write-Host "Building and pushing container..." -ForegroundColor Cyan
gcloud builds submit --tag $IMAGE --project=$PROJECT
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed!" -ForegroundColor Red; exit 1 }

Write-Host "Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy $SERVICE `
    --image=$IMAGE `
    --region=$REGION `
    --platform=managed `
    --port=8080 `
    --allow-unauthenticated `
    --memory=2Gi `
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
Write-Host "Every /v1/chat/* call needs: Authorization: Bearer <Him Cab rider JWT>" -ForegroundColor DarkGray
