@echo off
title UrbanGraph-SG Launcher
echo ============================================
echo   UrbanGraph-SG — Launching...
echo ============================================
echo.
echo [1/2] Starting Neo4j (background)...
start "Neo4j" /MIN "C:\Users\Jzh20\Desktop\neo4j-community-5.26.4\bin\neo4j.bat" console
echo   Waiting for Neo4j to be ready...
timeout /t 20 /nobreak > nul
echo   Neo4j should be ready at http://localhost:7474
echo.
echo [2/2] Starting Streamlit UI...
echo   Open http://localhost:8502 in your browser
echo.
streamlit run src\ui\streamlit_app.py --server.port=8502
