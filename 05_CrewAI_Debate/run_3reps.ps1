$ErrorActionPreference = "Stop"
$env:OLLAMA_CONTEXT_LENGTH = "8192"
# El cliente ChatOpenAI (langchain_openai) respeta el proxy corporativo de la
# UMA configurado a nivel de Windows (registro, ProxyOverride=<local>), que no
# excluye correctamente localhost para este cliente -> las peticiones a
# localhost:11434 se desviaban al proxy y este las rechazaba (403 Squid).
# NO_PROXY/no_proxy fuerza el bypass explicito, verificado con una llamada real.
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = "localhost,127.0.0.1"
$logFile = "run_3reps.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Tee-Object -FilePath $logFile -Append
}

Log "=== Iniciando 3 repeticiones de CrewAI (8B heterogeneo, num_ctx=8192 y proxy corregidos) ==="

foreach ($rep in 1,2,3) {
    $decisionCheck = "Resultados\rep$rep\q050\q050_debateC_decision.json"
    if (Test-Path $decisionCheck) {
        Log "SKIP rep${rep}: ya tiene q050."
        continue
    }
    Log "LANZANDO rep$rep..."
    $psi = Start-Process -FilePath ".\venv\Scripts\python.exe" `
        -ArgumentList "crewai_debate.py --start 1 --rep $rep" `
        -WorkingDirectory "c:\Users\Laura\Desktop\TFG_Laura\05_CrewAI_Debate" `
        -RedirectStandardOutput "rep${rep}_stdout.log" `
        -RedirectStandardError "rep${rep}_stderr.log" `
        -NoNewWindow -PassThru
    $psi.WaitForExit()
    if (Test-Path $decisionCheck) {
        Log "  OK: rep$rep completado."
    } else {
        Log "  AVISO: rep$rep termino (code=$($psi.ExitCode)) pero q050 no existe."
    }
}

Log "=== 3 repeticiones de CrewAI TERMINADAS ==="
