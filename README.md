# LLM Load Test Web App

A Flask front-end for the vLLM/OpenAI-compatible load-test script — enter the
server, model, concurrent-user levels, and input-token sweep in a form, then
watch progress live and download the PDF report (10 charts) and raw JSON.

## Run it

```bash
pip install -r requirements.txt
python3 app.py
```

Then open http://localhost:5000

## What you can configure in the form

- **Base URL / model** — your OpenAI-compatible endpoint (e.g. vLLM at `http://localhost:8000`) and model name/path.
- **Concurrent users** — comma-separated list, e.g. `10,20,40,60,80,100`. Each value is tested at every input-token level.
- **Input token sweep** — min, max, and step (tokens are approximated ~4 chars/token, then the *actual* prompt_tokens value vLLM returns is used for all stats).
- **Generation settings** — max output tokens, temperature, per-request timeout, cooldown between rounds.
- **Base prompt** — optional; defaults to a red-black-tree coding prompt that gets padded with filler text to hit each target input-token size.

## What you get

- A live dashboard (polls every 2s) with a progress bar, running log, and a
  results table that fills in round by round.
- A "Stop after current round" button to cut a run short without losing what
  already completed.
- Once finished: a PDF report with 10 charts (throughput, input throughput,
  output-tokens/sec-per-user, p50/p90/p99 latency, success rate — each vs.
  input tokens, plus two "vs. concurrent users" views) plus a detailed table
  and any sample errors, and a downloadable results.json.

## Notes

- Each run gets its own folder under `runs/<run_id>/` with `results.json`,
  `report.pdf`, and `charts/*.png`.
- This uses Flask's built-in dev server (`threaded=True`), which is fine for
  local/lab use running one operator's tests. For anything exposed beyond
  localhost, put it behind a real WSGI server (gunicorn, etc.).
- Runs are tracked in memory — restarting the app clears the run registry
  (finished PDFs/JSON on disk are unaffected, just not linked in the UI).
