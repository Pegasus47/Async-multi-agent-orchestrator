# Local generation, packaging, and WebGen-Bench pilot

Implemented and verified on 29 October 2026. HPC was not accessed or modified.

## What now works

The dashboard defaults to Ollama. It discovers models on the already-running endpoint at `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`), lets you select planner and workers, and makes keyless generation requests. It does not install/pull models, start Ollama, or submit Slurm jobs. OpenRouter remains an explicit optional provider. Unavailable models and mixed providers fail before inference.

The CLI and dashboard share the same pipeline:

1. Planner returns a validated project contract plus exactly one safe, unique path per worker.
2. Workers receive the shared runtime, file tree, interface, and dependency requirements. Nested parent directories are created during assembly.
3. The planner runs a second time as packaging reviewer, seeing actual source and worker errors. It can add missing dependency manifests, but cannot overwrite worker source or produce arbitrary shell text for automatic execution.
4. Structural validation checks the entry point, exact file set, local HTML script/style references, package versions and declared npm scripts. Missing workers, malformed contracts, invalid paths, or failed reviews cannot produce ready packaging. One corrective retry is allowed for malformed structured output.
5. The application creates `project.json`, `setup.sh`, `start.sh`, and `SETUP.md` from validated structured data. It saves `packaging.json`, a `first_attempt.json` contract/source-hash record, and an atomic `run.json`.
6. Full metadata persists through SQLite and run.json imports. Legacy databases are migrated without dropping existing rows. Planner, worker, and packaging tokens are tracked separately; totals include all three phases.

`ready` means packaging structure passed. It does not mean dependencies were installed, the app started, or functional tests passed. Those statuses remain `not_run` until an actual runner/evaluator supplies evidence. Commands are quoted argv lists, but dependency installation and generated application code still require an isolated runtime and review. Backend scripts are not executed inside the dashboard process.

Static start.sh uses Python's HTTP server on localhost. Python setup.sh creates a fresh `.venv`; Node/Python scripts execute declared install/build/start commands. The project contract declares environment-variable names, never secret values. The runner does not provision system packages or secret values.

A separate application server provides a normal origin for apps using localStorage or backend requests. The inline sandbox supports classic scripts and client-side form handlers, but deliberately lacks the dashboard's origin/storage and blocks fetch connections and native form submissions. Dashboard API CORS does not allow opaque preview origins. Module-based apps should use the separately started application server. Use the separately started app for realistic functional evaluation.

## Local verification evidence

No real local LLM was installed or invoked, as requested. The integration fixture implements real HTTP `/api/tags` and `/v1/chat/completions` responses with deterministic source and metadata. The actual dashboard, model client, orchestration, workers, packaging, database and generated launchers are exercised.

- 22 Python tests pass, including actual static, Python and Node setup/start scripts and HTTP startup checks in temporary directories.
- No OpenRouter key or authorization header is needed in the Ollama tests.
- Invalid plans cannot dispatch workers; missing models cannot begin inference; packaging cannot overwrite source.
- Packaging failures retain run.json and error details without emitting a ready launcher.
- Node manifest/version mismatches, missing HTML assets, traversal and symlinks are rejected.
- Legacy SQLite migration and new project/phase metadata round trips pass.
- Browser verification: selected Ollama, clicked Run Build without a key, saw ready packaging/project details and generated files, and added a task inside the preview. This caught and fixed sandbox form blocking.
- Generated standalone start.sh was also exercised in a browser; the add-task operation worked.
- Build, compile/import checks, `pip check`, and `git diff --check` passed.

Run the tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
npm run build --prefix frontend
```

Run the optional browser fixture (isolated database on port 8001, stop with Ctrl+C):

```bash
.venv/bin/python tests/dashboard_protocol_server.py /tmp/orchestrator-dashboard-protocol
```

This serves controlled replies. It is not a model-quality test. Real generation, dependency inference quality, and final-review quality remain to be tested with actual HPC models.

## WebGen-Bench inspection and selected pilot

Inspected upstream commit `c89e0743438e458d617cebaf7a682e0e66e41049`:

- Repository: https://github.com/mnluzimu/WebGen-Bench
- Source data: `data/test.jsonl`, 101 generation prompts and 647 UI cases.
- Pilot: task `000080`, advanced to-do application, preserved verbatim in `benchmarks/webgen/task_000080.json` with source/task SHA-256 and provenance.
- Cases: create, edit, complete, delete, search, filter, navigation, plus one design/colour case.
- Functional-only pilot excludes the design case explicitly; original indices and expected results are preserved.

The original UI evaluator uses WebVoyager with a model-driven browser agent and YES/NO/PARTIAL outcomes. Its existing wrappers assume other generators, include Windows paths, and have a fixed judge endpoint. Our adapter exports the input schema needed by that evaluator; it does not pretend that the unmodified upstream wrapper already supports our generator.

### Prepare the exact original generation prompt

```bash
.venv/bin/python tools/webgen_pilot.py prepare --output benchmark-results/webgen-000080
```

This writes the original `prompt.txt` and task/provenance JSON. On the next explicitly authorized HPC run, pass that exact prompt to the CLI within an appropriate Slurm compute allocation. Do not append testing hints or change the task if comparing against this prompt. Select actual installed planner/worker model IDs. Avoid an artificially small `LLM_MAX_TOKENS` cap that truncates the larger project contract or packaging JSON. No HPC run was initiated during implementation.

### After generation and local import

Inspect packaging findings and scripts. Start the app locally in an appropriate isolated runtime, independently of the dashboard. Then export only the first original functional case (create a task):

```bash
.venv/bin/python tools/webgen_pilot.py map \
  --manifest workspaces/YOUR_RUN_ID/run.json \
  --url http://127.0.0.1:8080/index.html \
  --case-index 0 \
  --output benchmark-results/webgen-000080/evaluation
```

Omit `--case-index` to export all seven non-design cases. The adapter verifies the original generation prompt matches exactly, requires ready packaging, preserves each operation and expected result, and writes `webvoyager_tasks.jsonl` plus `mapping.json`. It accepts only localhost target URLs. It does not verify startup or assign scores.

The input JSONL can be used for manual browser testing first, or fed to upstream `webvoyager/run.py --test_file ...` after configuring a real judge endpoint/model and its isolated evaluator environment. Before running upstream, parameterize its hardcoded `OpenAI(..., base_url=...)` in `webvoyager/run.py` using environment configuration, use Linux paths, and avoid copying credentials from example wrappers. The upstream browser-agent dependencies have not been installed here. The judge must be chosen explicitly; API-key-free generation does not imply that a hosted judge is free or keyless. An authorized HPC vision-model endpoint is another possible judge.

### What was demonstrated here

We passed the original task through the complete controlled Ollama protocol pipeline, generated the to-do fixture, ran its actual setup/start scripts, exported case `000080_0`, then filled and submitted Add Task in a real browser. The new task and confirmation appeared under the fixture's simulated account. It also worked inside the dashboard preview after the sandbox fix.

This verifies adapter wiring and a browser-operation smoke check. It is not a real LLM generation result, an independent judge result, or an official WebGen-Bench score. The simulated fixture account does not establish real authentication or persistence. `functional_score` remains null. Preserve first-attempt scores separately from any future repairs.

## Next HPC session

Local implementation is ready for a controlled real-model trial, not for an unqualified correctness claim. First inspect the remote branch, installed model list, active scheduler limits and SSH access under the existing read-only boundary. Any source transfer, remote dependency changes or Slurm/model run needs the user's subsequent authorization. Use the supplied Sharanga FAQ: no application execution on login nodes, Slurm for compute, and SCP/rsync downloads for local evaluation.

## Retrieving this change on HPC

The handoff records different Git histories: GitHub main was synchronized by copying source,
while the old HPC checkout was on featRecursiveChecks. Do not blindly merge or pull main into
that old checkout. Inspect branch/status/history first; a fresh clone of GitHub main into a
separate directory is the simplest way to preserve the existing HPC checkout and its results.
Do not reset, clean, force-pull, or overwrite the existing HPC checkout/environment.

When launching from a separate clone, export REPO_DIR to that clone's absolute path before
using run_orchestrator_ollama.sh. Otherwise the launcher defaults to the original repository
path under your HPC home directory. The launcher contains a previously used partition/GPU
request; confirm current allocation rules and availability before launching it. Run generation
only within a Slurm compute allocation. Do not run the local regression suite, dashboard,
or generated application setup/start scripts on the login node.

The intended order is:

1. Retrieve the committed GitHub source on HPC without merging unrelated histories.
2. Within an authorized Slurm allocation, run generation using the exact prepared benchmark prompt.
3. The final packaging/model review runs automatically after the workers, saving run.json,
   packaging findings, project.json and setup/start scripts when structural checks pass.
4. Pull finished workspaces to the laptop with run_cli.py --pull.
5. Locally review packaging findings and generated scripts, set up/start the generated app
   in an appropriate isolated runtime, then perform browser/functional evaluation.
6. Use the local dashboard to show source, generation/packaging metadata, and HTML previews.
   Module/back-end applications use their separate local server for full functional testing.

Functional evaluation reports are not automatically ingested into the dashboard yet; the
benchmark adapter exports case inputs/mapping, and independent evaluator results need to
be recorded separately until that reporting integration is implemented.
