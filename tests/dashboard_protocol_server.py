"""Browser-only fixture server. Replies are deterministic; no real model inference."""
import os
from pathlib import Path
import sys


def main():
    repo=Path(__file__).resolve().parent.parent
    sys.path[:0]=[str(repo),str(repo/'tests')]
    output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=True);os.chdir(output)
    from test_project_pipeline import FixtureServer
    from protocol_demo import HTML,JS,CSS
    from server import db
    db.DB_PATH=str(output/'data'/'orchestrator_study.db')
    from server.app import app
    import uvicorn
    with FixtureServer() as fixture:
        fixture.project=dict(application_type='static',runtime='browser',runtime_version='modern',entry_point='index.html',
            files=['index.html','css/style.css','js/app.js'],port=8080)
        fixture.sources={'index.html':HTML,'css/style.css':CSS,'js/app.js':JS}
        print('CONTROLLED OLLAMA PROTOCOL FIXTURE; NOT A REAL LLM',flush=True)
        uvicorn.run(app,host='127.0.0.1',port=8001)


if __name__=='__main__':main()
