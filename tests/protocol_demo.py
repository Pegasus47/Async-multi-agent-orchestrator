"""Manual browser demo using controlled protocol replies. Never a real LLM benchmark result."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO),str(REPO/'tests')]
from test_project_pipeline import FixtureServer, unused_port
from core.orchestrator import run_orchestration
from server.db import save_run
from tools.webgen_pilot import load_task,map_run

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Local protocol fixture: to-do</title><link rel="stylesheet" href="css/style.css"></head><body>
<h1>To-do — LOCAL TEST FIXTURE</h1><p>Controlled Ollama protocol replies; no real model inference or benchmark score.</p><p>Test account: fixture-user</p>
<form id="add"><label>Task title <input id="title" required></label><button>Add Task</button></form><label>Search <input id="search"></label>
<label>Filter <select id="filter"><option value="all">All</option><option value="done">Completed</option><option value="active">Uncompleted</option></select></label>
<ul id="tasks"></ul><output id="status"></output><script src="js/app.js"></script></body></html>'''
JS='''const tasks=[]; const list=document.querySelector('#tasks');
function render(){list.replaceChildren();for(const task of tasks){if(!task.title.toLowerCase().includes(document.querySelector('#search').value.toLowerCase()))continue;const filter=document.querySelector('#filter').value;if(filter==='done'&&!task.done||filter==='active'&&task.done)continue;const li=document.createElement('li');const text=document.createElement('span');text.textContent=task.title;const done=document.createElement('input');done.type='checkbox';done.checked=task.done;done.setAttribute('aria-label','Complete '+task.title);done.onchange=()=>{task.done=done.checked;render()};if(task.done)text.style.textDecoration='line-through';const edit=document.createElement('button');edit.textContent='Edit';edit.onclick=()=>{const name=prompt('New task description',task.title);if(name){task.title=name;render()}};const del=document.createElement('button');del.textContent='Delete';del.onclick=()=>{tasks.splice(tasks.indexOf(task),1);render()};li.append(done,text,edit,del);list.append(li)}}
document.querySelector('#add').onsubmit=e=>{e.preventDefault();tasks.push({title:document.querySelector('#title').value,done:false});document.querySelector('#title').value='';document.querySelector('#status').textContent='Task added to fixture-user account';render()};document.querySelector('#search').oninput=render;document.querySelector('#filter').onchange=render;'''
CSS='body{background:lavender;color:indigo;font-family:sans-serif;margin:3rem}button,input,select{padding:.5rem;margin:.5rem}li{padding:.5rem}'


def main():
    output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=True);os.chdir(output)
    from server import db
    db.DB_PATH=str(output/'data'/'orchestrator_study.db')
    record=load_task()
    with FixtureServer() as fixture:
        fixture.project=dict(application_type='static',runtime='browser',runtime_version='modern',entry_point='index.html',
            files=['index.html','css/style.css','js/app.js'],port=unused_port())
        fixture.sources={'index.html':HTML,'css/style.css':CSS,'js/app.js':JS}
        report=asyncio.run(run_orchestration(record['task']['instruction'],3,'ollama/fixture-coder'))
        if report.packaging['status']!='ready':raise RuntimeError(report.packaging)
        save_run(report.to_dict())
        workspace=Path(report.output_dir)
        subprocess.run(['bash','setup.sh'],cwd=workspace,check=True)
        url=f"http://127.0.0.1:{fixture.project['port']}/index.html"
        mapping=map_run(workspace/'run.json',url,output/'benchmark-input',record)
        mapping['generation_kind']='controlled_ollama_protocol_fixture'
        (output/'benchmark-input'/'mapping.json').write_text(json.dumps(mapping,indent=2))
        evidence={'run_id':report.run_id,'workspace':str(workspace),'url':url,'generation_kind':'controlled_ollama_protocol_fixture',
            'functional_score':None,'startup_script':'bash start.sh', 'real_llm_inference':False}
        (output/'demo.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence),flush=True)
        subprocess.run(['bash','start.sh'],cwd=workspace,check=True)

if __name__=='__main__':main()
