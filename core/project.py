"""Validated project contracts and reviewable, deterministic launch artifacts.

Model output is data. No install/start command is executed during generation.
"""
import hashlib
import json
import re
import shlex
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import posixpath
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANAGED_FILES = {'run.json', 'project.json', 'setup.sh', 'start.sh', 'SETUP.md', 'first_attempt.json', 'packaging.json'}


def safe_filename(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value or any(ord(char) < 32 for char in value):
        raise ValueError('File path must be a nonempty POSIX relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') or p.startswith('.') for p in value.split('/')):
        raise ValueError(f'Unsafe file path: {value}')
    if ':' in value or value.casefold() in {name.casefold() for name in MANAGED_FILES}:
        raise ValueError(f'Reserved or invalid file path: {value}')
    return value


def validate_tree(files: list[str]) -> list[str]:
    normalized = [safe_filename(f) for f in files]
    folded = [f.casefold() for f in normalized]
    if len(folded) != len(set(folded)):
        raise ValueError('Duplicate filenames (including case-only duplicates)')
    for name in folded:
        if any(name.startswith(other + '/') for other in folded if other != name):
            raise ValueError('File/directory path collision')
    return normalized


class Dependency(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ecosystem: Literal['npm', 'pip', 'system']
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @model_validator(mode='after')
    def pinned_version(self):
        pattern = (r'[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?'
                   if self.ecosystem == 'npm' else
                   r'[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc|\.post|\.dev)[0-9]+)?(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?')
        if self.ecosystem != 'system' and not re.fullmatch(pattern, self.version):
            raise ValueError('Declare an exact package version, not a range, URL or latest')
        return self


class ProjectContract(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    application_type: Literal['static', 'node', 'python']
    runtime: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    entry_point: str
    files: list[str] = Field(min_length=1)
    dependencies: list[Dependency] = Field(default_factory=list)
    dependency_manifests: list[str] = Field(default_factory=list)
    environment_variables: list[str] = Field(default_factory=list)
    install_commands: list[list[str]] = Field(default_factory=list)
    build_commands: list[list[str]] = Field(default_factory=list)
    start_command: list[str] = Field(default_factory=list)
    port: int = Field(default=8080, ge=1024, le=65535)

    @model_validator(mode='after')
    def check_contract(self):
        validate_tree(self.files)
        safe_filename(self.entry_point)
        if self.entry_point not in self.files:
            raise ValueError('Entry point must be listed in files')
        for manifest in self.dependency_manifests:
            safe_filename(manifest)
            if manifest not in self.files:
                raise ValueError('Dependency manifest must be listed in files')
        if any(not re.fullmatch(r'[A-Z_][A-Z0-9_]*', name) for name in self.environment_variables):
            raise ValueError('Environment variables must be names, never secret values')
        identities = [(d.ecosystem, d.name.lower().replace('_', '-')) for d in self.dependencies]
        if len(identities) != len(set(identities)):
            raise ValueError('Duplicate dependency declarations')
        commands = self.install_commands + self.build_commands + ([self.start_command] if self.start_command else [])
        for command in commands:
            if not command or any(not isinstance(arg, str) or not arg or '\x00' in arg or '\n' in arg for arg in command):
                raise ValueError('Commands must be nonempty argv lists')
            if command[0] not in {'npm', 'python3', 'python', '.venv/bin/python', '.venv/bin/pip'}:
                raise ValueError('Unsupported command executable; use npm or Python, not shell commands')
        if self.application_type == 'static':
            if self.dependencies or self.dependency_manifests or commands or self.environment_variables:
                raise ValueError('Static projects must need no dependency installation or custom commands')
            if not self.entry_point.lower().endswith(('.html', '.htm')):
                raise ValueError('Static entry point must be HTML')
        else:
            if not self.start_command:
                raise ValueError('Node/Python projects require an explicit start_command')
            if self.application_type == 'node' and 'package.json' not in self.dependency_manifests:
                raise ValueError('Node projects require package.json')
            if self.application_type == 'python' and self.start_command[0] != '.venv/bin/python':
                raise ValueError('Python start_command must use .venv/bin/python')
            if self.application_type == 'python' and self.dependencies and 'requirements.txt' not in self.dependency_manifests:
                raise ValueError('Python dependencies require requirements.txt')
        return self


def validate_artifacts(project: ProjectContract, files: dict[str, str]) -> list[str]:
    """Structural checks; never claim application correctness."""
    if set(project.files) != set(files):
        raise ValueError(f'Contract/file mismatch: missing={sorted(set(project.files)-set(files))}, undeclared={sorted(set(files)-set(project.files))}')
    if not files[project.entry_point].strip():
        raise ValueError('Entry point is empty')
    warnings = []
    if project.application_type == 'static':
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for attr, target in attrs:
                    if not target or not ((tag == 'script' and attr == 'src') or (tag == 'link' and attr == 'href')):
                        continue
                    parsed = urlsplit(target)
                    if parsed.scheme or parsed.netloc or not parsed.path:
                        continue
                    name = posixpath.normpath(posixpath.join(posixpath.dirname(self.filename), unquote(parsed.path))) if not parsed.path.startswith('/') else unquote(parsed.path).lstrip('/')
                    if name not in files:
                        raise ValueError(f'HTML references missing local asset: {self.filename} -> {name}')
        for name, content in files.items():
            if name.lower().endswith(('.html', '.htm')):
                parser = Links()
                parser.filename = name
                parser.feed(content)
    if project.application_type == 'node':
        package = json.loads(files['package.json'])
        if not isinstance(package, dict):
            raise ValueError('package.json must contain an object')
        declared = package.get('dependencies', {}) | package.get('devDependencies', {})
        expected = {d.name: d.version for d in project.dependencies if d.ecosystem == 'npm'}
        if declared != expected:
            raise ValueError('package.json dependencies must exactly match declared npm dependencies')
        for dependency in project.dependencies:
            if dependency.ecosystem == 'npm' and declared.get(dependency.name) != dependency.version:
                raise ValueError(f'Dependency missing or version mismatch in package.json: {dependency.name}')
        for command in project.build_commands + [project.start_command]:
            if command[:2] == ['npm', 'run'] and (len(command) < 3 or command[2] not in package.get('scripts', {})):
                raise ValueError('Referenced npm script is missing from package.json')
        if 'package-lock.json' not in files:
            warnings.append('No package-lock.json: installation is not locked to exact transitive versions')
        if any(cmd[:2] == ['npm', 'ci'] for cmd in project.install_commands) and 'package-lock.json' not in files:
            raise ValueError('npm ci requires package-lock.json')
    if project.application_type == 'python' and 'requirements.txt' in files:
        declared = {}
        for line in files['requirements.txt'].splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
            if not match:
                raise ValueError('requirements.txt must list exact package pins; nested files, URLs and options are unsupported')
            name, version = match.groups()
            normalized = name.lower().replace('_', '-')
            if normalized in declared:
                raise ValueError('Duplicate requirement')
            declared[normalized] = version
        expected = {d.name.lower().replace('_', '-'): d.version for d in project.dependencies if d.ecosystem == 'pip'}
        if declared != expected:
            raise ValueError('requirements.txt must exactly match declared pip dependencies')
    return warnings


def write_packaging(workspace: Path, project: ProjectContract, findings: list[str]) -> list[str]:
    """Quote argv, never interpolate executable model-authored shell text."""
    header = '#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "${BASH_SOURCE[0]}")"\n'
    setup = header
    if project.application_type == 'python':
        setup += 'python3 -m venv .venv\n'
    for command in project.install_commands + project.build_commands:
        setup += shlex.join(command) + '\n'
    setup += "printf '%s\\n' 'Setup complete; application functionality has not been tested.'\n"
    if project.application_type == 'static':
        command = ['python3', '-m', 'http.server', str(project.port), '--bind', '127.0.0.1']
    else:
        command = project.start_command
    start = header + 'exec ' + shlex.join(command) + '\n'
    notes = [
        '# Project setup', '', f'Type: {project.application_type}',
        f'Runtime: {project.runtime} {project.runtime_version}', f'Entry point: {project.entry_point}',
        '', 'Review project.json and commands before running setup.sh or start.sh.',
        'Commands may install dependencies and run generated application code. They are not executed by the dashboard.',
        'Use an isolated runtime for generated backends. No sudo is needed by these wrappers.',
        '', f'Expected port: {project.port}',
        'Required environment variable names: ' + (', '.join(project.environment_variables) or 'none'),
        '', 'Packaging findings:', *[f'- {finding}' for finding in findings],
        '', 'Packaging checks establish structure only. Startup and functional correctness are separate.',
    ]
    if project.application_type == 'static':
        notes += ['', 'Run: bash setup.sh, then bash start.sh',
                  f'Open http://127.0.0.1:{project.port}/{project.entry_point}']
    artifacts = {'project.json': json.dumps(project.model_dump(), indent=2), 'setup.sh': setup,
                 'start.sh': start, 'SETUP.md': '\n'.join(notes) + '\n'}
    for name, content in artifacts.items():
        (workspace / name).write_text(content)
        if name.endswith('.sh'):
            (workspace / name).chmod(0o755)
    return list(artifacts)


def snapshot(files: dict[str, str]) -> dict:
    return {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
