"""Create a human-readable, per-run snapshot of training parameters."""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def _git_output(arguments, cwd):
    try:
        result = subprocess.run(
            ['git', *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolved_input_paths(arguments):
    path_fields = {}
    for name, value in vars(arguments).items():
        if value in (None, ''):
            continue
        normalized_name = name.lower()
        if not any(token in normalized_name for token in ('config', 'folder', 'root')):
            continue
        if isinstance(value, (str, os.PathLike)):
            path_fields[name] = os.path.abspath(os.path.expanduser(os.fspath(value)))
    return path_fields


def _markdown_cell(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def write_training_run_document(
    out_folder,
    cfg,
    args,
    *,
    model=None,
    train_size=None,
    val_size=None,
    effective_settings=None,
):
    """Write a unique Markdown record immediately before a training run starts."""
    started_at = datetime.now().astimezone()
    output_dir = Path(out_folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"training_run_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
        f"_pid{os.getpid()}.md"
    )
    document_path = output_dir / filename

    cwd = Path.cwd().resolve()
    git_root = _git_output(['rev-parse', '--show-toplevel'], cwd)
    git_branch = _git_output(['branch', '--show-current'], cwd)
    git_commit = _git_output(['rev-parse', 'HEAD'], cwd)
    git_status = _git_output(['status', '--short'], cwd)
    config_text = cfg.dump().strip() if hasattr(cfg, 'dump') else str(cfg).strip()
    arguments = {key: value for key, value in sorted(vars(args).items())}
    input_paths = _resolved_input_paths(args)
    effective_settings = dict(effective_settings or {})

    model_name = type(model).__name__ if model is not None else 'unavailable'
    total_parameters = (
        sum(parameter.numel() for parameter in model.parameters())
        if model is not None else None
    )
    trainable_parameters = (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if model is not None else None
    )

    command = ' '.join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    lines = [
        '# Training Run Record',
        '',
        '## Run identity',
        '',
        f'- Started: `{started_at.isoformat()}`',
        f'- Process ID: `{os.getpid()}`',
        f'- Working directory: `{cwd}`',
        f'- Output directory: `{output_dir}`',
        f'- Command: `{command}`',
        '',
        '## Git state',
        '',
        f'- Repository: `{git_root or "unavailable"}`',
        f'- Branch: `{git_branch or "detached/unavailable"}`',
        f'- Commit: `{git_commit or "unavailable"}`',
        f'- Worktree: `{"dirty" if git_status else "clean"}`',
        '',
        '```text',
        git_status or '(clean or unavailable)',
        '```',
        '',
        '## Training objects',
        '',
        f'- Model: `{model_name}`',
        f'- Total parameters: `{total_parameters if total_parameters is not None else "unavailable"}`',
        f'- Trainable parameters: `{trainable_parameters if trainable_parameters is not None else "unavailable"}`',
        f'- Training samples: `{train_size if train_size is not None else "unavailable"}`',
        f'- Validation samples: `{val_size if val_size is not None else "not used"}`',
        '',
        '## Effective runtime settings',
        '',
        '| Setting | Value |',
        '|---|---|',
    ]
    for name, value in sorted(effective_settings.items()):
        lines.append(f'| {_markdown_cell(name)} | {_markdown_cell(value)} |')

    lines.extend([
        '',
        '## Runtime environment',
        '',
        f'- Python: `{platform.python_version()}`',
        f'- Python executable: `{sys.executable}`',
        f'- Platform: `{platform.platform()}`',
        '',
        '## Resolved input paths',
        '',
        '| Argument | Absolute path |',
        '|---|---|',
    ])
    if input_paths:
        for name, value in sorted(input_paths.items()):
            lines.append(f'| {_markdown_cell(name)} | `{_markdown_cell(value)}` |')
    else:
        lines.append('| — | No path-like arguments detected |')

    lines.extend([
        '',
        '## Command-line arguments',
        '',
        '```json',
        json.dumps(arguments, indent=2, ensure_ascii=False, default=str),
        '```',
        '',
        '## Merged configuration',
        '',
        '```yaml',
        config_text,
        '```',
        '',
    ])

    document_path.write_text('\n'.join(lines), encoding='utf-8')
    return document_path
