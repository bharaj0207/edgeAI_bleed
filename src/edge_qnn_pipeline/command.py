from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOG = logging.getLogger(__name__)


@dataclass
class CommandResult:
    command: list[str]
    return_code: int
    stdout: str
    stderr: str


class CommandRunner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, command: Sequence[str], cwd: Path | None = None) -> CommandResult:
        command_list = list(command)
        rendered = " ".join(shlex.quote(part) for part in command_list)
        LOG.info("Running command: %s", rendered)

        if self.dry_run:
            return CommandResult(command_list, 0, "", "")

        completed = subprocess.run(
            command_list,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            capture_output=True,
        )
        result = CommandResult(command_list, completed.returncode, completed.stdout, completed.stderr)
        if result.return_code != 0:
            raise RuntimeError(
                f"Command failed ({result.return_code}): {rendered}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
