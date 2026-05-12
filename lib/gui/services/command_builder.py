#!/usr/bin python3
"""Pure command construction for Faceswap GUI commands."""

from __future__ import annotations

import os
import sys
import typing as T

from lib.utils import get_module_objects


class CommandBuilder:
    """Build Faceswap command line arguments without GUI side effects.

    The ``values`` mapping is expected to contain CLI switches as keys. Values are interpreted as:

    * ``False``, ``None`` or ``""``: omitted
    * ``True``: grouped into short boolean switches when possible
    * sequences: expanded as ``switch value1 value2 ...``
    * anything else: emitted as ``switch value``
    """

    def __init__(self, *, executable: str | None = None, base_path: str | None = None) -> None:
        self._executable = executable or sys.executable
        self._base_path = (
            os.path.realpath(os.path.dirname(sys.argv[0])) if base_path is None else base_path
        )

    def build(
        self,
        category: str,
        command: str,
        values: T.Mapping[str, object],
        *,
        gui_mode: bool = True,
        generate: bool = False,
    ) -> list[str]:
        """Build the command arguments for a Faceswap command.

        Parameters
        ----------
        category
            The script category, for example ``faceswap`` or ``tools``.
        command
            The Faceswap command to execute.
        values
            Mapping of CLI switch to value.
        gui_mode
            Add the GUI marker argument when ``True``. Default: ``True``.
        generate
            Build a display command instead of an executable command. Default: ``False``.

        Returns
        -------
        list[str]
            Command arguments suitable for ``subprocess.Popen`` or display.
        """
        script = os.path.join(self._base_path, f"{category}.py")
        args = [self._executable] if generate else [self._executable, "-u"]
        args.extend([script, command])
        args.extend(self.build_options(values))

        if gui_mode and not generate:
            args.append("-G")

        return self.quote_args(args) if generate else args

    @classmethod
    def quote_args(cls, args: list[str]) -> list[str]:
        """Quote arguments for readable generated command output."""
        return [
            f'"{arg}"'
            if " " in arg and not arg.startswith(("[", "(")) and not arg.endswith(("]", ")"))
            else arg
            for arg in args
        ]

    @classmethod
    def build_options(cls, values: T.Mapping[str, object]) -> list[str]:
        """Build flat CLI switch/value arguments from a mapping."""
        return [arg for group in cls.build_option_groups(values) for arg in group]

    @classmethod
    def build_option_groups(cls, values: T.Mapping[str, object]) -> list[tuple[str, ...]]:
        """Build grouped CLI switch/value arguments from a mapping.

        This preserves the legacy ``gen_cli_arguments()`` tuple shape while sharing the same
        option-emission logic as :meth:`build`.
        """
        groups: list[tuple[str, ...]] = []
        switches = ""

        for switch, value in values.items():
            if cls._is_empty_value(value):
                continue
            if value is True:
                if switch.startswith("-") and not switch.startswith("--"):
                    switches += switch.lstrip("-")
                else:
                    groups.append((switch,))
                continue
            if isinstance(value, list | tuple):
                groups.append((switch, *(str(item) for item in value)))
                continue
            groups.append((switch, str(value)))

        return ([] if not switches else [(f"-{switches}",)]) + groups

    @classmethod
    def _is_empty_value(cls, value: object) -> bool:
        """Return ``True`` for values that should not emit a CLI argument."""
        if value is None or value is False:
            return True
        if isinstance(value, str) and value == "":
            return True
        return bool(isinstance(value, list | tuple) and not value)


__all__ = get_module_objects(__name__)
