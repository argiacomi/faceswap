#!/usr/bin/env python3
"""Handling of Faceswap GUI Projects, Tasks and Last Session"""

from __future__ import annotations

import logging
import os
import tkinter as tk
import typing as T
from tkinter import messagebox

from lib.gui import gui_config as cfg
from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import get_module_objects

from .models.project import ProjectFile
from .services.gui_option_applier import GuiOptionApplier
from .services.modified_state_tracker import ModifiedStateTracker
from .services.project_session_state import ProjectSessionState
from .services.project_store import ProjectStore
from .services.recent_files_store import RecentFilesStore
from .services.session_file_picker import SessionFilePicker

if T.TYPE_CHECKING:
    from .utils import FileHandler
    from .utils.config import Config

logger = logging.getLogger(__name__)


class _GuiSession:  # pylint:disable=too-few-public-methods
    """Parent class for GUI Session Handlers.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object

    """

    def __init__(self, config: Config, file_handler: type[FileHandler] | None = None) -> None:
        # NB file_handler has to be passed in to avoid circular imports
        logger.debug(parse_class_init(locals()))
        self._serializer = get_serializer("json")
        self._config = config
        self._project_store = ProjectStore(self._serializer)
        recent_filename = os.path.join(self._config.path_cache, ".recent.json")
        self._recent_files = RecentFilesStore(self._serializer, recent_filename)
        self._modified_tracker: ModifiedStateTracker | None = None
        self._option_applier: GuiOptionApplier | None = None
        self._file_picker = None if file_handler is None else SessionFilePicker(file_handler)
        self._saved_tasks = None
        self._modified = False
        self._state = ProjectSessionState()

    @property
    def _active_tab(self) -> str:
        """The name of the currently selected :class:`lib.gui.command.CommandNotebook` tab"""
        notebook = self._config.command_notebook
        assert notebook is not None
        tools_book = self._config.tools_notebook
        command = notebook.tab(notebook.select(), "text").lower()
        if command == "tools":
            command = tools_book.tab(tools_book.select(), "text").lower()
        logger.debug("Active tab: %s", command)
        return command

    @property
    def _modified_vars(self) -> dict[str, tk.BooleanVar]:
        """The tkinter Boolean vars indicating the modified state for each tab."""
        return self._config.modified_vars

    @property
    def _tracker(self) -> ModifiedStateTracker:
        """The modified-state tracker, initialized after command tabs exist."""
        if self._modified_tracker is None:
            self._modified_tracker = ModifiedStateTracker(self._modified_vars)
        return self._modified_tracker

    @property
    def _applier(self) -> GuiOptionApplier:
        """The GUI option applier, initialized after Config owns GUI objects."""
        if self._option_applier is None:
            self._option_applier = GuiOptionApplier(
                self._config.cli_opts,
                self._config.set_active_tab_by_name,
            )
        return self._option_applier

    @property
    def _file_exists(self) -> bool:
        """``True`` if the current session state has an existing file."""
        return self._state.has_file

    @property
    def _cli_options(self) -> dict[str, dict[str, bool | int | float | str]]:
        """The raw cli options from session state with project fields removed."""
        return self._state.cli_options

    @property
    def _default_options(self) -> dict[str, T.Any]:
        """The default options for all tabs"""
        return self._config.default_options

    @property
    def _selected_to_choices(self) -> dict[str, dict[str, dict[str, T.Any]]]:
        """The selected value and valid choices for multi-option, radio or combo options."""
        # TODO do instance check on CliOption. Not done for now due to circular import
        # pylint:disable=line-too-long
        valid_choices = {
            cmd: {
                opt: {
                    "choices": val.panel_option.choices,  # pyright:ignore[reportAttributeAccessIssue]  # noqa: E501
                    "is_multi": val.panel_option.is_multi_option,  # pyright:ignore[reportAttributeAccessIssue]  # noqa: E501
                }
                for opt, val in data.items()
                if hasattr(val, "panel_option")  # Filter out helptext
                and val.panel_option.choices is not None  # pyright:ignore[reportAttributeAccessIssue]  # noqa: E501
            }
            for cmd, data in self._config.cli_opts.opts.items()
        }
        logger.trace("valid_choices: %s", valid_choices)  # type:ignore[attr-defined]
        options_by_command = self._state.options
        assert options_by_command is not None
        retval = {
            command: {
                option: {
                    "value": value,
                    "is_multi": valid_choices[command][option]["is_multi"],
                    "choices": valid_choices[command][option]["choices"],
                }
                for option, value in options.items()
                if value and command in valid_choices and option in valid_choices[command]
            }
            for command, options in options_by_command.items()
            if isinstance(options, dict)
        }
        logger.trace("returning: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _current_gui_state(
        self, command: str | None = None
    ) -> dict[str, dict[str, bool | int | float | str]]:
        """The current state of the GUI.

        Parameters
        ----------
        command
            If provided, returns the state of just the given tab command. If ``None`` returns
            options for all tabs. Default ``None``

        Returns
        -------
        The options currently set in the GUI
        """
        return self._config.cli_opts.get_option_values(command)

    def _set_filename(
        self,
        filename: str | None = None,
        session_type: T.Literal["all", "project", "task"] = "project",
    ) -> bool:
        """Set the current session filename."""
        logger.debug("filename: '%s', session_type: '%s'", filename, session_type)
        assert self._file_picker is not None

        picked = self._file_picker.open(session_type, filename)
        if picked is None:
            logger.debug(
                "No valid filename selected. session_type: '%s', filename: '%s'",
                session_type,
                filename,
            )
            return False

        logger.debug("Setting filename: '%s'", picked.filename)
        self._state.set_filename(picked.filename)
        return True

    # GUI STATE SETTING
    def _set_options(self, command: str | None = None) -> None:
        """Set the GUI options based on the current session state and set the active tab.

        Parameters
        ----------
        command
            The tab to set the options for. If None then sets options for all tabs.
            Default: ``None``
        """
        options = self._state.options
        assert options is not None
        logger.debug("command: %s, options: %s", command, options)
        applied = self._applier.apply_project(options, command=command)
        if applied:
            return

        self._config.tk_vars.console_clear.set(True)
        logger.info("No %s section found in file", command)

    def _reset_modified_var(self, command: str | None = None) -> None:
        """Reset :attr:`_modified_vars` variables back to unmodified (`False`) for all
        commands or for the given command.

        Parameters
        ----------
        command
            The command to reset the modified tkinter variable for. If ``None`` then all tkinter
            modified variables are reset to `False`. Default: ``None``
        """
        logger.debug("Reset modified state for: %s", command)
        self._tracker.reset(command)

    # RECENT FILE HANDLING
    def _add_to_recent(self, command: str | None = None) -> None:
        """Add the file for this session to the recent files list.

        Parameters
        ----------
        command
            The command that this session relates to. If `None` then the whole project is added.
            Default: ``None``
        """
        logger.debug(command)
        filename = self._state.filename
        if filename is None:
            logger.debug("No filename for selected file. Not adding to recent.")
            return
        f_type = "project" if command is None else command
        self._recent_files.add(filename, f_type)

    def _del_from_recent(
        self,
        filename: str,
        recent_files: list[tuple[str, str]] | None = None,
        save: bool = False,
    ) -> list[tuple[str, str]] | None:
        """Remove an item from the recent files list.

        Parameters
        ----------
        filename
            The filename to be removed from the recent files list
        recent_files
            If the recent files list has already been loaded, it can be passed in to avoid
            loading again. If ``None`` then load the recent files list from disk. Default: ``None``
        save
            Whether the recent files list should be saved after removing the file. ``True`` saves
            the file, ``False`` does not. Default: ``False``

        Returns
        -------
        List of recent files and their filetypes
        """
        if recent_files is None:
            removed = self._recent_files.remove(filename, save=save)
        else:
            removed = [
                item
                for item in RecentFilesStore.decode_many(recent_files)
                if item.filename != filename
            ]
            if save:
                self._recent_files.save(removed)
        return [(item.filename, item.kind) for item in removed]

    def _get_lone_task(self) -> str | None:
        """Get the sole command name from the current session state.

        Returns
        -------
        The only existing command name in current session options or ``None`` if there
        are multiple commands stored.
        """
        command = None
        if len(self._cli_options) == 1:
            command = list(self._cli_options.keys())[0]
        logger.debug(command)
        return command

    # DISK IO
    def _load(self) -> bool:
        """Load GUI options from the current session state's file location.

        Returns
        -------
        ``True`` if successfully loaded otherwise ``False``
        """
        if not self._file_exists:
            logger.debug("File doesn't exist. Aborting")
            return False

        logger.debug("Loading config")
        try:
            self._load_state()
        except ValueError as err:
            logger.error("Invalid project file: %s", err)
            messagebox.showerror("Invalid Project File", str(err))
            return False
        self._check_valid_choices()
        return True

    def _load_state(self) -> None:
        """Load project/task options from disk into session state."""
        filename = self._state.filename
        assert filename is not None
        project_file = self._project_store.load(filename)
        self._state.load(filename, project_file)

    def _check_valid_choices(self) -> None:
        """Check whether the loaded file has any selected combo/radio/multi-option values that are
        no longer valid and remove them so that they are not passed into faceswap."""
        options_by_command = self._state.options
        assert options_by_command is not None
        for command, options in self._selected_to_choices.items():
            opts = T.cast(dict[str, bool | int | float | str], options_by_command[command])
            for option, data in options.items():
                if not data["is_multi"] and data["value"] in data["choices"]:
                    continue
                if (
                    data["is_multi"]
                    and isinstance(data["value"], str)
                    and all(v in data["choices"] for v in data["value"].split())
                ):
                    continue
                if data["is_multi"] and isinstance(data["value"], str):
                    val = " ".join([v for v in data["value"].split() if v in data["choices"]])
                else:
                    val = ""
                val = val if val else self._default_options[command][option]
                logger.debug(
                    "Updating invalid value to default: (command: '%s', option: '%s', "
                    "original value: '%s', new value: '%s')",
                    command,
                    option,
                    opts[option],
                    val,
                )
                opts[option] = val

    def _save_as_to_filename(self, session_type: T.Literal["all", "task", "project"]) -> bool:
        """Set the current session filename from a save-as dialog."""
        logger.debug("Popping save as file handler. session_type: '%s'", session_type)
        assert self._file_picker is not None

        title = f"Save {f'{session_type.title()} ' if session_type != 'all' else ''}As..."
        picked = self._file_picker.save_as(
            session_type,
            title=title,
            initial_folder=self._state.dirname,
        )

        if picked is None:
            logger.debug("No filename provided. session_type: '%s'", session_type)
            return False

        self._state.set_filename(picked.filename)
        logger.debug(
            "Set filename: (session_type: '%s', filename: '%s')",
            session_type,
            self._state.filename,
        )
        return True

    def _save(self, command: str | None = None) -> None:
        """Collect the options in the current GUI state and save.

        Obtains the current options set in the GUI with the selected tab and applies them to
        session state. Saves the versioned project file to the current filename. Resets modified
        vars for either the given command or all commands,

        Parameters
        ----------
        command
            The tab to collect the current state for. If ``None`` then collects the current
            state for all tabs. Default: ``None``
        """
        tasks = self._current_gui_state(command)
        project_file = ProjectFile(tab_name=self._active_tab, tasks=tasks)
        self._state.set_project(project_file)
        logger.debug(
            "Saving options: (filename: %s, options: %s",
            self._state.filename,
            self._state.options,
        )
        filename = self._state.filename
        assert filename is not None
        self._project_store.save(filename, project_file)
        self._reset_modified_var(command)
        self._add_to_recent(command)


class Tasks(_GuiSession):
    """Faceswap ``.fst`` Task File handling.

    Faceswap tasks handle the management of each individual task tab in the GUI. Unlike
    :class:`Projects`, Tasks contains all the active tasks currently running, rather than an
    individual task.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object
    """

    def __init__(self, config: Config, file_handler: type[FileHandler]):
        super().__init__(config, file_handler)
        self._tasks: dict[
            str,
            dict[
                T.Literal["filename", "options", "is_project"],
                str | bool | dict[str, str | dict[str, bool | int | float | str]] | None,
            ],
        ] = {}

    @property
    def _is_project(self) -> bool:
        """``True`` if all tasks are from an overarching session project else ``False``."""
        retval = (
            False
            if not self._tasks
            else all(v.get("is_project", False) for v in self._tasks.values())
        )
        return retval

    @property
    def _project_filename(self) -> str | None:
        """The overarching session project filename."""
        fname = None
        if not self._is_project:
            return fname

        for val in self._tasks.values():
            fname = val["filename"]
            break
        assert fname is None or isinstance(fname, str)
        return fname

    def load(
        self,  # pylint:disable=unused-argument
        *args,
        filename: str | None = None,
        current_tab: bool = True,
    ) -> None:
        """Load a task into this :class:`Tasks` class.

        Tasks can be loaded from project ``.fsw`` files or task ``.fst`` files, depending on where
        this function is being called from.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        filename
            If a filename is passed in, This will be used, otherwise a file handler will be
            launched to select the relevant file.
        current_tab
            ``True`` if the task to be loaded must be for the currently selected tab. ``False``
            if loading a task into any tab. If current_tab is `True` then tasks can be loaded from
            ``.fsw`` and ``.fst`` files, otherwise they can only be loaded from ``.fst`` files.
            Default: ``True``
        """
        logger.debug(
            "Loading task config: (filename: '%s', current_tab: '%s')",
            filename,
            current_tab,
        )

        # Option to load specific task from project files:
        session_type: T.Literal["all", "task"] = "all" if current_tab else "task"

        is_legacy = (
            not self._is_project
            and filename is not None
            and session_type == "task"
            and os.path.splitext(filename)[1] == ".fsw"
        )
        if is_legacy:
            logger.debug("Legacy task found: '%s'", filename)
            assert filename is not None
            filename = self._update_legacy_task(filename)

        filename_set = self._set_filename(filename, session_type=session_type)
        if not filename_set:
            return
        loaded = self._load()
        if not loaded:
            return

        command = self._active_tab if current_tab else self._state.tab_name
        command = self._get_lone_task() if command is None else command
        if command is None:
            logger.error("Unable to determine task from the given file: '%s'", filename)
            return
        options = self._state.options
        assert options is not None
        if command not in options:
            logger.error("No '%s' task in '%s'", command, self._state.filename)
            return

        self._set_options(command)
        self._add_to_recent(command)

        if self._is_project:
            self._state.set_filename(self._project_filename)
        elif self._state.filename is not None and self._state.filename.endswith(".fsw"):
            self._state.clear_filename()

        self._add_task(command)
        if is_legacy:
            self.save()

        logger.debug("Loaded task config: (command: '%s', filename: '%s')", command, filename)

    def _update_legacy_task(self, filename: str) -> str:
        """Update legacy ``.fsw`` tasks to ``.fst`` tasks.

        Tasks loaded from the recent files menu may be passed in with a ``.fsw`` extension.
        This renames the file and removes it from the recent file list.

        Parameters
        ----------
        filename
            The filename of the `.fsw` file that needs converting

        Returns
        -------
        The new filename of the updated tasks file
        """
        # TODO remove this code after a period of time. Implemented November 2019
        logger.debug("original filename: '%s'", filename)
        fname, ext = os.path.splitext(filename)
        if ext != ".fsw":
            logger.debug("Not a .fsw file: '%s'", filename)
            return filename

        new_filename = f"{fname}.fst"
        logger.debug("Renaming '%s' to '%s'", filename, new_filename)
        os.rename(filename, new_filename)
        self._del_from_recent(filename, save=True)
        logger.debug("new filename: '%s'", new_filename)
        return new_filename

    def save(self, save_as: bool = False) -> None:
        """Save the current GUI state for the active tab to a ``.fst`` faceswap task file.

        Parameters
        ----------
        save_as
            Whether to save to the stored filename, or pop open a file handler to ask for a
            location. If there is no stored filename, then a file handler will automatically be
            popped. Default: ``False``
        """
        logger.debug("Saving config...")
        self._set_active_task()
        save_as = save_as or self._is_project or self._state.filename is None

        if save_as and not self._save_as_to_filename("task"):
            return

        command = self._active_tab
        self._save(command=command)
        self._add_task(command)
        if not save_as:
            logger.info("Saved project to: '%s'", self._state.filename)
        else:
            logger.debug("Saved project to: '%s'", self._state.filename)

    def clear(self) -> None:
        """Reset all GUI options to their default values for the active tab."""
        self._config.cli_opts.reset(self._active_tab)

    def reload(self) -> None:
        """Reset currently selected tab GUI options to their last saved state."""
        self._set_active_task()

        if self._state.options is None:
            logger.info("No active task to reload")
            return
        logger.debug("Reloading task")
        self.load(filename=self._state.filename, current_tab=True)
        if self._is_project:
            self._reset_modified_var(self._active_tab)

    def _add_task(self, command: str) -> None:
        """Add the currently active task to the internal :attr:`_tasks` dict.

        If the currently stored task is from an overarching session project, then
        only the options are updated. When resetting a tab to saved a project will always
        be preferred to a task loaded into the project, so the original reference file name
        stays with the project.

        Parameters
        ----------
        command
            The tab that pertains to the currently active task
        """
        self._tasks[command] = {
            "filename": self._state.filename,
            "options": self._state.options,
            "is_project": self._is_project,
        }

    def clear_tasks(self) -> None:
        """Clears all of the stored tasks.

        This is required when loading a task stored in a legacy project file, and is only to be
        called by :class:`Project` when a project has been loaded which is in fact a task.
        """
        logger.debug("Clearing stored tasks")
        self._tasks = {}

    def add_project_task(
        self,
        filename: str,
        command: str,
        options: dict[str, str | dict[str, bool | int | float | str]],
    ) -> None:
        """Add an individual task from a loaded :class:`Project` to the internal :attr:`_tasks`
        dict.

        Project tasks take priority over any other tasks, so the individual tasks from a new
        project must be placed in the _tasks dict.

        Parameters
        ----------
        filename
            The filename of the session project file
        command
            The tab that this task's options belong to
        options
            The options for this task loaded from the project
        """
        self._tasks[command] = {
            "filename": filename,
            "options": options,
            "is_project": True,
        }

    def _set_active_task(self, command: str | None = None) -> None:
        """Set session state to the currently selected tab's saved task.

        Parameters
        ----------
        command
            If a command is passed in then set the given tab to active, If this is none set the tab
            which currently has focus to active. Default: ``None``
        """
        logger.debug(command)
        command = self._active_tab if command is None else command
        task = self._tasks.get(command, None)
        if task is None:
            self._state.clear()
        else:
            filename = task.get("filename", None)
            opts = task.get("options", None)
            assert filename is None or isinstance(filename, str)
            assert opts is None or isinstance(opts, dict)
            self._state.set_legacy(filename, opts)
        logger.debug(
            "tab: %s, filename: %s, options: %s",
            self._active_tab,
            self._state.filename,
            self._state.options,
        )


class Project(_GuiSession):
    """Faceswap ``.fsw`` Project File handling.

    Faceswap projects handle the management of all task tabs in the GUI and updates
    the main Faceswap title bar with the project name and modified state.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object
    """

    def __init__(self, config: Config, file_handler: type[FileHandler]) -> None:
        super().__init__(config, file_handler)
        self._update_root_title()

    @property
    def filename(self) -> str | None:
        """The currently active project filename."""
        return self._state.filename

    @property
    def cli_options(self) -> dict[str, dict[str, bool | int | float | str]]:
        """The raw cli options from :attr:`_options` with project fields removed."""
        return self._cli_options

    @property
    def _project_modified(self) -> bool:
        """``True`` if the project has been modified otherwise ``False``."""
        return self._tracker.any_modified()

    @property
    def _tasks(self) -> Tasks:
        """The current session's :class:``Tasks``."""
        return self._config.tasks

    def set_default_options(self) -> None:
        """Set the default options. The Default GUI options are stored on Faceswap startup.

        Exposed as the :attr:`_default_options` for a project cannot be set until after the main
        Command Tabs have been loaded.
        """
        logger.debug("Setting options to default")
        self._state.set_options(
            T.cast(
                dict[str, str | dict[str, bool | int | float | str]],
                self._default_options,
            )
        )

    # MODIFIED STATE CALLBACK
    def set_modified_callback(self) -> None:
        """Adds a callback to each of the :attr:`_modified_vars` tkinter variables
        When one of these variables is changed, triggers :func:`_modified_callback`
        with the command that was changed.

        This is exposed as the callback can only be added after the main Command Tabs have
        been drawn, and their options' initial values have been set."""
        for key, tk_var in self._modified_vars.items():
            logger.debug("Adding callback for tab: %s", key)
            tk_var.trace("w", self._modified_callback)

    def _modified_callback(self, *args) -> None:  # pylint:disable=unused-argument
        """Update the project modified state on a GUI modification change and
        update the Faceswap title bar."""
        if self._project_modified and self._current_gui_state() == self._cli_options:
            logger.debug("Project is same as stored. Setting modified to False")
            self._reset_modified_var()

        if self._modified != self._project_modified:
            logger.debug(
                "Updating project state from variable: (modified: %s)",
                self._project_modified,
            )
            self._modified = self._project_modified
            self._update_root_title()

    def load(
        self,  # pylint:disable=unused-argument
        *args,
        filename: str | None = None,
        last_session: bool = False,
    ) -> None:
        """Load a project from a saved ``.fsw`` project file.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        filename
            If a filename is passed in, This will be used, otherwise a file handler will be
            launched to select the relevant file.
        last_session
            ``True`` if the project is being loaded from the last opened session ``False`` if the
            project is being loaded directly from disk. Default: ``False``
        """
        logger.debug(
            "Loading project config: (filename: '%s', last_session: %s)",
            filename,
            last_session,
        )
        filename_set = self._set_filename(filename, session_type="project")

        if not filename_set:
            logger.debug("No filename set")
            return
        loaded = self._load()
        if not loaded:
            logger.debug("Options not loaded")
            return

        # Legacy .fsw files could store projects or tasks. Check if this is a legacy file
        # and hand off file to Tasks if necessary
        command = self._get_lone_task()
        legacy = command is not None
        if legacy:
            self._handoff_legacy_task()
            return

        if not last_session:
            self._set_options()  # Options will be set by last session. Don't set now
        self._update_tasks()
        self._add_to_recent()
        self._reset_modified_var()
        self._update_root_title()
        logger.debug("Loaded project config: (command: '%s', filename: '%s')", command, filename)

    def _handoff_legacy_task(self) -> None:
        """Update legacy tasks saved with the old file extension ``.fsw`` to tasks ``.fst``.

        Hands off file handling to :class:`Tasks` and resets project to default."""
        logger.debug("Updating legacy task '%s", self._state.filename)
        filename = self._state.filename
        self._state.clear_filename()
        self.set_default_options()
        self._tasks.clear_tasks()
        self._tasks.load(filename=filename, current_tab=False)
        logger.debug("Updated legacy task and reset project")

    def _update_tasks(self) -> None:
        """Add the tasks from the loaded project to the :class:`Tasks` class."""
        filename = self._state.filename
        assert filename is not None
        for key, val in self._cli_options.items():
            opts: dict[str, str | dict[str, bool | int | float | str]] = {key: val}
            opts["tab_name"] = key
            self._tasks.add_project_task(filename, key, opts)

    def reload(self, *args) -> None:  # pylint:disable=unused-argument
        """Reset all GUI's option tabs to their last saved state.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        if self._state.options is None:
            logger.info("No active project to reload")
            return
        logger.debug("Reloading project")
        self._set_options()
        self._update_tasks()
        self._reset_modified_var()
        self._update_root_title()

    def _update_root_title(self) -> None:
        """Update the root Window title with the project name. Add a asterisk if the file is
        modified."""
        text = "<untitled project>" if self._state.basename is None else self._state.basename
        text += "*" if self._modified else ""
        self._config.set_root_title(text=text)

    def save(self, *args, save_as: bool = False) -> None:  # pylint:disable=unused-argument
        """Save the current GUI state to a ``.fsw`` project file.

        Parameters
        ----------
        *args: tuple
            Unused, but needs to be present for arguments passed by tkinter event handling
        save_as: bool, optional
            Whether to save to the stored filename, or pop open a file handler to ask for a
            location. If there is no stored filename, then a file handler will automatically be
            popped.
        """
        logger.debug("Saving config as...")

        save_as = save_as or self._state.filename is None
        if save_as and not self._save_as_to_filename("project"):
            return
        self._save()
        self._update_tasks()
        self._update_root_title()
        if not save_as:
            logger.info("Saved project to: '%s'", self._state.filename)
        else:
            logger.debug("Saved project to: '%s'", self._state.filename)

    def new(self, *args) -> None:  # pylint:disable=unused-argument
        """Create a new project with default options.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        logger.debug("Creating new project")

        if not self.confirm_close():
            logger.debug("New project cancelled")
            return

        if not self._save_as_to_filename("project"):
            logger.debug("No filename selected for new project")
            return

        self.set_default_options()
        self._set_options()
        self._tasks.clear_tasks()
        self._reset_modified_var()
        self._modified = False
        self._save()
        self._update_root_title()

        logger.debug("Created new project: '%s'", self._state.filename)

    def close(self, *args) -> None:  # pylint:disable=unused-argument
        """Clear the current project and set all options to default.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        logger.debug("Close requested")
        if not self.confirm_close():
            logger.debug("Close cancelled")
            return
        self._config.cli_opts.reset()
        self._state.clear()
        self.set_default_options()
        self._reset_modified_var()
        self._update_root_title()
        self._config.set_active_tab_by_name(cfg.tab())

    def confirm_close(self) -> bool:
        """Pop a message box to get confirmation that an unsaved project should be closed

        Returns
        -------
        ``True`` if user confirms close, ``False`` if user cancels close
        """
        if not self._modified:
            logger.debug("Project is not modified")
            return True
        confirm_txt = "You have unsaved changes.\n\nAre you sure you want to close the project?"
        if messagebox.askokcancel("Close", confirm_txt, default="cancel", icon="warning"):
            logger.debug("Close Cancelled")
            return True
        logger.debug("Close confirmed")
        return False


class LastSession(_GuiSession):
    """Faceswap Last Session handling.

    Faceswap :class:`LastSession` handles saving the state of the Faceswap GUI at close and
    reloading the state  at launch.

    Last Session behavior can be configured in :file:`config.gui.ini`.

    Parameters
    ----------
    config
        The master GUI config
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._state.set_filename(os.path.join(self._config.path_cache, ".last_session.json"))
        if not self._enabled:
            return

        if cfg.autosave_last_session() == "prompt":
            self.ask_load()
        elif cfg.autosave_last_session() == "always":
            self.load()

    @property
    def _enabled(self) -> bool:
        """``True`` if autosave is enabled otherwise ``False``."""
        return cfg.autosave_last_session() != "never"

    def from_dict(self, options: dict[str, str | dict[str, bool | int | float | str]]) -> None:
        """Set session options from the given dictionary and update the GUI.

        This function is required for reloading the GUI state when the GUI has been force
        refreshed on a config change.

        Parameters
        ----------
        options
            The options to set. Should be the output of :func:`to_dict`
        """
        logger.debug("Setting options from dict: %s", options)
        self._state.set_options(options)
        self._set_options()

    def to_dict(self) -> dict[str, str | dict[str, bool | int | float | str]] | None:
        """Collect the current GUI options and place them in a dict for retrieval or storage.

        This function is required for reloading the GUI state when the GUI has been force
        refreshed on a config change.

        Returns
        -------
        The current cli options ready for saving or retrieval by :func:`from_dict`
        """
        opts = T.cast(
            dict[str, str | dict[str, bool | int | float | str]],
            self._current_gui_state(),
        )
        logger.debug("Collected opts: %s", opts)
        if not opts or opts == self._default_options:
            logger.debug("Default session, or no opts found. Not saving last session.")
            return None
        opts["tab_name"] = self._active_tab
        fname = self._config.project.filename
        if fname is not None:
            opts["project"] = fname
        logger.debug(
            "Added project items: %s",
            {k: v for k, v in opts.items() if k in ("tab_name", "project")},
        )
        return opts

    def _load_state(self) -> None:
        """Load last-session payloads without project/task model coercion."""
        filename = self._state.filename
        assert filename is not None
        assert self._serializer is not None
        self._state.set_options(
            T.cast(
                dict[str, str | dict[str, bool | int | float | str]],
                self._serializer.load(filename),  # pyright:ignore[reportCallIssue]
            )
        )

    def ask_load(self) -> None:
        """Pop a message box to ask the user if they wish to load their last session."""
        if not self._file_exists:
            logger.debug("No last session file found")
        elif messagebox.askyesno("Last Session", "Load last session?"):
            logger.debug("Loading last session at user request")
            self.load()
        else:
            logger.debug("Not loading last session at user request")
            logger.debug("Deleting LastSession file")
            filename = self._state.filename
            assert filename is not None
            os.remove(filename)

    def load(self) -> None:
        """Load the last session.

        Loads the last saved session options. Checks if a previous project was loaded
        and whether there have been changes since the last saved version of the project.
        Sets the display and :class:`Project` and :class:`Task` objects accordingly."""
        loaded = self._load()
        if not loaded:
            return
        self._set_project()
        self._set_options()

    def _set_project(self) -> None:
        """Set the :class:`Project` if session is resuming from one."""
        options = self._state.options
        assert options is not None
        if options.get("project", None) is None:
            logger.debug("No project stored")
        else:
            logger.debug("Loading stored project")
            fname = options["project"]
            assert isinstance(fname, str)
            self._config.project.load(filename=fname, last_session=True)

    def save(self) -> None:
        """Save a snapshot of currently set GUI config options.

        Called on Faceswap shutdown.
        """
        filename = self._state.filename
        assert filename is not None
        if not self._enabled:
            logger.debug("LastSession not enabled")
            if os.path.exists(filename):
                logger.debug("Deleting existing LastSession file")
                os.remove(filename)
            return

        opts = self.to_dict()
        if opts is None and os.path.exists(filename):
            logger.debug("Last session default or blank. Clearing saved last session.")
            os.remove(filename)
        if opts is not None:
            assert self._serializer is not None
            self._serializer.save(filename, opts)  # pyright:ignore[reportCallIssue]
            logger.debug("Saved last session. (filename: %s, options: %s", filename, opts)


__all__ = get_module_objects(__name__)
