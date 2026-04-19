from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.core.agent_runtime import LocalCodingAgent


@dataclass(frozen=True)
class ParsedSlashCommand:
    command_name: str
    args: str
    is_mcp: bool


@dataclass(frozen=True)
class SlashCommandResult:
    handled: bool
    should_query: bool
    prompt: str | None = None
    output: str = ''
    transcript: tuple[dict[str, Any], ...] = ()


SlashCommandHandler = Callable[['LocalCodingAgent', str, str], SlashCommandResult]


@dataclass(frozen=True)
class SlashCommandSpec:
    names: tuple[str, ...]
    description: str
    handler: SlashCommandHandler


def parse_slash_command(input_text: str) -> ParsedSlashCommand | None:
    trimmed = input_text.strip()
    if not trimmed.startswith('/'):
        return None

    without_slash = trimmed[1:]
    words = without_slash.split(' ')
    if not words or not words[0]:
        return None

    command_name = words[0]
    is_mcp = False
    args_start_index = 1
    if len(words) > 1 and words[1] == '(MCP)':
        command_name = f'{command_name} (MCP)'
        is_mcp = True
        args_start_index = 2

    return ParsedSlashCommand(
        command_name=command_name,
        args=' '.join(words[args_start_index:]),
        is_mcp=is_mcp,
    )


def looks_like_command(command_name: str) -> bool:
    return re.search(r'[^a-zA-Z0-9:\-_]', command_name) is None


def preprocess_slash_command(
    agent: 'LocalCodingAgent',
    input_text: str,
) -> SlashCommandResult:
    if not input_text.strip().startswith('/'):
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    parsed = parse_slash_command(input_text)
    if parsed is None:
        return _local_result(
            input_text,
            'Commands are in the form `/command [args]`.',
        )

    normalized_name = (
        parsed.command_name[:-6]
        if parsed.is_mcp and parsed.command_name.endswith(' (MCP)')
        else parsed.command_name
    )
    spec = find_slash_command(normalized_name)
    if spec is None:
        if looks_like_command(parsed.command_name):
            label = normalized_name if parsed.is_mcp else parsed.command_name
            return _local_result(input_text, f'Unknown skill: {label}')
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    return spec.handler(agent, parsed.args.strip(), input_text)


def get_slash_command_specs() -> tuple[SlashCommandSpec, ...]:
    return (
        SlashCommandSpec(
            names=('help', 'commands'),
            description='Show the built-in Python slash commands.',
            handler=_handle_help,
        ),
        SlashCommandSpec(
            names=('context', 'usage'),
            description='Show estimated session context usage similar to the npm /context command.',
            handler=_handle_context,
        ),
        SlashCommandSpec(
            names=('context-raw', 'env'),
            description='Show the raw environment, user context, and system context snapshot.',
            handler=_handle_context_raw,
        ),
        SlashCommandSpec(
            names=('token-budget', 'budget'),
            description='Show the current token-budget window, reserves, and prompt-length limits.',
            handler=_handle_token_budget,
        ),
        SlashCommandSpec(
            names=('mcp',),
            description='Show discovered local MCP manifests and resource counts.',
            handler=_handle_mcp,
        ),
        SlashCommandSpec(
            names=('search',),
            description='Show search runtime status, list or activate providers, or run a real web search query.',
            handler=_handle_search,
        ),
        SlashCommandSpec(
            names=('remote',),
            description='Show local remote runtime status or activate a remote target/profile.',
            handler=_handle_remote,
        ),
        SlashCommandSpec(
            names=('worktree',),
            description='Show managed git worktree status or enter/exit the current managed worktree session.',
            handler=_handle_worktree,
        ),
        SlashCommandSpec(
            names=('account',),
            description='Show local account runtime status or configured account profiles.',
            handler=_handle_account,
        ),
        SlashCommandSpec(
            names=('ask',),
            description='Show local ask-user runtime status or ask-user history.',
            handler=_handle_ask,
        ),
        SlashCommandSpec(
            names=('login',),
            description='Activate a local account profile or ephemeral identity.',
            handler=_handle_login,
        ),
        SlashCommandSpec(
            names=('logout',),
            description='Clear the active local account session.',
            handler=_handle_logout,
        ),
        SlashCommandSpec(
            names=('config', 'settings'),
            description='Show local config runtime state, effective config, config sources, or a config value.',
            handler=_handle_config,
        ),
        SlashCommandSpec(
            names=('lsp',),
            description='Show local LSP runtime status or run document symbols, definition, references, hover, call hierarchy, and diagnostics queries.',
            handler=_handle_lsp,
        ),
        SlashCommandSpec(
            names=('remotes',),
            description='List configured local remote profiles.',
            handler=_handle_remotes,
        ),
        SlashCommandSpec(
            names=('ssh',),
            description='Activate a local SSH remote target/profile.',
            handler=_handle_ssh,
        ),
        SlashCommandSpec(
            names=('teleport',),
            description='Activate a local teleport remote target/profile.',
            handler=_handle_teleport,
        ),
        SlashCommandSpec(
            names=('direct-connect',),
            description='Activate a local direct-connect remote target/profile.',
            handler=_handle_direct_connect,
        ),
        SlashCommandSpec(
            names=('deep-link',),
            description='Activate a local deep-link remote target/profile.',
            handler=_handle_deep_link,
        ),
        SlashCommandSpec(
            names=('disconnect', 'remote-disconnect'),
            description='Disconnect the active local remote runtime target.',
            handler=_handle_remote_disconnect,
        ),
        SlashCommandSpec(
            names=('resources',),
            description='List local MCP resources, optionally filtered by a query string.',
            handler=_handle_resources,
        ),
        SlashCommandSpec(
            names=('resource',),
            description='Render a local MCP resource by URI.',
            handler=_handle_resource,
        ),
        SlashCommandSpec(
            names=('tasks', 'todo'),
            description='Show the local runtime task list, optionally filtered by status.',
            handler=_handle_tasks,
        ),
        SlashCommandSpec(
            names=('workflows',),
            description='List local workflows discovered from workflow manifests.',
            handler=_handle_workflows,
        ),
        SlashCommandSpec(
            names=('workflow',),
            description='Show or run one local workflow by name.',
            handler=_handle_workflow,
        ),
        SlashCommandSpec(
            names=('triggers',),
            description='List local remote triggers discovered from remote trigger manifests.',
            handler=_handle_triggers,
        ),
        SlashCommandSpec(
            names=('trigger',),
            description='Show or run one local remote trigger by id.',
            handler=_handle_trigger,
        ),
        SlashCommandSpec(
            names=('teams',),
            description='List the locally configured collaboration teams.',
            handler=_handle_teams,
        ),
        SlashCommandSpec(
            names=('team',),
            description='Show one local collaboration team by name.',
            handler=_handle_team,
        ),
        SlashCommandSpec(
            names=('messages',),
            description='Show recorded collaboration messages for all teams or one team.',
            handler=_handle_messages,
        ),
        SlashCommandSpec(
            names=('task-next', 'next-task'),
            description='Show the next actionable tasks from the local runtime task list.',
            handler=_handle_task_next,
        ),
        SlashCommandSpec(
            names=('plan', 'planner'),
            description='Show the current local runtime plan.',
            handler=_handle_plan,
        ),
        SlashCommandSpec(
            names=('task',),
            description='Show a local runtime task by id.',
            handler=_handle_task,
        ),
        SlashCommandSpec(
            names=('prompt', 'system-prompt'),
            description='Render the effective Python system prompt.',
            handler=_handle_prompt,
        ),
        SlashCommandSpec(
            names=('permissions',),
            description='Show the active tool permission mode.',
            handler=_handle_permissions,
        ),
        SlashCommandSpec(
            names=('hooks', 'policy'),
            description='Show discovered local hook and policy manifests.',
            handler=_handle_hooks,
        ),
        SlashCommandSpec(
            names=('trust',),
            description='Show workspace trust mode, managed settings, and safe environment values.',
            handler=_handle_trust,
        ),
        SlashCommandSpec(
            names=('model',),
            description='Show or update the active model for the current agent instance.',
            handler=_handle_model,
        ),
        SlashCommandSpec(
            names=('tools',),
            description='List the registered tools and whether the current permissions allow them.',
            handler=_handle_tools,
        ),
        SlashCommandSpec(
            names=('agents',),
            description='List active local agent configurations or show one agent definition by name.',
            handler=_handle_agents,
        ),
        SlashCommandSpec(
            names=('memory',),
            description='Show the currently loaded CLAUDE.md memory bundle and discovered files.',
            handler=_handle_memory,
        ),
        SlashCommandSpec(
            names=('status', 'session'),
            description='Show a short runtime/session status summary.',
            handler=_handle_status,
        ),
        SlashCommandSpec(
            names=('clear',),
            description='Clear ephemeral Python runtime state for this process.',
            handler=_handle_clear,
        ),
        SlashCommandSpec(
            names=('compact',),
            description='Summarise and compact the conversation to free context space.',
            handler=_handle_compact,
        ),
        SlashCommandSpec(
            names=('cost',),
            description='Show the total cost and duration of the current session.',
            handler=_handle_cost,
        ),
        SlashCommandSpec(
            names=('exit', 'quit'),
            description='Exit the REPL.',
            handler=_handle_exit,
        ),
        SlashCommandSpec(
            names=('diff',),
            description='View uncommitted changes (git diff) in the working directory.',
            handler=_handle_diff,
        ),
        SlashCommandSpec(
            names=('files',),
            description='List files currently loaded in the session context.',
            handler=_handle_files,
        ),
        SlashCommandSpec(
            names=('copy',),
            description='Copy the last assistant response to a temp file.',
            handler=_handle_copy,
        ),
        SlashCommandSpec(
            names=('export',),
            description='Export the conversation to a text file.',
            handler=_handle_export,
        ),
        SlashCommandSpec(
            names=('stats',),
            description='Show session usage statistics.',
            handler=_handle_stats,
        ),
        SlashCommandSpec(
            names=('tag',),
            description='Add or remove a searchable tag on the current session.',
            handler=_handle_tag,
        ),
        SlashCommandSpec(
            names=('rename',),
            description='Rename the current conversation.',
            handler=_handle_rename,
        ),
        SlashCommandSpec(
            names=('branch',),
            description='Create a fork/branch of the current conversation.',
            handler=_handle_branch,
        ),
        SlashCommandSpec(
            names=('effort',),
            description='Show or set the model effort level (low, medium, high, max, auto).',
            handler=_handle_effort,
        ),
        SlashCommandSpec(
            names=('doctor',),
            description='Diagnose and verify the claw-code installation and settings.',
            handler=_handle_doctor,
        ),
        SlashCommandSpec(
            names=('commit',),
            description='Create a git commit.',
            handler=_handle_commit,
        ),
        SlashCommandSpec(
            names=('pr-comments', 'pr_comments'),
            description='Get comments from a GitHub pull request.',
            handler=_handle_pr_comments,
        ),
        SlashCommandSpec(
            names=('resume', 'continue'),
            description='Resume a previous conversation.',
            handler=_handle_resume,
        ),
        SlashCommandSpec(
            names=('add-dir',),
            description='Add a new working directory.',
            handler=_handle_add_dir,
        ),
        SlashCommandSpec(
            names=('skills',),
            description='List available skills.',
            handler=_handle_skills,
        ),
        SlashCommandSpec(
            names=('fast',),
            description='Toggle fast mode.',
            handler=_handle_fast,
        ),
        SlashCommandSpec(
            names=('vim',),
            description='Toggle between Vim and Normal editing modes.',
            handler=_handle_vim,
        ),
        SlashCommandSpec(
            names=('rewind', 'checkpoint'),
            description='Restore the conversation to a previous point.',
            handler=_handle_rewind,
        ),
    )


def find_slash_command(command_name: str) -> SlashCommandSpec | None:
    lowered = command_name.lower()
    for spec in get_slash_command_specs():
        if lowered in spec.names:
            return spec
    return None


def _handle_help(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    lines = ['# Slash Commands', '']
    for spec in get_slash_command_specs():
        primary = f'/{spec.names[0]}'
        aliases = ', '.join(f'/{name}' for name in spec.names[1:])
        label = f'{primary} ({aliases})' if aliases else primary
        lines.append(f'- `{label}`: {spec.description}')
    lines.extend(
        [
            '',
            'These commands are handled locally before the model loop, similar to the npm runtime.',
        ]
    )
    return _local_result(input_text, '\n'.join(lines))


def _handle_context(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    prompt = args or None
    return _local_result(input_text, agent.render_context_report(prompt))


def _handle_context_raw(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_context_snapshot_report())


def _handle_token_budget(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_token_budget_report())


def _handle_mcp(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_mcp_report())
    if command == 'tools':
        return _local_result(input_text, agent.render_mcp_tools_report())
    if command.startswith('tools '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_mcp_tools_report(query or None))
    if command.startswith('tool '):
        tool_name = command.split(' ', 1)[1].strip()
        if not tool_name:
            return _local_result(input_text, 'Usage: /mcp tool <tool-name>')
        return _local_result(input_text, agent.render_mcp_call_tool_report(tool_name))
    return _local_result(input_text, agent.render_mcp_report(command))


def _handle_search(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_search_report())
    if command == 'providers':
        return _local_result(input_text, agent.render_search_providers_report())
    if command.startswith('providers '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_search_providers_report(query or None))
    if command.startswith('provider '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search provider <name>')
        return _local_result(input_text, agent.render_search_report(provider=provider))
    if command.startswith('use '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search use <name>')
        return _local_result(input_text, agent.render_search_activate_report(provider))
    return _local_result(input_text, agent.render_search_report(command))


def _handle_remote(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args or None
    return _local_result(input_text, agent.render_remote_report(target))


def _handle_worktree(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_worktree_report())
    if command == 'history':
        return _local_result(input_text, agent.render_worktree_history_report())
    if command.startswith('enter'):
        name = command.split(' ', 1)[1].strip() if ' ' in command else None
        return _local_result(input_text, agent.render_worktree_enter_report(name or None))
    if command.startswith('exit'):
        parts = command.split()
        action = parts[1] if len(parts) > 1 else 'keep'
        discard_changes = any(part in {'discard', 'discard_changes=true'} for part in parts[2:])
        return _local_result(
            input_text,
            agent.render_worktree_exit_report(
                action=action,
                discard_changes=discard_changes,
            ),
        )
    return _local_result(
        input_text,
        'Usage: /worktree [history|enter <name>|exit <keep|remove> [discard]]',
    )


def _handle_account(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_account_report())
    if command == 'profiles':
        return _local_result(input_text, agent.render_account_profiles_report())
    if command.startswith('profile '):
        profile = command.split(' ', 1)[1].strip()
        if not profile:
            return _local_result(input_text, 'Usage: /account profile <name>')
        return _local_result(input_text, agent.render_account_report(profile))
    return _local_result(input_text, 'Usage: /account [profiles|profile <name>]')


def _handle_ask(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_ask_user_report())
    if command == 'history':
        return _local_result(input_text, agent.render_ask_user_history_report())
    return _local_result(input_text, 'Usage: /ask [history]')


def _handle_login(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args.strip()
    if not target:
        return _local_result(input_text, 'Usage: /login <profile-or-identity>')
    return _local_result(input_text, agent.render_account_login_report(target))


def _handle_logout(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_account_logout_report())


def _handle_config(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_config_report())
    if command == 'effective':
        return _local_result(input_text, agent.render_config_effective_report())
    if command.startswith('source '):
        source = command.split(' ', 1)[1].strip()
        if not source:
            return _local_result(input_text, 'Usage: /config source <source-name>')
        return _local_result(input_text, agent.render_config_source_report(source))
    if command.startswith('get '):
        key_path = command.split(' ', 1)[1].strip()
        if not key_path:
            return _local_result(input_text, 'Usage: /config get <key-path>')
        return _local_result(input_text, agent.render_config_value_report(key_path))
    return _local_result(
        input_text,
        'Usage: /config [effective|source <name>|get <key-path>]',
    )


def _handle_lsp(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_lsp_report())
    if command == 'diagnostics':
        return _local_result(input_text, agent.render_lsp_diagnostics_report())
    if command.startswith('diagnostics '):
        file_path = command.split(' ', 1)[1].strip()
        if not file_path:
            return _local_result(input_text, 'Usage: /lsp diagnostics [file-path]')
        return _local_result(input_text, agent.render_lsp_diagnostics_report(file_path))
    if command.startswith('symbols '):
        file_path = command.split(' ', 1)[1].strip()
        if not file_path:
            return _local_result(input_text, 'Usage: /lsp symbols <file-path>')
        return _local_result(input_text, agent.render_lsp_document_symbols_report(file_path))
    if command.startswith('workspace '):
        query = command.split(' ', 1)[1].strip()
        if not query:
            return _local_result(input_text, 'Usage: /lsp workspace <query>')
        return _local_result(input_text, agent.render_lsp_workspace_symbols_report(query))
    parts = command.split()
    if len(parts) == 4 and parts[0] in {
        'definition',
        'references',
        'hover',
        'hierarchy',
        'incoming',
        'outgoing',
    }:
        subcommand, file_path, line_text, character_text = parts
        try:
            line = int(line_text)
            character = int(character_text)
        except ValueError:
            return _local_result(
                input_text,
                f'Usage: /lsp {subcommand} <file-path> <line> <character>',
            )
        if subcommand == 'definition':
            return _local_result(
                input_text,
                agent.render_lsp_definition_report(file_path, line, character),
            )
        if subcommand == 'references':
            return _local_result(
                input_text,
                agent.render_lsp_references_report(file_path, line, character),
            )
        if subcommand == 'hover':
            return _local_result(
                input_text,
                agent.render_lsp_hover_report(file_path, line, character),
            )
        if subcommand == 'hierarchy':
            return _local_result(
                input_text,
                agent.render_lsp_prepare_call_hierarchy_report(file_path, line, character),
            )
        if subcommand == 'incoming':
            return _local_result(
                input_text,
                agent.render_lsp_incoming_calls_report(file_path, line, character),
            )
        if subcommand == 'outgoing':
            return _local_result(
                input_text,
                agent.render_lsp_outgoing_calls_report(file_path, line, character),
            )
    return _local_result(
        input_text,
        'Usage: /lsp [symbols <file>|workspace <query>|definition <file> <line> <character>|references <file> <line> <character>|hover <file> <line> <character>|hierarchy <file> <line> <character>|incoming <file> <line> <character>|outgoing <file> <line> <character>|diagnostics [file]]',
    )


def _handle_remotes(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_profiles_report(query))


def _handle_ssh(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /ssh <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='ssh'))


def _handle_teleport(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /teleport <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='teleport'))


def _handle_direct_connect(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /direct-connect <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='direct-connect'))


def _handle_deep_link(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /deep-link <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='deep-link'))


def _handle_remote_disconnect(
    agent: 'LocalCodingAgent',
    _args: str,
    input_text: str,
) -> SlashCommandResult:
    return _local_result(input_text, agent.render_remote_disconnect_report())


def _handle_resources(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_mcp_resources_report(query))


def _handle_resource(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /resource <mcp-resource-uri>')
    return _local_result(input_text, agent.render_mcp_resource_report(args))


def _handle_tasks(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    status = args or None
    return _local_result(input_text, agent.render_tasks_report(status))


def _handle_workflows(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_workflows_report(query))


def _handle_workflow(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /workflow <name> | /workflow run <name>')
    if command.startswith('run '):
        workflow_name = command.split(' ', 1)[1].strip()
        if not workflow_name:
            return _local_result(input_text, 'Usage: /workflow run <name>')
        return _local_result(input_text, agent.render_workflow_run_report(workflow_name))
    return _local_result(input_text, agent.render_workflow_report(command))


def _handle_triggers(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_triggers_report(query))


def _handle_trigger(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /trigger <id> | /trigger run <id>')
    if command.startswith('run '):
        trigger_id = command.split(' ', 1)[1].strip()
        if not trigger_id:
            return _local_result(input_text, 'Usage: /trigger run <id>')
        return _local_result(
            input_text,
            agent.render_remote_trigger_action_report('run', trigger_id=trigger_id),
        )
    return _local_result(input_text, agent.render_remote_trigger_report(command))


def _handle_teams(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_teams_report(query))


def _handle_team(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip()
    if not team_name:
        return _local_result(input_text, 'Usage: /team <team-name>')
    return _local_result(input_text, agent.render_team_report(team_name))


def _handle_messages(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip() or None
    return _local_result(input_text, agent.render_team_messages_report(team_name))


def _handle_task_next(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_next_tasks_report())


def _handle_plan(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_plan_report())


def _handle_task(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /task <task-id>')
    return _local_result(input_text, agent.render_task_report(args))


def _handle_prompt(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_system_prompt())


def _handle_permissions(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_permissions_report())


def _handle_hooks(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_hook_policy_report())


def _handle_trust(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_trust_report())


def _handle_model(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, f'Current model: {agent.model_config.model}')
    agent.set_model(args)
    return _local_result(input_text, f'Set model to {agent.model_config.model}')


def _handle_tools(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_tools_report())


def _handle_agents(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    trimmed = args.strip()
    if not trimmed or trimmed in {'active', 'list'}:
        return _local_result(input_text, agent.render_agents_report())
    if trimmed == 'all':
        return _local_result(input_text, agent.render_agents_report(show_all=True))
    if trimmed.startswith('show '):
        agent_type = trimmed[5:].strip()
        if not agent_type:
            return _local_result(input_text, 'Usage: /agents [all|active|show <agent-type>]')
        return _local_result(input_text, agent.render_agent_detail_report(agent_type))
    return _local_result(input_text, agent.render_agent_detail_report(trimmed))


def _handle_memory(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_memory_report())


def _handle_status(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_status_report())


def _handle_clear(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    agent.clear_runtime_state()
    return _local_result(
        input_text,
        'Cleared ephemeral Python agent state for this process.',
    )


def _handle_compact(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .compact import compact_conversation

    custom_instructions = args.strip() if args.strip() else None
    result = compact_conversation(agent, custom_instructions)

    if result.error:
        return _local_result(input_text, f'Compact failed: {result.error}')

    lines = ['Conversation compacted.']
    if result.pre_compact_token_count:
        lines.append(
            f'  Tokens before: ~{result.pre_compact_token_count:,}  '
            f'→  after: ~{result.post_compact_token_count:,}'
        )
    return _local_result(input_text, '\n'.join(lines))


def _handle_cost(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    usage = agent.cumulative_usage
    cost = agent.cumulative_cost_usd

    def _fmt_cost(usd: float) -> str:
        if usd < 0.01:
            return f'${usd:.4f}'
        return f'${usd:.2f}'

    lines = [
        f'Total cost:            {_fmt_cost(cost)}',
        f'Total input tokens:    {usage.input_tokens:,}',
        f'Total output tokens:   {usage.output_tokens:,}',
    ]
    if usage.cache_read_input_tokens:
        lines.append(f'Cache read tokens:     {usage.cache_read_input_tokens:,}')
    if usage.cache_creation_input_tokens:
        lines.append(f'Cache creation tokens:  {usage.cache_creation_input_tokens:,}')
    if usage.reasoning_tokens:
        lines.append(f'Reasoning tokens:      {usage.reasoning_tokens:,}')
    lines.append(f'Total tokens:          {usage.total_tokens:,}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_exit(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    import random
    import sys

    messages = ['Goodbye!', 'See ya!', 'Bye!', 'Catch you later!']
    output = random.choice(messages)
    # Build the result first so the transcript is recorded, then exit.
    result = _local_result(input_text, output)
    print(output)
    sys.exit(0)
    return result  # unreachable, but satisfies the type checker


def _handle_diff(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    import subprocess

    cwd = str(agent.runtime_config.cwd)
    try:
        proc = subprocess.run(
            ['git', 'diff'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        diff_output = proc.stdout.strip()
        if not diff_output:
            # Also check staged changes
            proc_staged = subprocess.run(
                ['git', 'diff', '--staged'],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            diff_output = proc_staged.stdout.strip()
            if not diff_output:
                return _local_result(input_text, 'No uncommitted changes.')
            return _local_result(input_text, f'Staged changes:\n{diff_output}')
        return _local_result(input_text, diff_output)
    except FileNotFoundError:
        return _local_result(input_text, 'git is not available.')
    except subprocess.TimeoutExpired:
        return _local_result(input_text, 'git diff timed out.')
    except Exception as exc:
        return _local_result(input_text, f'Error running git diff: {exc}')


def _handle_files(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """List files loaded in the session context (from readFileState)."""
    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session.')

    # Collect file paths mentioned in tool results
    file_paths: list[str] = []
    for msg in session.messages:
        if msg.role == 'tool' and msg.name in ('Read', 'read_file', 'ReadFile'):
            # Extract path from content or metadata
            path = msg.metadata.get('path')
            if isinstance(path, str):
                file_paths.append(path)
            elif msg.content and msg.content.startswith('/'):
                # First line might be the path
                first_line = msg.content.split('\n', 1)[0].strip()
                if '/' in first_line and len(first_line) < 256:
                    file_paths.append(first_line)

    # Also look at tool_calls in assistant messages
    for msg in session.messages:
        if msg.role == 'assistant' and msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.get('function', {}) if isinstance(tc, dict) else {}
                if func.get('name') in ('Read', 'read_file', 'ReadFile', 'View'):
                    import json as _json
                    try:
                        args = _json.loads(func.get('arguments', '{}'))
                        path = args.get('file_path') or args.get('path')
                        if isinstance(path, str):
                            file_paths.append(path)
                    except (ValueError, TypeError):
                        pass

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in file_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        return _local_result(input_text, 'No files loaded in context.')

    cwd = str(agent.runtime_config.cwd)
    relative_paths = []
    for p in unique_paths:
        if p.startswith(cwd):
            relative_paths.append(p[len(cwd):].lstrip('/'))
        else:
            relative_paths.append(p)

    lines = [f'Files in context ({len(relative_paths)}):']
    for p in relative_paths:
        lines.append(f'  {p}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_copy(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Copy the last assistant response to a temp file."""
    import tempfile as _tempfile

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session.')

    # Find the Nth most recent assistant message (default N=0 = latest)
    n = 0
    if args.strip().isdigit():
        n = min(int(args.strip()), 20)

    assistant_messages = [
        msg for msg in session.messages
        if msg.role == 'assistant' and msg.content.strip()
    ]
    if not assistant_messages:
        return _local_result(input_text, 'No assistant responses to copy.')

    index = len(assistant_messages) - 1 - n
    if index < 0:
        return _local_result(
            input_text,
            f'Only {len(assistant_messages)} assistant responses available.',
        )

    content = assistant_messages[index].content

    # Write to temp file
    from pathlib import Path as _Path
    tmp_dir = _Path(_tempfile.gettempdir()) / 'claw-code'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / 'response.md'
    out_path.write_text(content, encoding='utf-8')

    char_count = len(content)
    line_count = content.count('\n') + 1
    return _local_result(
        input_text,
        f'Copied {char_count:,} chars ({line_count} lines) to {out_path}',
    )


def _handle_export(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Export the conversation transcript to a text file."""
    from pathlib import Path as _Path
    import time as _time

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session to export.')

    # Build plain-text transcript
    lines: list[str] = []
    for msg in session.messages:
        label = msg.role.upper()
        if msg.role == 'tool' and msg.name:
            label = f'TOOL:{msg.name}'
        lines.append(f'--- {label} ---')
        lines.append(msg.content)
        lines.append('')
    text = '\n'.join(lines)

    # Determine output path
    filename = args.strip()
    if not filename:
        timestamp = _time.strftime('%Y%m%d_%H%M%S')
        filename = f'conversation_{timestamp}.txt'
    if not filename.endswith('.txt'):
        filename += '.txt'

    out_path = _Path(str(agent.runtime_config.cwd)) / filename
    out_path.write_text(text, encoding='utf-8')
    return _local_result(
        input_text,
        f'Exported {len(session.messages)} messages to {out_path}',
    )


def _handle_stats(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """Show session usage statistics."""
    usage = agent.cumulative_usage
    cost = agent.cumulative_cost_usd

    session = agent.last_session
    msg_count = len(session.messages) if session else 0
    user_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'user')
    assistant_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'assistant')
    tool_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'tool')

    lines = [
        '## Session Statistics',
        '',
        f'Messages:     {msg_count} total ({user_msgs} user, {assistant_msgs} assistant, {tool_msgs} tool)',
        f'Input tokens:  {usage.input_tokens:,}',
        f'Output tokens: {usage.output_tokens:,}',
        f'Total tokens:  {usage.total_tokens:,}',
        f'Cost:          ${cost:.4f}',
        f'Model:         {agent.model_config.model}',
    ]
    if agent.active_session_id:
        lines.append(f'Session ID:    {agent.active_session_id}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_tag(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Add or remove a tag on the current session."""
    tag = args.strip()
    if not tag:
        # Show current tags
        tags = getattr(agent, '_session_tags', set())
        if tags:
            return _local_result(input_text, f'Session tags: {", ".join(sorted(tags))}')
        return _local_result(input_text, 'No tags set. Usage: /tag <tag-name>')

    # Toggle tag
    if not hasattr(agent, '_session_tags'):
        agent._session_tags = set()

    if tag in agent._session_tags:
        agent._session_tags.discard(tag)
        return _local_result(input_text, f'Removed tag: {tag}')
    agent._session_tags.add(tag)
    return _local_result(input_text, f'Added tag: {tag}')


def _handle_rename(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Rename the current conversation."""
    name = args.strip()
    if not name:
        return _local_result(input_text, 'Usage: /rename <name>')

    if not hasattr(agent, '_session_name'):
        agent._session_name = None
    agent._session_name = name
    return _local_result(input_text, f'Session renamed to: {name}')


def _handle_branch(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Create a fork/branch of the current conversation."""
    import json as _json
    from uuid import uuid4

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session to branch.')

    branch_name = args.strip() or f'branch-{uuid4().hex[:8]}'
    new_session_id = uuid4().hex

    # Save a copy of the current transcript as a new session file
    session_dir = agent.runtime_config.session_directory
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f'{new_session_id}.json'

    transcript = [msg.to_transcript_entry() for msg in session.messages]
    branch_data = {
        'session_id': new_session_id,
        'branch_name': branch_name,
        'branched_from': agent.active_session_id,
        'messages': transcript,
        'model': agent.model_config.model,
    }

    try:
        session_path.write_text(_json.dumps(branch_data, indent=2), encoding='utf-8')
        return _local_result(
            input_text,
            f'Created branch "{branch_name}" (session: {new_session_id})\n'
            f'Saved to: {session_path}',
        )
    except Exception as exc:
        return _local_result(input_text, f'Error creating branch: {exc}')


def _handle_effort(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Show or set the model effort level."""
    import os

    valid_levels = ('low', 'medium', 'high', 'max', 'auto')
    current = getattr(agent.runtime_config, 'effort_level', None)
    env_override = os.environ.get('CLAUDE_CODE_EFFORT_LEVEL')

    if not args.strip():
        level = current or env_override or 'auto'
        msg = f'Current effort level: {level}'
        if env_override:
            msg += f' (from CLAUDE_CODE_EFFORT_LEVEL env var)'
        return _local_result(input_text, msg)

    level = args.strip().lower()
    if level not in valid_levels:
        return _local_result(
            input_text,
            f'Invalid effort level: {level}\nValid levels: {", ".join(valid_levels)}',
        )

    if env_override:
        return _local_result(
            input_text,
            f'Cannot change effort level — overridden by '
            f'CLAUDE_CODE_EFFORT_LEVEL={env_override}',
        )

    # Store effort level on the runtime config
    object.__setattr__(agent.runtime_config, 'effort_level', level)
    return _local_result(input_text, f'Set effort level to: {level}')


def _handle_doctor(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """Diagnose and verify the claw-code installation."""
    import shutil
    import sys

    checks: list[str] = []

    # Python version
    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    ok = sys.version_info >= (3, 10)
    checks.append(f'{"✓" if ok else "✗"} Python version: {py_ver} (need ≥3.10)')

    # Git available
    git_ok = shutil.which('git') is not None
    checks.append(f'{"✓" if git_ok else "✗"} git: {"found" if git_ok else "NOT FOUND"}')

    # Model config
    checks.append(f'✓ Model: {agent.model_config.model}')
    checks.append(f'✓ Base URL: {agent.model_config.base_url}')

    # Working directory
    cwd = agent.runtime_config.cwd
    checks.append(f'✓ Working directory: {cwd}')
    checks.append(f'{"✓" if cwd.exists() else "✗"} Working directory exists: {cwd.exists()}')

    # Session directory
    sess_dir = agent.runtime_config.session_directory
    checks.append(f'✓ Session directory: {sess_dir}')
    checks.append(f'{"✓" if sess_dir.exists() else "○"} Session directory exists: {sess_dir.exists()}')

    # API key
    has_key = bool(agent.model_config.api_key)
    checks.append(f'{"✓" if has_key else "✗"} API key: {"set" if has_key else "NOT SET"}')

    # Tools
    tool_count = len(agent.tool_registry) if agent.tool_registry else 0
    checks.append(f'✓ Registered tools: {tool_count}')

    # Memory files (CLAUDE.md)
    claude_md = cwd / 'CLAUDE.md'
    checks.append(
        f'{"✓" if claude_md.exists() else "○"} CLAUDE.md: '
        f'{"found" if claude_md.exists() else "not found (optional)"}'
    )

    output = '## Doctor Report\n\n' + '\n'.join(checks)
    return _local_result(input_text, output)


def _handle_commit(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Create a git commit — prompt-type command injecting git context."""
    import subprocess

    cwd = str(agent.runtime_config.cwd)

    def _git(cmd_args: list[str]) -> str:
        try:
            proc = subprocess.run(
                ['git'] + cmd_args,
                cwd=cwd, capture_output=True, text=True, timeout=15,
            )
            return proc.stdout.strip()
        except Exception:
            return ''

    status = _git(['status', '--short'])
    staged = _git(['diff', '--staged'])
    unstaged = _git(['diff'])
    branch = _git(['branch', '--show-current'])
    recent = _git(['log', '--oneline', '-10'])

    sections: list[str] = []
    if branch:
        sections.append(f'Current branch: {branch}')
    if status:
        sections.append(f'### Status\n```\n{status}\n```')
    if staged:
        sections.append(f'### Staged diff\n```\n{staged}\n```')
    if unstaged:
        sections.append(f'### Unstaged diff\n```\n{unstaged}\n```')
    if recent:
        sections.append(f'### Recent commits\n```\n{recent}\n```')

    git_context = '\n\n'.join(sections) if sections else 'No git changes detected.'

    prompt = f"""Create a git commit for the current changes.

{git_context}

Instructions:
1. Review the changes above
2. Stage appropriate files with `git add <file>` (prefer specific files over `git add -A`)
3. Draft a concise commit message following the repo's conventions
4. Use a HEREDOC for multi-line messages:
   git commit -m "$(cat <<'EOF'
   message here
   EOF
   )"

Safety:
- Always create NEW commits (never --amend)
- Never skip hooks (--no-verify)
- Never commit secrets (.env, credentials, etc.)
- No interactive flags (-i)"""

    if args.strip():
        prompt += f'\n\nAdditional instructions: {args.strip()}'

    return _prompt_result(input_text, prompt)


def _handle_pr_comments(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Fetch PR comments — prompt-type command using gh CLI."""
    pr_ref = args.strip()

    prompt = f"""Fetch and summarize comments from a GitHub pull request.

{f'PR reference: {pr_ref}' if pr_ref else 'Find the current PR for this branch.'}

Steps:
1. Get PR info: `gh pr view {pr_ref or ''} --json number,headRefName,headRepository`
2. Fetch PR-level comments: `gh api /repos/{{owner}}/{{repo}}/issues/{{number}}/comments --jq '.[] | {{body, user: .user.login, created_at}}'`
3. Fetch review comments: `gh api /repos/{{owner}}/{{repo}}/pulls/{{number}}/comments --jq '.[] | {{body, path, line, diff_hunk, user: .user.login}}'`
4. Present all comments with file/line context
5. Do not add explanatory text — just show the comments"""

    return _prompt_result(input_text, prompt)


def _handle_resume(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Resume a previous conversation."""
    import json as _json

    session_dir = agent.runtime_config.session_directory
    if not session_dir.exists():
        return _local_result(input_text, 'No session directory found.')

    search_term = args.strip()

    # If a session ID is provided, show its info
    if search_term:
        session_path = session_dir / f'{search_term}.json'
        if session_path.exists():
            try:
                data = _json.loads(session_path.read_text(encoding='utf-8'))
                msg_count = len(data.get('messages', []))
                return _local_result(
                    input_text,
                    f'Found session `{search_term}` with {msg_count} messages.\n'
                    f'To resume, restart with: `--resume {search_term}`',
                )
            except Exception as exc:
                return _local_result(input_text, f'Error reading session: {exc}')

    # List recent sessions
    session_files = sorted(
        session_dir.glob('*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:20]

    if not session_files:
        return _local_result(input_text, 'No previous sessions found.')

    lines = ['## Recent Sessions', '']
    for sf in session_files:
        try:
            data = _json.loads(sf.read_text(encoding='utf-8'))
            sid = sf.stem
            msg_count = len(data.get('messages', []))
            model = data.get('model', 'unknown')
            first_user = ''
            for msg in data.get('messages', []):
                if msg.get('role') == 'user' and msg.get('content', '').strip():
                    first_user = msg['content'].strip()[:80]
                    if len(msg['content'].strip()) > 80:
                        first_user += '...'
                    break
            line = f'- `{sid}` ({msg_count} msgs, {model})'
            if first_user:
                line += f': {first_user}'
            lines.append(line)
        except Exception:
            lines.append(f'- `{sf.stem}` (error reading)')

    if search_term:
        # Filter by search term
        filtered = [
            l for l in lines
            if search_term.lower() in l.lower() or l.startswith('#')
        ]
        if len(filtered) <= 2:
            filtered.append(f'\nNo sessions matching "{search_term}".')
        lines = filtered

    lines.append('\nTo resume: restart with `--resume <session-id>`')
    return _local_result(input_text, '\n'.join(lines))


def _handle_add_dir(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Add a new working directory."""
    from pathlib import Path as _Path

    path_str = args.strip()
    if not path_str:
        return _local_result(input_text, 'Usage: /add-dir <path>')

    path = _Path(path_str).resolve()

    if not path.exists():
        return _local_result(input_text, f'Path not found: {path}')
    if not path.is_dir():
        return _local_result(input_text, f'Not a directory: {path}')

    cwd = agent.runtime_config.cwd
    if path == cwd or str(path).startswith(str(cwd) + '/'):
        return _local_result(
            input_text,
            f'Path is already within the working directory: {cwd}',
        )

    if not hasattr(agent, '_additional_directories'):
        agent._additional_directories = []
    if path in agent._additional_directories:
        return _local_result(input_text, f'Directory already added: {path}')

    agent._additional_directories.append(path)
    return _local_result(
        input_text,
        f'Added working directory: {path}\n'
        f'Tools can now access files in this directory.',
    )


def _handle_skills(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """List available skills."""
    lines = ['## Available Skills', '']
    for spec in get_slash_command_specs():
        primary = f'/{spec.names[0]}'
        lines.append(f'- `{primary}`: {spec.description}')
    lines.extend(['', 'Use the Skill tool to invoke skills programmatically.'])
    return _local_result(input_text, '\n'.join(lines))


def _handle_fast(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Toggle fast mode."""
    arg = args.strip().lower()
    current = getattr(agent, '_fast_mode', False)

    if arg == 'on':
        agent._fast_mode = True
        return _local_result(input_text, 'Fast mode enabled.')
    if arg == 'off':
        agent._fast_mode = False
        return _local_result(input_text, 'Fast mode disabled.')
    if not arg:
        agent._fast_mode = not current
        state = 'enabled' if agent._fast_mode else 'disabled'
        return _local_result(input_text, f'Fast mode {state}.')
    return _local_result(input_text, 'Usage: /fast [on|off]')


def _handle_vim(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """Toggle between Vim and Normal editing modes."""
    current = getattr(agent, '_editor_mode', 'normal')
    if current == 'emacs':
        current = 'normal'

    new_mode = 'vim' if current == 'normal' else 'normal'
    agent._editor_mode = new_mode

    if new_mode == 'vim':
        hint = 'Use Escape key to toggle between INSERT and NORMAL modes.'
    else:
        hint = 'Using standard (readline) keyboard bindings.'

    return _local_result(input_text, f'Switched to {new_mode} mode. {hint}')


def _handle_rewind(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Rewind conversation to a previous message."""
    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session.')

    n_str = args.strip()

    if not n_str:
        msgs = session.messages
        lines = ['## Conversation History', '']
        for i, msg in enumerate(msgs):
            role = msg.role.upper()
            preview = msg.content[:60].replace('\n', ' ')
            if len(msg.content) > 60:
                preview += '...'
            lines.append(f'  {i}: [{role}] {preview}')
        lines.append('')
        lines.append('Usage: /rewind <message-number> to truncate to that point.')
        return _local_result(input_text, '\n'.join(lines))

    try:
        target = int(n_str)
    except ValueError:
        return _local_result(input_text, 'Usage: /rewind <message-number>')

    if target < 0 or target >= len(session.messages):
        return _local_result(
            input_text,
            f'Invalid message number. Valid range: 0-{len(session.messages) - 1}',
        )

    removed_count = len(session.messages) - target - 1
    session.messages[:] = session.messages[:target + 1]
    return _local_result(
        input_text,
        f'Rewound conversation to message {target}. Removed {removed_count} messages.',
    )


def _prompt_result(input_text: str, prompt: str) -> SlashCommandResult:
    """Return a prompt-type result — the prompt gets sent to the model."""
    return SlashCommandResult(
        handled=True,
        should_query=True,
        prompt=prompt,
        transcript=({'role': 'user', 'content': input_text},),
    )


def _local_result(input_text: str, output: str) -> SlashCommandResult:
    transcript = (
        {'role': 'user', 'content': input_text},
        {'role': 'assistant', 'content': output},
    )
    return SlashCommandResult(
        handled=True,
        should_query=False,
        output=output,
        transcript=transcript,
    )
