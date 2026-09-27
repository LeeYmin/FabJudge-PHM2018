# FabJudge — PHM2018 starter

## Windows 11 setup

### 1) Install VS Code + Python + extensions
Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_tools_windows.ps1
```

If VS Code was newly installed and the `code` command was not immediately available,
close and reopen PowerShell once.

### 2) Create the project virtual environment
From the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

This creates `.venv`, installs the packages in `requirements.txt`, and registers:

`Python (FabJudge-PHM2018)`

### 3) Open the project in VS Code

```powershell
code .
```

### 4) Select Python interpreter
`Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose `.venv`.

### 5) Select notebook kernel
Open `notebooks/00_environment_check.ipynb`.
Use the kernel picker at the top-right -> `Python (FabJudge-PHM2018)`.

Run all cells.

### 6) Sign in to Codex
Open the Codex icon in the VS Code sidebar.
If it is hidden:

`Ctrl+Shift+P` -> `Codex: Open Codex Sidebar`

Then sign in with your ChatGPT account.

### 7) Add PHM2018
Place the dataset under:

`data/raw/`

Do not rename or reshape source files yet. Run:

`notebooks/01_data_structure.ipynb`

first, then decide the analysis unit and preprocessing.

## Project structure

```text
FabJudge-PHM2018/
├─ .vscode/
│  └─ settings.json
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
│  ├─ 00_environment_check.ipynb
│  └─ 01_data_structure.ipynb
├─ src/
│  └─ fabjudge/
│     ├─ __init__.py
│     └─ paths.py
├─ tests/
├─ AGENTS.md
├─ requirements.txt
├─ install_tools_windows.ps1
├─ setup_windows.ps1
└─ .gitignore
```
