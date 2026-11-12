# Publishing RoadForge to GitHub

The local Git repository is initialized on `main` and contains the initial commit. To create the public GitHub repository and push this history from a Windows development machine:

1. Install [GitHub CLI](https://cli.github.com/).
2. Open Command Prompt in this project folder and authenticate:

   ```bat
   gh auth login
   ```

3. Run:

   ```bat
   publish_to_github.bat
   ```

The script creates `ArvinNotDev/RoadForge` as a public repository, sets the project description, adds the `origin` remote, and pushes the existing `main` branch.

The command is safe to run only when that repository does not already exist. If you already created it manually, run:

```bat
git remote add origin https://github.com/ArvinNotDev/RoadForge.git
git push -u origin main
```

Do not paste a GitHub token into source files, issues, or chat. Authenticate through the GitHub CLI's normal browser/device flow.
