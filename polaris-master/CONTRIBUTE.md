# Polaris Contributing Guide

### ML for exploring and analyzing telemetry data from the SatNOGS network and beyond :rocket:

Thank you for considering and taking the time to contribute! We're very glad that you want to improve Polaris :)

We also suggest that it would be best to try Polaris before contributing to it, here's our [getting started guide](https://docs.polarisml.space/en/latest/using/getting_started_with_polaris.html).

## Prerequisites
---
You’ll need to have a few things before we can start:

- You need Linux or Unix-like environment or Windows(7+) with Python 3.6 - 3.9, but support is limited for Windows.

- You’ll need to be comfortable installing Python packages – we strongly recommend using `venv` to create a separate environment for Polaris. If you haven’t done this before, [this guide should help you out.](https://docs.python.org/3.8/library/venv.html)
- You’ll need to be comfortable with the CLI.
- You’ll need a good network connection, so you can download telemetry quickly.
- [Git](https://git-scm.com/) installed on OS.
- Attitude to never give up.

## Setting up Local Development Enviornment
---
1. ### **Forking repository**:
   - Firstly you have to make your own copy of project. For that you have to fork the repository.
   - Kindly wait till it gets forked.
   - After that copy will look like `<your-user-name>/polaris` forked from ` librespacefoundation/Polaris/polaris`.
   - For more information click [here](https://docs.gitlab.com/ee/user/project/working_with_projects.html).

2. ### **Clone repository**:
   - Now you have your own copy of project. Here you have to start your work.
   - Go to desired location on your computer where you want to set-up the project.
   - Open `terminal` on your OS. you can learn more [here](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html).
   - Type the command git clone <your-fork-url>.git and hit enter.
   - Wait for few seconds till the project gets copied.
   ```bash
      $ git clone https://gitlab.com/librespacefoundation/polaris/polaris.git --recurse-submodules
      Cloning into 'polaris'...
      remote: Counting objects: 179, done.
      remote: Total 179 (delta 0), reused 0 (delta 0), pack-reused 179
      Receiving objects: 100% (179/179), 24.07 KiB | 0 bytes/s, done.
      Resolving deltas: 100% (79/79), done.
      Checking connectivity... done
   ```

3. ### **Adding LSF's repo as alternate remote**:
   - Now you have to set up remote repositories.
   - Type `git remote -v` in terminal to list remote connections to your repo.
   - It will show something like this:
   ```bash
    origin  https://gitlab.com/<your-user-name>/polaris.git (fetch)
    origin  https://gitlab.com/<your-user-name>/polaris.git (push)
    ```

   - Now type the command `git remote add upstream https://gitlab.com/librespacefoundation/polaris/polaris.git` this will set upstream as main directory.
   - Again type in command `git remote -v` to check if remote has been set up correctly.
   - It should show something like this:

   ```bash
    origin  https://gitlab.com/Jeevan-Kiran-Lenka/polaris.git (fetch)
    origin  https://gitlab.com/Jeevan-Kiran-Lenka/polaris.git (push)
    upstream        https://gitlab.com/librespacefoundation/polaris/polaris.git (fetch)
    upstream        https://gitlab.com/librespacefoundation/polaris/polaris.git (push)
    ```
    - You can learn more about `remote` [here](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)

4. ### **More info for developers**

   Building the package from the sources:

   ```bash
   # Clone the repo
   $ git clone https://gitlab.com/librespacefoundation/polaris/polaris.git --recurse-submodules

   # Activate the virtual environment:
   $ source .venv/bin/activate

   # Build and install the package in editable mode; any changes
   # to your code will be reflected when you run polaris.
   $ (.venv) pip install -e .
   ```

   **Note:** If you run into problems installing Polaris via pip, [try
   using the new Pip resolver](https://pip.pypa.io/en/stable/news/#id18):

   ``` BASH
   pip install -e . --use-feature=2020-resolver
   ```

   It is important to format the code before commiting, otherwise the
   CI engine will fail. We have a tox command setup to run tests before
   committing so you will never have to push failing pipelines. Code
   linting is also done to ensure the code does not have any errors
   before committing.

   First you will have to install Prettier. Be sure to have a node version equal or greater than version 10.13.0. In case you don't have a good node version here is how to install/update it:

   ```bash
   $ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash

   # Feel free to install any version you like, but >= 10.13.0
   $ nvm install v13.8.0
   $ nvm use v13.8.0
   ```

   After the installation of node, you have to restart your terminal.
   Then, to install Prettier:

   ```bash
   npm install -g prettier
   ```

   You can learn more about npm [here](https://www.npmjs.com/).

   ```bash
   # Install tox to execute CI tasks
   $ (.venv) pip install tox

   # Auto-format the code
   $ (.venv) tox -e yapf-apply -e isort-apply -e prettier-apply
   ______________________ summary______________________
   yapf-apply: commands succeeded
   isort: commands succeeded
   prettier-apply: commands succeeded
   congratulations :)

   # Verify CI test passes
   $ (.venv) tox
   # If all goes well, you will get something like this:
   ______________________ summary______________________
   flake8: commands succeeded
   isort: commands succeeded
   yapf: commands succeeded
   pylint: commands succeeded
   build: commands succeeded
   pytest: commands succeeded
   prettier: commands succeeded
   congratulations :)

   ```

   You can learn more about tox [here](https://tox.readthedocs.io/en/latest/).

   If you want to develop polaris reports, then you have to install the dependencies with yarn. To do that run the following:
   ```
   cd polaris/reports/application
   yarn build
   ```
   To know more about installation of yarn, you can refer to [yarn installation page](https://classic.yarnpkg.com/en/docs/install).

   For more info on setup and developing `polaris reports`, you can checkout [https://gitlab.com/librespacefoundation/polaris/polaris-reports/-/blob/main/README.md](https://gitlab.com/librespacefoundation/polaris/polaris-reports/-/blob/main/README.md)


5. ### **Creating a branch**:

    - While submitting a MR to polaris, it is highly recommended that changes are made to a separate branch on your fork rather than on the master branch.

    - Click [here](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) to learn more about branching

 ## How to Report Bugs
 ---

 Please open a [new issue in the appropriate GitLab repository](https://gitlab.com/librespacefoundation/polaris/polaris/-/issues) with steps to reproduce the problem you're experiencing.

 Any additional detail you can submit, such as *screenshots*, *text output*, and *both your expected and actual results* will help us to understand your request.

 ## How to Request Enhancements
 ---

 First, please refer to the applicable [GitLab repository](https://gitlab.com/librespacefoundation/polaris/polaris) and search the [repository's GitLab issues](https://gitlab.com/librespacefoundation/polaris/polaris/-/issues) to make sure your idea has not been (or is not still) considered.

 Then, please [create a new issue in the Gitlab repository](https://gitlab.com/librespacefoundation/polaris/polaris/-/issues/new?issue%5Bassignee_id%5D=&issue%5Bmilestone_id%5D=) describing your enhancement.

 Any additional detail you can submit, such as *step-by-step descriptions*, *specific examples*, *screenshots or mockups*, and *reasoning* will help us to understand your request.
