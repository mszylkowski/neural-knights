# Lichess bot

## Setup

Create a token with bot play at https://lichess.org/account/oauth/token, and paste the token in the file `lichess/TOKEN.txt` (in this directory). Do not upload this file to Github.

```
git update-index --skip-worktree .\lichess\TOKEN.txt
```

## Run

Run the bot with the command (while being on the root directory).

```
python -m lichess.bot
```
