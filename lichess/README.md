# Lichess bot

## Setup

Create a token with bot play at https://lichess.org/account/oauth/token, and paste the token in the file TOKEN.txt. Make sure to untrack the file so changes to it don't get pushed to Github.

```
git update-index --skip-worktree .\lichess\TOKEN.txt
```

## Run

Run the bot with the command (while being on the root directory).

```
python -m lichess.bot
```
