# Voice control for collecting episodes

The recorder reacts to **arrow keys only**. macOS Voice Control, built in and on-device, presses
an arrow when you say a phrase. Letters are ignored on purpose: when Voice Control doesn't
recognize a phrase as a command it types the words, and typed letters used to trigger controls.

## The five commands

| Say | Key it presses | What it does |
|---|---|---|
| **robot begin** | → Right arrow | Start a demo · at review: keep the recording and start the next demo |
| **robot done** | → Right arrow | End the demo (otherwise it ends after 20 s) |
| **robot capture** | ↑ Up arrow | Replay the demo while the cameras record |
| **robot scrap** | ← Left arrow | Throw away the demo or recording and redo it |
| **robot finish** | ↓ Down arrow | Keep what's saved, upload, and quit |

## Setup (once)

1. **Delete the old commands.** In System Settings → Accessibility → Voice Control → **Commands…**,
   under **Custom**, select each old one (start take, next take, stop take, record take,
   capture take, redo take, scrap take, stop session, end session) and click **−**.
2. **Add the five commands above.** For each one click **+**, then:
   - **When I say:** the phrase, exactly as written (lowercase is fine)
   - **When using:** Any App
   - **Perform:** **Press Keyboard Shortcut**, then click **Configure…** and press the arrow key
   - Make sure **Active** is on
   Don't use "Type Text": typed letters are ignored.
3. **Test in TextEdit.** Type a few lines, say **"command mode"**, then say each phrase. The cursor
   should move: robot begin / robot done → right, robot capture → up, robot scrap → left,
   robot finish → down. If the words get typed instead, the command didn't match: check the
   spelling and that it's Active.
4. **Let the recorder hear keys from any window:** System Settings → Privacy & Security →
   **Accessibility** and **Input Monitoring** → turn on your terminal app, then quit and reopen it.
   Without this, keep the terminal window focused while recording.
5. **Quit Wispr Flow** while recording.

## Every session

Say **"command mode"** first. In dictation mode Voice Control types what you say.

## How an episode goes (`./collect.sh replay`)

1. **"Demo 3 of 15. Say robot begin when ready."** The arm follows the leader, nothing recorded.
   Put the bottle on its taped spot.
2. **"robot begin"**, do the demo, **"robot done"**.
   "robot scrap" during the demo → throw it away, back to step 1.
3. The arm glides to where the demo started: **"Put the bottle back, then say robot capture."**
   "robot scrap" here → throw the demo away, back to step 1.
4. **"robot capture"**: the arm replays your demo while the cameras record.
   "robot scrap" during the replay → throw that recording away, back to step 3.
5. **Review. Nothing is saved yet.** The arm follows the leader again so you can reset.
   - **"robot begin"**: keep it and start the next demo right away
   - **"robot scrap"**: throw it away, back to step 3 (scrap again there to redo the demo)
   - **"robot finish"**: keep it and quit

`./collect.sh live` has no replay: "robot begin" records your teleop directly, "robot done" saves it.

## If a command misfires

```bash
grep "\[keys\]" $(ls -t ~/so101/logs/collect-*.log | head -1)
```
Each phrase should show up once as `heard <arrow> -> <control>`. `typed text ... ignored` lines
mean Voice Control dictated the phrase instead of running the command.
