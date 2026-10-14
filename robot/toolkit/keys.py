"""Recording key controls, shared by the recorder scripts. Arrow keys only.

Voice phrases (macOS Voice Control -> Commands -> Press Keyboard Shortcut). The scripts' spoken
prompts read PHRASES, so change a phrase here and in Voice Control together.
  "robot begin"   Right   start a demo / keep a recording and start the next demo
  "robot done"    Right   end a demo early
  "robot capture" Up      replay the demo while the cameras record (demo_record)
  "robot scrap"   Left    throw away the demo or recording and redo it
  "robot finish"  Down    end the session (Esc also works)

Letters are ignored on purpose. When macOS Voice Control doesn't match a phrase to a command it
types the words as text, and LeRobot's n / r / q letter shortcuts then fire: "capture take" typed
an "r" and threw the demo away. Typed text is logged as ignored so dictation shows up in the log.

Every key is logged as "[keys] heard <key> -> <control>", so a voice command that sends the wrong
key shows up in the log. A repeat of the same key within DEBOUNCE_S is ignored: Voice Control
sometimes fires one phrase twice, which would otherwise skip a step.
"""

import logging
import time

from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener

DEBOUNCE_S = 1.0
PHRASES = {"begin": "robot begin", "done": "robot done", "capture": "robot capture",
           "scrap": "robot scrap", "finish": "robot finish"}
CONTROLS = {"right": "begin/done", "left": "scrap", "up": "capture", "down": "finish", "esc": "finish"}
LEROBOT_KEY = {"begin/done": "right", "scrap": "left", "finish": "esc"}


def init_listener():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False, "record": False}
    last: dict[str, float] = {}

    def on_key(name: str) -> None:
        name = name.lower()
        control = CONTROLS.get(name)
        if control is None:
            if len(name) == 1:
                logging.info(f"[keys] typed text {name!r} ignored (Voice Control dictating instead of running a command?)")
            return
        now = time.monotonic()
        if now - last.get(control, -DEBOUNCE_S) < DEBOUNCE_S:
            logging.info(f"[keys] heard {name} -> {control} (repeat within {DEBOUNCE_S:.0f} s, ignored)")
            return
        last[control] = now
        logging.info(f"[keys] heard {name} -> {control}")
        if control == "capture":
            events["record"] = True
            events["exit_early"] = True
        else:
            apply_recording_control(LEROBOT_KEY[control], events)

    listener = create_key_listener(
        on_key, controls_help="Right = begin / done, Up = capture, Left = scrap, Down/Esc = finish (letters ignored)"
    )
    return listener, events
