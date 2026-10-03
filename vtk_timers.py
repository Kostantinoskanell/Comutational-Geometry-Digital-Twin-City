"""Correct VTK interactor timers.

pyvista 0.47's `Plotter.add_timer_event` registers a TimerEvent observer that
is never removed and never checks WHICH timer fired: every registered callback
runs on every timer's tick, and a finished one-shot keeps calling
`DestroyTimer(stale_id)` on later events — VTK reuses ids, so it can kill a
newer, unrelated timer. `add_timer` below filters on `GetTimerEventId()` and
removes its observer when done.
"""
from __future__ import annotations

from typing import Callable


def add_timer(plotter, duration_ms: int, callback: Callable[[int], None],
              repeating: bool = True, max_steps: int | None = None,
              render: bool = True):
    """Start a timer that fires `callback(step)` only for its own ticks.

    One-shot when `repeating` is False. A repeating timer stops after
    `max_steps` ticks if given. `render=True` renders after each tick, matching
    pyvista's add_timer_event. Returns a zero-arg function that cancels it.
    """
    iren = plotter.iren.interactor
    state = {"step": 0, "id": None, "tag": None, "alive": True}

    def _stop(obj=None) -> None:
        if not state["alive"]:
            return
        state["alive"] = False
        target = obj if obj is not None else iren
        if state["id"] is not None:
            target.DestroyTimer(state["id"])
        if state["tag"] is not None:
            target.RemoveObserver(state["tag"])

    def _on_timer(obj, _event) -> None:
        if not state["alive"] or obj.GetTimerEventId() != state["id"]:
            return
        step = state["step"]
        state["step"] += 1
        done = (not repeating) or (max_steps is not None and state["step"] >= max_steps)
        if done:
            _stop(obj)
        callback(step)
        if render:
            obj.GetRenderWindow().Render()

    state["tag"] = iren.AddObserver("TimerEvent", _on_timer)
    state["id"] = (iren.CreateRepeatingTimer(int(duration_ms)) if repeating
                   else iren.CreateOneShotTimer(int(duration_ms)))
    return _stop
