# Copyright 2026 The RLinf Authors.

"""HITL viewer for ModelArts Jupyter.

``/files/`` is cookie-authenticated (token query 302s to login). A standalone
``console.html`` iframe is sandboxed without ``allow-same-origin``, so images
stay blank. This module injects ``<img src="{base}/files/...">`` into the
notebook page, which already has the Jupyter cookie.
"""

from __future__ import annotations

import base64
import html
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from IPython.display import HTML, display

_LIVE = Path(__file__).resolve().parent / "live"
_KEYS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
_WATCH_STOP: threading.Event | None = None
_WATCH_THREAD: threading.Thread | None = None


def live_dir() -> Path:
    return _LIVE


def jupyter_prefix() -> str:
    server = os.environ.get("JUPYTER_SERVER_URL", "").rstrip("/")
    prefix = urlparse(server).path.rstrip("/") if server else ""
    return prefix or "/f352f131-c066-41af-85cc-9fc46022853c"


def mailbox_rel() -> str:
    root = Path(os.environ.get("JUPYTER_SERVER_ROOT", "/home/ma-user/work")).resolve()
    return _LIVE.resolve().relative_to(root).as_posix()


def files_url(name: str) -> str:
    """Same-origin Jupyter FilesHandler URL (needs the browser cookie)."""
    return f"{jupyter_prefix()}/files/{mailbox_rel()}/{name}"


def contents_url(name: str) -> str:
    return f"{jupyter_prefix()}/api/contents/{mailbox_rel()}/{name}"


def _data_uri_cam(name: str, title: str) -> str:
    path = _LIVE / name
    if not path.is_file() or path.stat().st_size <= 0:
        return f'<div style="color:#c00">{html.escape(title)}: 没有 {html.escape(name)}</div>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<div><div style="color:#eee;font:12px sans-serif">{html.escape(title)}</div>'
        f'<img src="data:image/jpeg;base64,{b64}" '
        'style="width:100%;max-width:420px;background:#222"></div>'
    )


def _data_uri_row() -> str:
    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;flex:0 0 auto;'
        'background:#111;padding:0">'
        + _data_uri_cam("frame_main.jpg", "agentview (data-uri)")
        + _data_uri_cam("frame_wrist.jpg", "wrist (data-uri)")
        + "</div>"
    )


def _meta_grid_html(parsed: dict | None) -> str:
    if parsed is None:
        cells = [("seq", "-"), ("step", "-"), ("status", "waiting"), ("hold", "-")]
    else:
        elapsed = parsed.get("elapsed", "?")
        max_steps = parsed.get("max_episode_steps", "?")
        chunk = parsed.get("chunk") or {}
        action = parsed.get("action") or []
        action_s = " ".join(f"{float(x):.2f}" for x in action[:7]) if action else "-"
        cells = [
            ("seq", parsed.get("seq", "-")),
            ("step", f"{elapsed}/{max_steps}"),
            ("status", parsed.get("episode_status", "-")),
            ("hold", parsed.get("hold", "-")),
            ("intervened", parsed.get("intervened", "-")),
            ("task", parsed.get("task_id", "-")),
            (
                "chunk",
                f"#{chunk.get('id', '-')} {chunk.get('step', '?')}/{chunk.get('len', '?')}"
                if chunk
                else "-",
            ),
            ("human_steps", chunk.get("human_steps", "-") if chunk else "-"),
            ("exec", action_s),
        ]
    tiles = []
    for key, value in cells:
        tiles.append(
            "<div style='background:#1a2430;padding:6px 8px;border-radius:6px'>"
            f"<b style='display:block;color:#8aa;font:11px sans-serif'>{html.escape(str(key))}</b>"
            f"<span style='color:#eee;font:12px ui-monospace,monospace'>"
            f"{html.escape(str(value))}</span></div>"
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(90px,1fr));"
        "gap:6px'>"
        + "".join(tiles)
        + "</div>"
    )


def _briefing_lines(parsed: dict | None) -> tuple[str, str]:
    """Prompt + episode status text for the panel right of the seq cells."""
    if parsed is None:
        return "prompt: (waiting)\noutcome: (waiting)", "#9ef"
    prompt = str(parsed.get("prompt") or "").strip()
    task_id = parsed.get("task_id", "-")
    prompt_line = (
        f"prompt [task {task_id}]: {prompt}"
        if prompt
        else f"prompt [task {task_id}]: (empty)"
    )
    status = str(parsed.get("episode_status") or "unknown")
    intervened = bool(parsed.get("intervened"))
    counts = parsed.get("counts") or {}
    last = parsed.get("last_outcome")
    max_steps = parsed.get("max_episode_steps", "?")
    elapsed = parsed.get("elapsed", "?")
    mode_now = "HITL接管中" if intervened else "纯VLA"
    if status == "running":
        cur = f"进行中 · {mode_now} · step {elapsed}/{max_steps}"
        color = "#9ef"
    elif status == "success":
        cur = f"本集成功 · {mode_now}"
        color = "#6f6"
    elif status == "skipped":
        cur = "本集跳过 · 下一局即将开始"
        color = "#f66"
    else:
        cur = f"本集失败 · {mode_now}"
        color = "#f66"
    if isinstance(last, dict):
        last_mode = "HITL" if last.get("intervened") else "VLA"
        last_line = (
            f"上一集: {last_mode} → {last.get('result')} "
            f"(step {last.get('elapsed')}/{last.get('max_episode_steps')})"
        )
    else:
        last_line = "上一集: (尚无结束过的 episode)"
    tally = (
        f"累计 VLA成功={counts.get('vla_success', 0)} "
        f"VLA失败={counts.get('vla_fail', 0)} | "
        f"HITL成功={counts.get('hitl_success', 0)} "
        f"HITL失败={counts.get('hitl_fail', 0)}"
    )
    note = (
        "说明: intervened=false 的 episode 才是纯 VLA 成败；"
        "接管后无法反事实原轨迹。"
    )
    return "\n".join([prompt_line, cur, last_line, tally, note]), color


def _briefing_html(parsed: dict | None) -> str:
    text, color = _briefing_lines(parsed)
    return (
        f"<pre style='color:{color};background:#1a2430;padding:8px 10px;"
        f"white-space:pre-wrap;margin:0;max-height:220px;overflow:auto;"
        f"font:13px/1.4 sans-serif'>{html.escape(text)}</pre>"
    )


def _kernel_stage_html(parsed: dict | None) -> str:
    return (
        "<div style='display:grid;grid-template-columns:minmax(180px,1fr) "
        "minmax(180px,1fr) minmax(320px,1.3fr);grid-template-rows:auto auto;"
        "gap:10px;align-items:start;background:#111;padding:8px'>"
        f"<div>{_data_uri_cam('frame_main.jpg', 'agentview')}</div>"
        f"<div>{_data_uri_cam('frame_wrist.jpg', 'wrist')}</div>"
        f"<div>{_chunk_pre_html(parsed)}</div>"
        f"<div style='grid-column:1 / span 2'>{_meta_grid_html(parsed)}</div>"
        f"<div style='grid-column:3'>{_briefing_html(parsed)}</div>"
        "</div>"
    )


def _img_row(cache_bust: str | None = None) -> str:
    suffix = f"?t={cache_bust}" if cache_bust else f"?t={int(time.time() * 1000)}"
    main = files_url("frame_main.jpg") + suffix
    wrist = files_url("frame_wrist.jpg") + suffix
    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;background:#111;padding:8px">'
        f'<div><div style="color:#eee;font:12px sans-serif">agentview</div>'
        f'<img src="{html.escape(main)}" '
        'style="max-width:420px;min-width:240px;min-height:160px;background:#222"></div>'
        f'<div><div style="color:#eee;font:12px sans-serif">wrist</div>'
        f'<img src="{html.escape(wrist)}" '
        'style="max-width:420px;min-width:240px;min-height:160px;background:#222"></div>'
        "</div>"
    )


def _outcome_banner(parsed: dict) -> str:
    """Human-readable VLA / HITL success-fail summary from mailbox state."""
    return _briefing_html(parsed)


def _load_state() -> dict | None:
    state = _LIVE / "state.json"
    if not state.is_file():
        return None
    try:
        return json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _chunk_pre_html(parsed: dict | None) -> str:
    if parsed is None:
        body = "chunk: (waiting)"
    else:
        body = str(parsed.get("chunk_text") or "chunk: (waiting)")
        last = parsed.get("last_chunk_text")
        if last:
            body += "\n\nlast " + str(last)
    return (
        "<pre style='color:#b6e3b6;background:#151a15;padding:8px 10px;"
        "white-space:pre;overflow:auto;flex:1 1 0;min-width:280px;"
        "max-height:520px;margin:0;font:12px/1.35 ui-monospace,monospace'>"
        f"{html.escape(body)}</pre>"
    )


def _jpeg_stats_html() -> str:
    main = _LIVE / "frame_main.jpg"
    wrist = _LIVE / "frame_wrist.jpg"
    main_stat = (
        f"{main.stat().st_size}B @{main.stat().st_mtime:.3f}"
        if main.is_file()
        else "missing"
    )
    wrist_stat = (
        f"{wrist.stat().st_size}B @{wrist.stat().st_mtime:.3f}"
        if wrist.is_file()
        else "missing"
    )
    return (
        "<pre style='color:#666;background:#111;padding:4px 8px;font:11px monospace;"
        f"margin:0'>main={main_stat} wrist={wrist_stat}</pre>"
    )


def _state_status() -> str:
    parsed = _load_state()
    if parsed is None:
        return "<pre style='color:#c00;background:#111;padding:8px'>还没有 state.json</pre>"
    return _jpeg_stats_html()


def show_html_frames():
    """One-shot snapshot of disk JPEGs. Use ``watch_live()`` to see motion."""
    _LIVE.mkdir(parents=True, exist_ok=True)
    print("open_console=", Path(__file__).resolve())
    print("exports=", [name for name in globals() if name.startswith("show_")])
    print("/files/  ", files_url("frame_main.jpg"))
    print("contents", contents_url("frame_main.jpg"))
    state = _LIVE / "state.json"
    if state.is_file():
        print("state=", state.read_text(encoding="utf-8").strip())
    else:
        print("还没有 JPEG 时先跑：python toolkits/hitl/smoke_mailbox.py --seconds 120")
    print("这是快照，不会自己刷新。训练时请跑 watch_live()。")
    print("上排：data-uri（不走 /files/）。下排：Jupyter /files/ 链接。")
    display(HTML(_data_uri_row()))
    display(HTML(_img_row()))
    display(HTML(_state_status()))


def _ack_idle() -> str:
    """Write an idle cmd so ``block: True`` env can keep stepping.

    Skip when the HTML console currently holds, so WASD is not overwritten.
    """
    cmd_path = _LIVE / "cmd.json"
    if cmd_path.is_file():
        try:
            data = json.loads(cmd_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            data = {}
        if bool(data.get("hold", False)):
            return "hold-active"
        if bool(data.get("pause", False)):
            return "pause-active"
        controller = data.get("controller", "human")
    else:
        controller = "human"
    set_cmd(hold=False, controller=controller)
    return "acked-idle"


def stop_live() -> None:
    """Stop the background camera thread started by ``start_live``."""
    global _WATCH_STOP, _WATCH_THREAD
    if _WATCH_STOP is not None:
        _WATCH_STOP.set()
    thread = _WATCH_THREAD
    current = threading.current_thread()
    if thread is not None and thread.is_alive() and thread is not current:
        thread.join(timeout=2.0)
    _WATCH_THREAD = None
    _WATCH_STOP = None


def start_live(hz: float = 4.0) -> None:
    """Refresh camera frames in a daemon thread so the kernel stays free."""
    global _WATCH_STOP, _WATCH_THREAD
    stop_live()
    _LIVE.mkdir(parents=True, exist_ok=True)
    handle = display(
        HTML(
            "<p style='color:#eee;background:#111;padding:8px'>"
            "waiting for mailbox JPEG…</p>"
        ),
        display_id=True,
    )
    stop = threading.Event()
    _WATCH_STOP = stop
    period = 1.0 / max(float(hz), 0.5)

    def _loop() -> None:
        last_sig = None
        while not stop.is_set():
            try:
                ack = _ack_idle()
                main = _LIVE / "frame_main.jpg"
                sig = (
                    main.stat().st_mtime if main.is_file() else 0.0,
                    main.stat().st_size if main.is_file() else 0,
                )
                changed = sig != last_sig
                last_sig = sig
                flag = ("CHANGED" if changed else "same-file") + f" ack={ack}"
                parsed = _load_state()
                handle.update(
                    HTML(
                        _kernel_stage_html(parsed)
                        + _jpeg_stats_html()
                        + f"<pre style='color:#9f9;background:#111;padding:8px;"
                        f"margin:0'>{flag}</pre>"
                    )
                )
            except Exception:
                pass
            stop.wait(period)

    _WATCH_THREAD = threading.Thread(target=_loop, name="hitl-watch-live", daemon=True)
    _WATCH_THREAD.start()
    print(
        "这是调试用 kernel 刷图。日常 HITL 请只跑 show_hitl() / show_html_console()。"
        "停止：stop_live()"
    )


def watch_live(seconds: float = 3600.0, hz: float = 4.0, block: bool = False):
    """Start the live camera. Returns immediately unless ``block=True``."""
    del seconds
    start_live(hz=hz)
    if block:
        print("block=True：Interrupt 才会停。下一格会一直排队。")
        try:
            while _WATCH_THREAD is not None and _WATCH_THREAD.is_alive():
                time.sleep(0.25)
        except KeyboardInterrupt:
            stop_live()
            print("watch_live() interrupted")


def show_hitl(hz: float = 4.0):
    """Keyboard console with cameras in one iframe. Does not block the kernel.

    Stops the kernel data-URI camera cell if it is running — that extra
    output is unused and often hides the iframe images below it.
    """
    del hz
    stop_live()
    show_html_console()


def watch_html(seconds: float = 120.0, hz: float = 8.0, hold: bool = False, **axes):
    """Refresh the ``/files/`` ``<img>`` tags. Interrupt to stop."""
    _LIVE.mkdir(parents=True, exist_ok=True)
    handle = display(HTML(_img_row()), display_id=True)
    deadline = time.time() + float(seconds)
    period = 1.0 / max(float(hz), 1.0)
    print(
        "图是 HTML <img> 拉 Jupyter /files/。接管用另一 cell：set_cmd(hold=True, dx=0.4)"
    )
    while time.time() < deadline:
        seq = 0
        state = _LIVE / "state.json"
        if state.is_file():
            try:
                seq = int(json.loads(state.read_text(encoding="utf-8")).get("seq", 0))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                seq = 0
        action = {
            key: float(axes.get(key, -1.0 if key == "gripper" else 0.0))
            for key in _KEYS
        }
        set_cmd(hold=hold, seq=seq, **action)
        handle.update(HTML(_img_row(cache_bust=str(int(time.time() * 1000)))))
        time.sleep(period)
    print("watch_html() finished")


def show_html_console():
    """Keyboard HITL console with agentview/wrist, chunk table, and WASD.

    Injected as srcdoc (allow-same-origin) so JS keeps the Jupyter cookie.
    Images load the stable ``frame_main.jpg`` / ``frame_wrist.jpg`` as blobs
    (numbered per-seq copies are cleaned and often 404). Do not also run
    ``start_live()`` / ``watch_live()`` in another cell.
    """
    _LIVE.mkdir(parents=True, exist_ok=True)
    prefix = jupyter_prefix()
    rel = mailbox_rel()
    files_base = f"{prefix}/files/{rel}"
    api_base = f"{prefix}/api/contents/{rel}"
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body{{font-family:sans-serif;margin:8px;background:#111;color:#eee}}
  img{{width:100%;min-height:200px;background:#222;display:block;object-fit:contain}}
  .cam-main,.cam-wrist{{min-height:200px;background:#1a1a1a}}
  button{{font-size:16px;padding:8px 16px;border:0;border-radius:8px;cursor:pointer}}
  button.off{{background:#444;color:#ccc}} button.on{{background:#c0392b;color:#fff}}
  #btns{{display:flex;flex-direction:column;gap:6px}}
  #hold,#skip,#pause,#controller{{width:100%;font-size:16px;padding:10px 16px}}
  #skip{{background:#a85b12;color:#fff}}
  #skip:disabled{{opacity:0.45;cursor:default}}
  #pause.off{{background:#1a5f8a;color:#fff}}
  #pause.on{{background:#2a8bc7;color:#fff}}
  #controller.off{{background:#2a4a6b;color:#fff}}
  #controller.on{{background:#1a6b3c;color:#fff}}
  kbd{{background:#333;padding:1px 6px;border-radius:4px}}
  #prompt{{color:#9ef;background:#1a2430;padding:8px 10px;border-radius:6px;
    white-space:pre-wrap;margin:0;font:13px/1.4 sans-serif}}
  #outcome{{padding:8px 10px;border-radius:6px;white-space:pre-wrap;margin:0;
    font:13px/1.4 sans-serif;background:#1a2430;color:#9ef}}
  #outcome.ok{{color:#6f6}} #outcome.bad{{color:#f66}} #outcome.run{{color:#9ef}}
  .stage{{display:grid;grid-template-columns:minmax(160px,1fr) minmax(160px,1fr) minmax(340px,1.35fr);
    grid-template-rows:auto auto;gap:10px;align-items:start}}
  .cam-main{{grid-column:1;grid-row:1}}
  .cam-wrist{{grid-column:2;grid-row:1}}
  #chunk{{grid-column:3;grid-row:1;margin:0;min-height:200px;max-height:420px;
    color:#b6e3b6;background:#151a15;padding:8px 10px;border-radius:6px;
    white-space:pre;overflow:auto;font:12px/1.35 ui-monospace,monospace}}
  #meta{{grid-column:1 / span 2;grid-row:2;display:grid;
    grid-template-columns:repeat(3,minmax(80px,1fr));gap:6px}}
  #meta .cell{{background:#1a2430;padding:6px 8px;border-radius:6px}}
  #meta .cell b{{display:block;color:#8aa;font:11px sans-serif}}
  #meta .cell span{{color:#eee;font:12px ui-monospace,monospace;word-break:break-all}}
  #briefing{{grid-column:3;grid-row:2;display:flex;flex-direction:column;gap:6px}}
</style></head><body>
<h3>HITL HTML</h3>
<p>点进本框后：默认<strong>人类遥操作</strong>（xy / z / rpy，不是只有 WASD），点「切专家」后接管改由 ZAP 调整（再点接管结束）。
人类模式 <kbd>Space</kbd> 按住接管；专家模式 <kbd>Space</kbd> 或按钮开关接管。
<kbd>WASD</kbd> 水平 xy；<kbd>R/F</kbd> 上下 z；
<kbd>Q/E</kbd> 或 ←/→ 偏航 yaw；<kbd>T/G</kbd> 俯仰 pitch；<kbd>Y/H</kbd> 滚转 roll。
键盘按 SpaceMouse：每环境步加一点力（不是等推理时顶满），松手逐步回中。
满打平移 0.50 / 旋转 0.35；<kbd>Shift</kbd> 细调。按下的轴覆盖 π，未按的轴沿用 π。
yaml 可冻 actor 部分轴；本局 bowl 夹爪锁在 VLA，Z/X 无效。
无法成功时点「结束本局」或 <kbd>N</kbd>/<kbd>Esc</kbd>：记失败并立刻下一局。
临时走开点「暂停」或 <kbd>P</kbd>：env 停在当前步，回来再点继续。
专家接管从下一 chunk 生效（约 5 步延迟）。</p>
<div class="stage">
  <div class="cam-main">agentview<br><img id="main" alt="main"></div>
  <div class="cam-wrist">wrist<br><img id="wrist" alt="wrist"></div>
  <pre id="chunk">chunk: (waiting — 重新 show_html_console)</pre>
  <div id="meta"></div>
  <div id="briefing">
    <div id="prompt">prompt: (waiting…)</div>
    <div id="outcome">outcome: (waiting for episode end…)</div>
    <div id="btns">
      <button id="controller" class="off">当前：人类（点击切专家）</button>
      <button id="hold" class="off">VLA（点击接管）</button>
      <button id="pause" class="off">暂停（走开）</button>
      <button id="skip">结束本局 / 下一局</button>
    </div>
  </div>
</div>
<div id="sliders"></div>
<script>
const FILES = {json.dumps(files_base)};
const API = {json.dumps(api_base)};
const KEYS = ["dx","dy","dz","droll","dpitch","dyaw","gripper"];
const POSE_KEYS = ["dx","dy","dz","droll","dpitch","dyaw"];
const SM_TRANS = 0.50;
const SM_ROT = 0.35;
const SM_FINE = 0.36;
const SM_IDLE = 0.02;
const SM_STEP = 0.25;
const action = {{dx:0,dy:0,dz:0,droll:0,dpitch:0,dyaw:0,gripper:-1}};
const stick = {{dx:0,dy:0,dz:0,droll:0,dpitch:0,dyaw:0,gripper:-1}};
let hold = false, spaceDown = false, seq = 0, lastImgSeq = -1, lastImgOk = false, ticking = false, elapsed = 0;
let controller = "human";
let prompt = "", taskId = -1, gripperTouched = false;
let episodeStatus = "running", intervened = false, maxSteps = 0;
let lastOutcome = null, counts = {{}};
let analogSeq = -1;
let analogArmed = true;
let lastKeySig = "";
let shiftDown = false;
let skipHold = false;
let skipId = 0;
let paused = false;
const keysDown = new Set();
const keyHint = {{
  dx:"W/S 满0.50 / Shift细", dy:"A/D 满0.50 / Shift细", dz:"R/F 满0.50 / Shift细",
  droll:"Y/H 满0.35 / Shift细", dpitch:"T/G 满0.35 / Shift细",
  dyaw:"Q/E 或 ←/→ 满0.35 / Shift细",
  gripper:"Z/X 才覆盖策略夹爪"
}};
const keyMap = {{
  KeyW:["dx",1], KeyS:["dx",-1], KeyA:["dy",1], KeyD:["dy",-1],
  KeyR:["dz",1], KeyF:["dz",-1],
  KeyY:["droll",1], KeyH:["droll",-1],
  KeyT:["dpitch",1], KeyG:["dpitch",-1],
  KeyQ:["dyaw",1], KeyE:["dyaw",-1],
  ArrowLeft:["dyaw",1], ArrowRight:["dyaw",-1],
  KeyZ:["gripper",1.0], KeyX:["gripper",-1.0]
}};
function axisScale(k, fine) {{
  const coarse = (k === "droll" || k === "dpitch" || k === "dyaw") ? SM_ROT : SM_TRANS;
  return (fine ? SM_FINE : 1) * coarse;
}}
function poseLimit(k) {{
  return axisScale(k, false);
}}
function clampPose(k, value) {{
  const lim = poseLimit(k);
  return Math.max(-lim, Math.min(lim, value));
}}
function xsrf() {{
  const m = document.cookie.match(/(?:^|; )_xsrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : "";
}}
function isExpert() {{ return controller === "expert"; }}
function holdLabel() {{
  if (isExpert()) return hold ? "EXPERT（ZAP接管中）" : "VLA（点击专家接管）";
  return hold ? "HUMAN（接管中）" : "VLA（点击接管）";
}}
function setHold(v) {{
  hold = v;
  if (!hold) gripperTouched = false;
  const b = document.getElementById("hold");
  b.className = hold ? "on" : "off";
  b.textContent = holdLabel();
}}
function setController(v) {{
  const next = (v === "expert") ? "expert" : "human";
  if (next !== controller) {{
    controller = next;
    setHold(false);
    spaceDown = false;
    keysDown.clear();
  }}
  const b = document.getElementById("controller");
  if (b) {{
    b.className = isExpert() ? "on" : "off";
    b.textContent = isExpert()
      ? "当前：专家 ZAP（点击切人类）"
      : "当前：人类（点击切专家）";
  }}
  const hb = document.getElementById("hold");
  if (hb) hb.textContent = holdLabel();
}}
document.getElementById("controller").onclick = () =>
  setController(isExpert() ? "human" : "expert");
document.getElementById("hold").onclick = () => setHold(!hold);
function setPaused(v) {{
  paused = !!v;
  if (paused) {{
    keysDown.clear();
    spaceDown = false;
    setHold(false);
  }}
  const b = document.getElementById("pause");
  if (b) {{
    b.className = paused ? "on" : "off";
    b.textContent = paused ? "继续（已暂停）" : "暂停（走开）";
  }}
}}
document.getElementById("pause").onclick = () => setPaused(!paused);
function requestSkip() {{
  if (skipHold) return;
  if (episodeStatus !== "running") return;
  skipId += 1;
  skipHold = true;
  keysDown.clear();
  spaceDown = false;
  setHold(false);
  setPaused(false);
  const b = document.getElementById("skip");
  if (b) {{ b.disabled = true; b.textContent = "结束中…"; }}
}}
document.getElementById("skip").onclick = () => requestSkip();
KEYS.forEach((k) => {{
  const lab = document.createElement("label");
  lab.style.display = "block";
  lab.style.margin = "4px 0";
  const tip = " (" + (keyHint[k] || "") + ")";
  const pose = k !== "gripper";
  const min = pose ? -poseLimit(k) : -1;
  const max = pose ? poseLimit(k) : 1;
  const step = pose ? 0.01 : 0.02;
  lab.innerHTML = k + tip + ' <input type="range" min="' + min + '" max="' + max +
    '" step="' + step + '" value="' + action[k] + '" id="s_' + k +
    '"> <span id="v_' + k + '">' + action[k] + '</span>';
  document.getElementById("sliders").appendChild(lab);
  lab.querySelector("input").addEventListener("input", (e) => {{
    const raw = parseFloat(e.target.value);
    action[k] = pose ? clampPose(k, raw) : raw;
    document.getElementById("v_" + k).textContent = action[k].toFixed(2);
    analogArmed = true;
    if (k === "gripper") gripperTouched = true;
  }});
}});
window.addEventListener("keydown", (e) => {{
  if (e.code === "ShiftLeft" || e.code === "ShiftRight") {{ shiftDown = true; return; }}
  if (e.code === "Escape" || e.code === "KeyN") {{
    e.preventDefault();
    requestSkip();
    return;
  }}
  if (e.code === "KeyP") {{
    e.preventDefault();
    setPaused(!paused);
    return;
  }}
  if (paused) return;
  if (e.code === "Space") {{
    e.preventDefault();
    if (e.repeat) return;
    if (isExpert()) {{ setHold(!hold); return; }}
    spaceDown = true;
    setHold(true);
    return;
  }}
  if (keyMap[e.code]) {{
    e.preventDefault();
    keysDown.add(e.code);
    if (e.code === "KeyZ" || e.code === "KeyX") gripperTouched = true;
  }}
}});
window.addEventListener("keyup", (e) => {{
  if (e.code === "ShiftLeft" || e.code === "ShiftRight") {{ shiftDown = false; return; }}
  if (e.code === "Space") {{
    e.preventDefault();
    if (isExpert()) return;
    spaceDown = false;
    const live = currentStick();
    const stillMoving = KEYS.some((k) => k !== "gripper" && Math.abs(live[k]) > SM_IDLE);
    if (!stillMoving) setHold(false);
    return;
  }}
  if (keyMap[e.code]) e.preventDefault();
  keysDown.delete(e.code);
}});
function keySignature() {{
  return Array.from(keysDown).sort().join(",") + "|s" + (shiftDown ? "1" : "0");
}}
function currentStick() {{
  stick.gripper = action.gripper;
  return {{
    dx:stick.dx, dy:stick.dy, dz:stick.dz,
    droll:stick.droll, dpitch:stick.dpitch, dyaw:stick.dyaw,
    gripper:stick.gripper
  }};
}}
function stepStick() {{
  const signs = {{dx:0, dy:0, dz:0, droll:0, dpitch:0, dyaw:0}};
  keysDown.forEach((code) => {{
    const m = keyMap[code];
    if (!m) return;
    if (m[0] === "gripper") {{
      action.gripper = m[1];
      return;
    }}
    signs[m[0]] += m[1];
  }});
  POSE_KEYS.forEach((k) => {{
    let target = clampPose(k, action[k]);
    if (Math.abs(signs[k]) > 0) {{
      target = Math.sign(signs[k]) * axisScale(k, shiftDown);
    }}
    stick[k] += (target - stick[k]) * SM_STEP;
    if (Math.abs(stick[k]) < 0.005 && Math.abs(target) < SM_IDLE) stick[k] = 0;
    const vn = document.getElementById("v_" + k);
    if (vn) vn.textContent = stick[k].toFixed(2);
  }});
  stick.gripper = action.gripper;
}}
function maybeStepAnalog(seqNow) {{
  const sig = keySignature();
  if (seqNow !== analogSeq) {{
    analogSeq = seqNow;
    analogArmed = true;
  }}
  if (sig !== lastKeySig) {{
    lastKeySig = sig;
    analogArmed = true;
  }}
  if (!analogArmed) return;
  stepStick();
  analogArmed = false;
}}
async function getContents(name) {{
  const r = await fetch(API + "/" + name + "?type=file&content=1&t=" + Date.now(), {{
    credentials: "same-origin",
    cache: "no-store"
  }});
  if (!r.ok) throw new Error("GET " + name + " " + r.status);
  return r.json();
}}
async function pullState() {{
  try {{
    const data = await getContents("state.json");
    const raw = data.format === "base64" ? atob(data.content || "") : (data.content || "");
    const parsed = JSON.parse(raw);
    if (parsed.seq) seq = parsed.seq;
    elapsed = parsed.elapsed || 0;
    maxSteps = parsed.max_episode_steps || 0;
    prompt = parsed.prompt || "";
    taskId = (parsed.task_id === undefined || parsed.task_id === null)
      ? -1 : parsed.task_id;
    episodeStatus = parsed.episode_status || "running";
    intervened = !!parsed.intervened;
    if (parsed.skip_id_consumed !== undefined &&
        Number(parsed.skip_id_consumed) >= skipId) {{
      skipHold = false;
    }}
    const skipBtn = document.getElementById("skip");
    if (skipBtn && !skipHold) {{
      skipBtn.disabled = episodeStatus !== "running";
      skipBtn.textContent = "结束本局 / 下一局";
    }}
    lastOutcome = parsed.last_outcome || null;
    counts = parsed.counts || {{}};
    // Sync UI gripper to env latch until the user presses Z/X or moves the slider.
    if (!gripperTouched && parsed.gripper_latch !== undefined && parsed.gripper_latch !== null) {{
      action.gripper = parseFloat(parsed.gripper_latch);
      const sg = document.getElementById("s_gripper");
      const vg = document.getElementById("v_gripper");
      if (sg) sg.value = action.gripper;
      if (vg) vg.textContent = action.gripper.toFixed(2);
    }}
    const promptEl = document.getElementById("prompt");
    if (promptEl) {{
      promptEl.textContent = prompt
        ? ("prompt [task " + taskId + "]: " + prompt)
        : "prompt: (empty / not published yet — restart training after this change)";
    }}
    const outEl = document.getElementById("outcome");
    if (outEl) {{
      const modeNow = intervened ? "HITL接管中" : "纯VLA";
      let cur, cls;
      if (episodeStatus === "success") {{ cur = "本集成功 · " + modeNow; cls = "ok"; }}
      else if (episodeStatus === "skipped") {{ cur = "本集跳过 · 下一局即将开始"; cls = "bad"; }}
      else if (episodeStatus === "fail") {{ cur = "本集失败 · " + modeNow; cls = "bad"; }}
      else if (paused || parsed.paused) {{
        cur = "已暂停 · env 停在当前步 · " + modeNow + " · step " + elapsed + "/" + maxSteps;
        cls = "run";
      }}
      else {{
        cur = "进行中 · " + modeNow + " · step " + elapsed + "/" + maxSteps;
        cls = "run";
      }}
      let lastLine = "上一集: (尚无结束过的 episode)";
      if (lastOutcome) {{
        const lm = lastOutcome.intervened ? "HITL" : "VLA";
        lastLine = "上一集: " + lm + " → " + lastOutcome.result +
          " (step " + lastOutcome.elapsed + "/" + lastOutcome.max_episode_steps + ")";
      }}
      const tally = "累计 VLA成功=" + (counts.vla_success||0) +
        " VLA失败=" + (counts.vla_fail||0) +
        " | HITL成功=" + (counts.hitl_success||0) +
        " HITL失败=" + (counts.hitl_fail||0);
      outEl.className = cls;
      outEl.textContent = cur + "\\n" + lastLine + "\\n" + tally +
        "\\n说明: intervened=false 的 episode 才是纯 VLA 成败；接管后无法反事实原轨迹。";
    }}
    const chunkEl = document.getElementById("chunk");
    if (chunkEl) {{
      chunkEl.textContent = parsed.chunk_text ||
        "chunk: (waiting — 重启训练后才会有 π=ref / a=actor / exec)";
    }}
    return parsed;
  }} catch (e) {{
    return {{}};
  }}
}}
async function setImg(id, name) {{
  const img = document.getElementById(id);
  if (!img) return "missing";
  const stable = (id === "wrist") ? "frame_wrist.jpg" : "frame_main.jpg";
  try {{
    const r = await fetch(FILES + "/" + stable + "?t=" + Date.now() + "&seq=" + seq, {{
      credentials: "include",
      cache: "no-store"
    }});
    if (r.ok) {{
      const blob = await r.blob();
      if (blob.size > 64) {{
        const url = URL.createObjectURL(blob);
        const old = img.dataset.bloburl;
        img.src = url;
        img.dataset.bloburl = url;
        if (old && old !== url) URL.revokeObjectURL(old);
        return "blob";
      }}
    }}
  }} catch (e) {{}}
  try {{
    const data = await getContents(stable);
    if (data.content) {{
      const b64 = data.format === "base64" ? data.content : btoa(data.content);
      img.src = "data:image/jpeg;base64," + b64.replace(/\\s/g, "");
      return "contents";
    }}
  }} catch (e) {{}}
  if (name && name !== stable) {{
    try {{
      const data = await getContents(name);
      if (data.content) {{
        const b64 = data.format === "base64" ? data.content : btoa(data.content);
        img.src = "data:image/jpeg;base64," + b64.replace(/\\s/g, "");
        return "numbered";
      }}
    }} catch (e) {{}}
  }}
  return "fail";
}}
async function putCmd(cur) {{
  const token = xsrf();
  const r = await fetch(API + "/cmd.json", {{
    method: "PUT",
    credentials: "same-origin",
    headers: {{
      "Content-Type": "application/json",
      ...(token ? {{"X-XSRFToken": token}} : {{}})
    }},
    body: JSON.stringify({{
      type: "file",
      format: "text",
      content: JSON.stringify({{
        seq: seq,
        hold: hold,
        inherit_gripper: true,
        skip: skipHold,
        skip_id: skipId,
        pause: paused,
        controller: controller,
        action: KEYS.map((k) => cur[k])
      }})
    }})
  }});
  if (!r.ok) throw new Error("PUT cmd " + r.status);
}}
async function tick() {{
  if (ticking) return;
  ticking = true;
  let mode = "hold-img";
  try {{
    const parsed = await pullState();
    maybeStepAnalog(seq);
    const cur = currentStick();
    const moving = KEYS.some((k) => k !== "gripper" && Math.abs(cur[k]) > SM_IDLE);
    // WASD auto-hold must drop when the stick is idle, otherwise zeros
    // overwrite the VLA as fake a^H. Stick only advances on env seq / key change.
    if (!paused && !isExpert()) {{
      if (moving) setHold(true);
      else if (!spaceDown) setHold(false);
    }}
    if (seq !== lastImgSeq || !lastImgOk) {{
      const main = parsed.main || "frame_main.jpg";
      const wrist = parsed.wrist || "frame_wrist.jpg";
      const a = await setImg("main", main);
      const b = await setImg("wrist", wrist);
      mode = a + "/" + b;
      lastImgSeq = seq;
      lastImgOk = a !== "fail" && a !== "missing";
    }}
    await putCmd(cur);
    const grip = gripperTouched ? "set" : "inherit";
    const ch = parsed.chunk || {{}};
    const exec = (parsed.action || []).map((x) => Number(x).toFixed(2)).join(" ");
    const cells = [
      ["seq", seq],
      ["step", elapsed + "/" + maxSteps],
      ["status", episodeStatus],
      ["controller", controller],
      ["hold", hold],
      ["pause", paused ? "on" : "off"],
      ["skip", skipHold ? ("id " + skipId) : "off"],
      ["intervened", intervened],
      ["task", taskId],
      ["chunk", ch.id != null ? ("#" + ch.id + " " + (ch.step||"?") + "/" + (ch.len||"?")) : "-"],
      ["human_steps", ch.human_steps != null ? ch.human_steps : "-"],
      ["cmd", KEYS.map((k) => cur[k].toFixed(2)).join(" ")],
      ["exec", exec || "-"],
      ["gripper", grip + " " + action.gripper.toFixed(2)],
      ["via", mode]
    ];
    const meta = document.getElementById("meta");
    if (meta) {{
      meta.innerHTML = cells.map((kv) =>
        "<div class=\\"cell\\"><b>" + kv[0] + "</b><span>" + kv[1] + "</span></div>"
      ).join("");
    }}
  }} catch (e) {{
    const meta = document.getElementById("meta");
    if (meta) meta.innerHTML = "<div class=\\"cell\\"><b>error</b><span>" + String(e) + " via=" + mode + "</span></div>";
  }} finally {{
    ticking = false;
  }}
}}
setInterval(tick, 50);
tick();
</script>
</body></html>"""
    escaped = html.escape(page, quote=True)
    display(
        HTML(
            "<p style='font:13px sans-serif'>点进这个框后键盘才生效："
            "默认人类，点「切专家」后接管=ZAP。"
            "<kbd>Space</kbd> 接管，<kbd>WASD</kbd> xy，<kbd>R/F</kbd> z，"
            "<kbd>Q/E</kbd>/←→ yaw，<kbd>T/G</kbd> pitch，<kbd>Y/H</kbd> roll，"
            "<kbd>Z/X</kbd> 夹爪，<kbd>P</kbd> 暂停，<kbd>N</kbd>/<kbd>Esc</kbd> 结束本局。"
            "画面就在本框，不要另开 watch_live。</p>"
            f'<iframe srcdoc="{escaped}" '
            'sandbox="allow-scripts allow-same-origin allow-forms" '
            'style="width:100%;height:920px;border:1px solid #444;background:#111">'
            "</iframe>"
        )
    )
    print("files:", files_url("frame_main.jpg"))
    print("contents:", contents_url("frame_main.jpg"))


def show_last_frames():
    """Show mailbox JPEGs as HTML data-URI (ModelArts often skips image/jpeg)."""
    _LIVE.mkdir(parents=True, exist_ok=True)
    print(f"mailbox={_LIVE}")
    state = _LIVE / "state.json"
    if state.is_file():
        print("state=", state.read_text(encoding="utf-8").strip())
    if not (_LIVE / "frame_main.jpg").is_file():
        print("终端：python toolkits/hitl/smoke_mailbox.py --seconds 120")
        return
    display(HTML(_data_uri_row()))


def set_cmd(
    *,
    hold: bool = False,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    droll: float = 0.0,
    dpitch: float = 0.0,
    dyaw: float = 0.0,
    gripper: float = -1.0,
    seq: int | None = None,
    inherit_gripper: bool | None = None,
    skip: bool = False,
    skip_id: int = 0,
    pause: bool = False,
    controller: str = "human",
) -> dict:
    _LIVE.mkdir(parents=True, exist_ok=True)
    if seq is None:
        seq = 0
        state = _LIVE / "state.json"
        if state.is_file():
            try:
                seq = int(json.loads(state.read_text(encoding="utf-8")).get("seq", 0))
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                seq = 0
    if inherit_gripper is None:
        # Default: do not force-open when gripper stays at UI default -1.
        # Pass inherit_gripper=False to deliberately command gripper=-1.
        inherit_gripper = True
    controller_name = str(controller).strip().lower()
    if controller_name not in {"expert", "zap", "teacher"}:
        controller_name = "human"
    else:
        controller_name = "expert"
    payload = {
        "seq": int(seq),
        "hold": bool(hold),
        "inherit_gripper": bool(inherit_gripper),
        "skip": bool(skip),
        "skip_id": int(skip_id),
        "pause": bool(pause),
        "controller": controller_name,
        "action": [
            float(dx),
            float(dy),
            float(dz),
            float(droll),
            float(dpitch),
            float(dyaw),
            float(gripper),
        ],
    }
    (_LIVE / "cmd.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def watch(seconds: float = 90.0, hz: float = 8.0, hold: bool = False, **axes):
    watch_html(seconds=seconds, hz=hz, hold=hold, **axes)


def show_console(seconds: float = 0.0, hz: float = 8.0):
    show_html_console()
    if seconds and seconds > 0:
        watch_html(seconds=seconds, hz=hz)


def console_src() -> str:
    return files_url("console.html")
