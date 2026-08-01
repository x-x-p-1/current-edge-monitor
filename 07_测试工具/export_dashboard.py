"""
导出 M0 仿真数据 → 自包含实时看板 HTML（运行：python 00_数据生成与仿真/export_dashboard.py）
输出：sim/dashboard.html（双击即可在浏览器打开，数据内嵌，无依赖）
看板：三相电流波形滚动 / 过程状态时间线 / RMS 包络 / 事件 / 慢路径边带，可播放/调速/拖动
"""
import os
import sys
import json
import importlib

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "00_数据生成与仿真"))

_sim = importlib.import_module("current_simulator")
generate_dataset, Fault = _sim.generate_dataset, _sim.Fault
_pp = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor, PreprocessConfig = _pp.CurrentPreprocessor, _pp.PreprocessConfig
_feat = importlib.import_module("02_特征提取")
_ps = importlib.import_module("03_检测模型.process_state")
ProcessStateClassifier, StateRuleConfig = _ps.ProcessStateClassifier, _ps.StateRuleConfig
_hyst = importlib.import_module("04_后处理与决策.hysteresis")
HysteresisAlarm, EventAggregator, AlarmState = (_hyst.HysteresisAlarm,
                                                _hyst.EventAggregator, _hyst.AlarmState)

FS = 16000.0
WIN, STRIDE = 256, 128
BASE_LOAD_RMS = 0.71


def main():
    sig, _, meta = generate_dataset(
        duration=30.0, f1=50.0,
        load_profile=[(0.0, 5.0, 0.15), (5.0, 20.0, 1.0),
                      (20.0, 22.0, 1.0), (22.0, 28.0, 1.0), (28.0, 30.0, 0.005)],
        faults=[
            Fault(kind="rotor_sideband", start=10.0, dur=4.0, slip=0.03, depth=0.03),
            Fault(kind="stall", start=20.0, dur=2.0, depth=1.8),
        ],
    )

    pp = CurrentPreprocessor(
        PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
    clf = ProcessStateClassifier(StateRuleConfig(initial_baseline=BASE_LOAD_RMS,
                                                 stall_confirm=3))
    alarm = HysteresisAlarm(threshold_upper=0.6, threshold_lower=0.4,
                            confirm_count=3, release_count=3)
    agg = EventAggregator(merge_window_ms=500.0, frame_interval_ms=1000 * STRIDE / FS)

    frames = []
    for i in range(0, len(sig) - WIN, STRIDE):
        win = pp.process(sig[i:i + WIN])
        fast = _feat.extract_fast_features(win, FS)
        res = clf.update(fast)
        frames.append({
            "t": round(i / FS, 3),
            "rms": [round(fast.get(f"ch{c}_rms", 0.0), 5) for c in range(3)],
            "state": res["state"],
        })
        score = 0.9 if res["state"] in ("STALL", "TRANSIENT") else 0.1
        st = alarm.update(score)
        agg.update(st == AlarmState.ALARM)

    events = [
        {"start_ms": e.get("start_time_ms", 0.0),
         "end_ms": e.get("end_time_ms", e.get("start_time_ms", 0.0) + 1000)}
        for e in agg.finalize()
    ]

    slow = []
    win_len = int(2.0 * FS)
    for ws in range(0, len(sig) - win_len + 1, win_len):
        seg = sig[ws:ws + win_len]
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        s = _feat.extract_slow_features(seg, FS, f1=50.0, slip=0.03)
        slow.append({
            "t": round(ws / FS, 1),
            "sideband": s.get("ch0_sideband_ratio", 0.0),
            "thd": s.get("ch0_thd", 0.0),
            "unbalance": s.get("3p_unbalance_pct", 0.0),
        })

    # 降采样波形（显示用，500 Hz）
    step = int(FS / 500)
    disp = sig[::step]
    disp_t = np.arange(len(disp)) * step / FS

    data = {
        "meta": {"fs": FS, "duration": 30.0, "f1": 50.0, "stride_s": STRIDE / FS},
        "wave": {"t": [round(v, 4) for v in disp_t],
                 "a": [round(v, 4) for v in disp[:, 0]],
                 "b": [round(v, 4) for v in disp[:, 1]],
                 "c": [round(v, 4) for v in disp[:, 2]]},
        "frames": frames,
        "events": events,
        "slow": slow,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs("sim", exist_ok=True)
    out = os.path.join("sim", "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"看板已生成: {os.path.abspath(out)}")
    print(f"帧数={len(frames)} 事件={len(events)} 慢路径={len(slow)} 波形点={len(disp_t)}")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>M0 仿真实时看板 — 三相电流监测</title>
<style>
  body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
  .wrap{max-width:1120px;margin:0 auto;padding:16px}
  h1{font-size:18px;margin:0 0 2px}
  .sub{color:#9aa0a6;font-size:12px;margin-bottom:12px}
  canvas{background:#15181d;border:1px solid #2a2e35;border-radius:6px;width:100%;display:block;margin-top:8px}
  .row{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
  .panel{flex:1;min-width:300px}
  .label-big{font-size:30px;font-weight:700;text-align:center;padding:12px;border-radius:6px;border:1px solid #2a2e35;font-variant-numeric:tabular-nums}
  .controls{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
  button{background:#2a2e35;color:#e8eaed;border:1px solid #3c4043;border-radius:5px;padding:6px 14px;cursor:pointer}
  button:hover{background:#3c4043}
  select{background:#2a2e35;color:#e8eaed;border:1px solid #3c4043;border-radius:5px;padding:5px}
  input[type=range]{flex:1;min-width:200px}
  .legend span{display:inline-block;margin-right:14px;font-size:12px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
  #eventLog{font-size:12px;color:#f28b82;margin-top:6px}
  .tick{color:#5f6368;font-size:11px}
</style>
</head>
<body>
<div class="wrap">
  <h1>M0 端到端仿真 · 实时看板</h1>
  <div class="sub">三相变频电机电流监测 · 30s 场景 · 数据发生器 → 预处理 → 快/慢特征 → 过程状态 → 事件</div>

  <canvas id="cvWave" height="180"></canvas>
  <canvas id="cvState" height="130"></canvas>
  <canvas id="cvSlow" height="90"></canvas>

  <div class="row">
    <div class="panel">
      <div class="label-big" id="stateLabel" style="background:#9aa0a6">STOP</div>
      <div class="controls">
        <button id="btnPlay">⏸ 暂停</button>
        <span class="tick">速度</span>
        <select id="speed">
          <option value="0.5">0.5x</option><option value="1" selected>1x</option>
          <option value="4">4x</option><option value="16">16x</option><option value="64">64x</option>
        </select>
        <input type="range" id="scrub" min="0" max="1" step="0.0005" value="0">
      </div>
      <div class="tick" id="timeLabel" style="margin-top:4px"></div>
    </div>
    <div class="panel">
      <div class="legend">
        <span><span class="dot" style="background:#1f77b4"></span>A相</span>
        <span><span class="dot" style="background:#ff7f0e"></span>B相</span>
        <span><span class="dot" style="background:#2ca02c"></span>C相</span>
        <span><span class="dot" style="background:#d62728"></span>堵转事件</span>
        <span><span class="dot" style="background:#ff7f0e"></span>转子条边带(10-14s)</span>
      </div>
      <div id="eventLog"></div>
      <div class="tick">提示：橙色横向高亮=转子条边带注入区，红色=堵转。波形窗口为当前时刻前 2 秒。</div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const frames = DATA.frames, wave = DATA.wave, events = DATA.events, slow = DATA.slow;
const STATE_ORDER = ["STOP","TRANSIENT","IDLE","LOAD","STALL","UNKNOWN"];
const SC = {STOP:"#9aa0a6",TRANSIENT:"#ff7f0e",IDLE:"#1f77b4",LOAD:"#2ca02c",STALL:"#d62728",UNKNOWN:"#9467bd"};
const T0 = frames[0].t, T1 = frames[frames.length-1].t;
const stride = DATA.meta.stride_s;

let playing = true, speed = 1, idx = 0;
const cvW = document.getElementById("cvWave"), cvS = document.getElementById("cvState"), cvL = document.getElementById("cvSlow");
const ctxW = cvW.getContext("2d"), ctxS = cvS.getContext("2d"), ctxL = cvL.getContext("2d");

function resize(cv){ const dpr = window.devicePixelRatio||1; const w = cv.clientWidth; if(cv.width !== w*dpr){ cv.width = w*dpr; cv.height = cv.height; } }
function setup(cv){ resize(cv); const dpr = window.devicePixelRatio||1; return cv.getContext("2d"); }

function Xall(t){ return (t-T0)/(T1-T0); }
// 防御式帧索引：idx 始终钳制到 [0, len-1] 且取整
function fi(){ return Math.max(0, Math.min(Math.floor(idx), frames.length-1)); }
function drawWave(){
  const dpr=window.devicePixelRatio||1, W=cvW.width, H=cvW.height;
  ctxW.clearRect(0,0,W,H);
  const tNow = frames[fi()].t;
  const win=2.0, x0=tNow-win, x1=tNow+win*0.15;
  const X=t=>(t-x0)/(x1-x0)*W, Y=v=>H/2 - v*(H/2-10*dpr);
  const st=frames[fi()].state;
  ctxW.fillStyle=SC[st]; ctxW.globalAlpha=0.12; ctxW.fillRect(0,0,W,H); ctxW.globalAlpha=1;
  ctxW.strokeStyle="#2a2e35"; ctxW.beginPath();
  for(let g=Math.ceil(x0); g<=x1; g++){ ctxW.moveTo(X(g),0); ctxW.lineTo(X(g),H); }
  ctxW.stroke();
  for(const [k,c] of [["a","#1f77b4"],["b","#ff7f0e"],["c","#2ca02c"]]){
    const arr=wave[k]; ctxW.strokeStyle=c; ctxW.lineWidth=1.2*dpr; ctxW.beginPath();
    let started=false;
    for(let i=0;i<arr.length;i++){ const t=wave.t[i]; if(t<x0)continue; if(t>x1)break;
      const x=X(t),y=Y(arr[i]); if(started)ctxW.lineTo(x,y); else{ctxW.moveTo(x,y);started=true;} }
    ctxW.stroke();
  }
}
function drawState(){
  const dpr=window.devicePixelRatio||1, W=cvS.width, H=cvS.height;
  ctxS.clearRect(0,0,W,H);
  const X=t=>Xall(t)*W, Yr=r=>H - r*(H-14*dpr);
  const upto = fi();
  // 状态色带
  let pt=frames[0].t, ps=frames[0].state;
  for(let i=1;i<=upto;i++){ const f=frames[i];
    ctxS.fillStyle=SC[ps]; ctxS.globalAlpha=0.7; ctxS.fillRect(X(pt),0,X(f.t)-X(pt),H);
    pt=f.t; ps=f.state; }
  ctxS.globalAlpha=1;
  // 事件高亮
  for(const e of events){ const a=X(e.start_ms/1000), b=X(e.end_ms/1000);
    ctxS.fillStyle="#d62728"; ctxS.globalAlpha=0.25; ctxS.fillRect(a,0,b-a,H); }
  ctxS.globalAlpha=1;
  // RMS 包络（3相）
  for(const [ch,c] of [[0,"#1f77b4"],[1,"#ff7f0e"],[2,"#2ca02c"]]){
    ctxS.strokeStyle=c; ctxS.globalAlpha=0.9; ctxS.lineWidth=1*dpr; ctxS.beginPath();
    for(let i=0;i<=upto;i++){ const f=frames[i];
      const x=X(f.t), y=Yr(f.rms[ch]); i?ctxS.lineTo(x,y):ctxS.moveTo(x,y); }
    ctxS.stroke();
  }
  ctxS.globalAlpha=1;
  // 当前时间线
  const tn=frames[fi()].t;
  ctxS.strokeStyle="#fff"; ctxS.globalAlpha=0.4; ctxS.beginPath(); ctxS.moveTo(X(tn),0); ctxS.lineTo(X(tn),H); ctxS.stroke(); ctxS.globalAlpha=1;
  ctxS.fillStyle="#9aa0a6"; ctxS.font=(10*dpr)+"px sans-serif";
  STATE_ORDER.forEach((s,i)=>ctxS.fillText(s,3*dpr,i*(H/6)+8*dpr));
  ctxS.fillStyle="#5f6368"; ctxS.fillText("RMS包络(细线) + 状态色带 + 事件(红)",3*dpr,H-4*dpr);
}
function drawSlow(){
  const dpr=window.devicePixelRatio||1, W=cvL.width, H=cvL.height;
  ctxL.clearRect(0,0,W,H);
  const X=t=>Xall(t)*W;
  const tn=frames[fi()].t;
  const sl=slow.filter(s=>s.t<=tn);
  const mx=Math.max(0.02, ...sl.map(s=>s.sideband));
  ctxL.strokeStyle="#ff7f0e"; ctxL.lineWidth=1.4*dpr; ctxL.beginPath();
  sl.forEach((s,i)=>{ const x=X(s.t), y=H-(s.sideband/mx)*(H-12*dpr); i?ctxL.lineTo(x,y):ctxL.moveTo(x,y); });
  ctxL.stroke();
  // 注入区标记
  ctxL.fillStyle="#ff7f0e"; ctxL.globalAlpha=0.12; ctxL.fillRect(X(10),0,X(14)-X(10),H); ctxL.globalAlpha=1;
  ctxL.fillStyle="#9aa0a6"; ctxL.font=(11*dpr)+"px sans-serif";
  ctxL.fillText("慢路径·转子条边带比（橙色） 已见窗="+sl.length, 4*dpr, 12*dpr);
}
function updateUI(){
  const f=frames[fi()];
  const el=document.getElementById("stateLabel");
  el.textContent = f.state + "  ·  " + f.t.toFixed(1) + "s";
  el.style.background = SC[f.state];
  document.getElementById("timeLabel").textContent = "播放进度 " + Math.round(Xall(f.t)*100) + "%  ·  帧 " + (fi()+1) + "/" + frames.length;
  document.getElementById("scrub").value = Xall(f.t);
  const ev=document.getElementById("eventLog");
  ev.textContent = events.length ? ("报警事件: " + events.map(e=>e.start_ms/1000+"s ~ "+e.end_ms/1000+"s").join("；")) : "暂无报警事件";
}
let last=performance.now();
function loop(now){
  const dt=(now-last)/1000; last=now;
  if(playing){
    if(idx >= frames.length-1){ idx = 0; }  // 播完自动从头循环，连续运行
    else { idx = Math.max(0, Math.min(Math.floor(idx + dt*speed/stride), frames.length-1)); }
  }
  resize(cvW);resize(cvS);resize(cvL);
  drawWave();drawState();drawSlow();updateUI();
  requestAnimationFrame(loop);
}
document.getElementById("btnPlay").onclick=()=>{ playing=!playing; document.getElementById("btnPlay").textContent=playing?"⏸ 暂停":"▶ 播放"; };
document.getElementById("speed").onchange=e=>{ speed=parseFloat(e.target.value); };
document.getElementById("scrub").oninput=e=>{ idx=Math.floor(parseFloat(e.target.value)*(frames.length-1)); playing=false; document.getElementById("btnPlay").textContent="▶ 播放"; };
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
