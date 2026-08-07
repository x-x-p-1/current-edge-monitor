"""
5kW 电机物理模式 · 交互式故障注入动态看板
（运行：python 07_测试工具/export_5kw_live_dashboard.py）
输出：sim/dashboard_5kw_live.html（双击浏览器打开，数据内嵌，无依赖）

交互方式：
  · 默认"正常运行"（5kW 满载 ≈10.5A），三相电流实时滚动
  · 点击故障按钮 → 从点击瞬间起，电流变成对应故障（相位连续，幅值跳变）：
      堵转 STALL     → ×6  → ≈63A
      负载突变/过载  → ×1.27 → ≈13.4A (130% 负载)
      三相不平衡     → B×0.95 / C×1.05
  · 点击"恢复正常" → 回到正常运行
  · 状态栏会短暂显示 TRANSIENT（模拟真实状态机的瞬态判定）后落入稳态
  · 底部电流历史曲线记录每次故障注入的时间与电流水平

实现：生成 4s 无缝循环的正常满载基波信号，故障 = 物理倍率变换（同相位），
     浏览器端按下按钮瞬间切换倍率，模拟"现场制造故障"。
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
generate_dataset = _sim.generate_dataset
Motor = _sim.Motor

FS = 16000.0
DISPLAY_HZ = 1000          # 显示降采样
LOOP_S = 4.0               # 无缝循环时长（整周期数）


def main():
    motor = Motor()
    # 基础正常信号：满载，4s（整工频周期数 → 无缝循环），同相位
    base, _, _ = generate_dataset(
        duration=LOOP_S, f1=50.0, motor=motor,
        load_profile=[(0.0, LOOP_S, 1.0)], seed=7,
    )
    step = int(FS / DISPLAY_HZ)
    disp = base[::step]                 # (N, 3) 显示用
    disp_t = np.arange(len(disp)) * step / FS
    base_rms = np.sqrt(np.mean(disp ** 2, axis=0))   # ≈ 额定 10.5A/相

    # 故障倍率 = 物理目标电流 / 额定电流（保证与基波同相位，切换即幅值跳变）
    # 负载突变目标 = 200% 过载（→ 电流从 10.5A 跳到 ~20.2A，肉眼可见）
    f_over = motor.phase_current_a(2.0) / motor.phase_current_a(1.0)
    f_stall = motor.stall_current_a / motor.rated_current_a

    faults = {
        "normal":    {"factor": [1.0, 1.0, 1.0],
                      "state": "LOAD", "label": "正常运行", "color": "#2ca02c"},
        "overload":  {"factor": [f_over, f_over, f_over],
                      "state": "LOAD", "label": "负载突变(200%过载)", "color": "#9467bd"},
        "stall":     {"factor": [f_stall, f_stall, f_stall],
                      "state": "STALL", "label": "堵转", "color": "#d62728"},
        "unbalance": {"factor": [1.0, 0.95, 1.05],
                      "state": "LOAD", "label": "三相不平衡", "color": "#8c564b"},
    }

    data = {
        "loop_s": LOOP_S,
        "n": len(disp),
        "dt_s": step / FS,
        "wave": {"t": [round(v, 4) for v in disp_t],
                 "a": [round(v, 4) for v in disp[:, 0]],
                 "b": [round(v, 4) for v in disp[:, 1]],
                 "c": [round(v, 4) for v in disp[:, 2]]},
        "base_rms": [round(v, 3) for v in base_rms],
        "refs": {"no_load": round(motor.no_load_current_a, 2),
                 "rated": round(motor.rated_current_a, 2),
                 "stall": round(motor.stall_current_a, 2)},
        "faults": faults,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    os.makedirs("sim", exist_ok=True)
    out = os.path.join("sim", "dashboard_5kw_live.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"看板已生成: {os.path.abspath(out)}")
    print(f"循环={LOOP_S}s 波形点={len(disp)}  倍率: 过载×{f_over:.2f} 堵转×{f_stall:.2f}")


TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>5kW 电机 · 交互式故障注入看板</title>
<style>
  body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
  .wrap{max-width:1120px;margin:0 auto;padding:16px}
  h1{font-size:18px;margin:0 0 2px}
  .sub{color:#9aa0a6;font-size:12px;margin-bottom:12px}
  canvas{background:#15181d;border:1px solid #2a2e35;border-radius:6px;width:100%;display:block;margin-top:8px}
  .row{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
  .panel{flex:1;min-width:300px}
  .label-big{font-size:24px;font-weight:700;text-align:center;padding:12px;border-radius:6px;border:1px solid #2a2e35;font-variant-numeric:tabular-nums}
  .faultbtns{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  button.fault{flex:1;min-width:120px;padding:12px 8px;border-radius:6px;border:1px solid #3c4043;background:#2a2e35;color:#e8eaed;font-size:14px;cursor:pointer;font-weight:600}
  button.fault:hover{filter:brightness(1.3)}
  button.fault.active{outline:3px solid #fff}
  button.fault b{display:block;font-size:18px}
  button.fault small{display:block;color:#c8ccd0;font-weight:400;font-size:11px;margin-top:2px}
  .controls{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
  button{background:#2a2e35;color:#e8eaed;border:1px solid #3c4043;border-radius:5px;padding:6px 14px;cursor:pointer}
  button:hover{background:#3c4043}
  select{background:#2a2e35;color:#e8eaed;border:1px solid #3c4043;border-radius:5px;padding:5px}
  input[type=range]{flex:1;min-width:200px}
  .legend span{display:inline-block;margin-right:14px;font-size:12px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
  #eventLog{font-size:12px;color:#f28b82;margin-top:6px;font-variant-numeric:tabular-nums}
  .tick{color:#5f6368;font-size:11px}
  #curBar{height:8px;background:#2a2e35;border-radius:4px;margin-top:6px;overflow:hidden}
  #curFill{height:100%;width:0%;background:linear-gradient(90deg,#1f77b4,#2ca02c,#d62728)}
</style>
</head>
<body>
<div class="wrap">
  <h1>5kW 电机 · 交互式故障注入看板</h1>
  <div class="sub">默认正常运行（满载 ≈10.5A）· 点故障按钮 → 从此刻起电流变成故障信号 · 点"恢复正常"回到正常</div>

  <canvas id="cvWave" height="220"></canvas>
  <canvas id="cvHist" height="110"></canvas>

  <div class="row">
    <div class="panel">
      <div class="label-big" id="stateLabel" style="background:#9aa0a6">—</div>
      <div class="tick" id="curLabel" style="text-align:center;margin-top:4px"></div>
      <div id="curBar"><div id="curFill"></div></div>
      <div class="controls">
        <button id="btnPlay">⏸ 暂停</button>
        <span class="tick">速度</span>
        <select id="speed">
          <option value="0.005">0.005x</option><option value="0.01">0.01x</option>
          <option value="0.02">0.02x</option><option value="0.05">0.05x</option>
          <option value="0.1">0.1x</option><option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option><option value="1" selected>1x</option>
          <option value="4">4x</option><option value="16">16x</option><option value="64">64x</option>
        </select>
        <span class="tick">微调</span>
        <input type="range" id="speedRange" min="0" max="300" value="300" style="min-width:130px;flex:0.5">
        <span class="tick" id="speedVal">1x</span>
        <span class="tick">量程</span>
        <select id="yscale">
          <option value="fixed" selected>固定±95A</option>
          <option value="auto">自动(自适应)</option>
        </select>
        <span class="tick">窗口</span>
        <select id="winSel">
          <option value="0.02">0.02s(单周期)</option>
          <option value="0.05">0.05s</option>
          <option value="0.2">0.2s</option>
          <option value="0.5">0.5s</option>
          <option value="1">1s</option>
          <option value="2" selected>2s</option>
        </select>
        <span class="tick">不平衡度</span>
        <input type="range" id="unbalRange" min="1" max="15" step="0.5" value="5" disabled style="min-width:110px;flex:0.4">
        <span class="tick" id="unbalVal">5%</span>
        <span class="tick" id="timeLabel"></span>
      </div>
    </div>
    <div class="panel">
      <div class="legend">
        <span><span class="dot" style="background:#1f77b4"></span>A相</span>
        <span><span class="dot" style="background:#ff7f0e"></span>B相</span>
        <span><span class="dot" style="background:#2ca02c"></span>C相</span>
      </div>
      <div id="eventLog"></div>
      <div class="tick">底部曲线：A相电流水平历史（含空载/额定/堵转参考线），每点一次故障记一笔。</div>
    </div>
  </div>

  <div class="faultbtns">
    <button class="fault" id="btn_normal" data-f="normal" style="background:#2e7d32">
      <b>正常运行</b><small>≈10.5A 满载</small>
    </button>
    <button class="fault" id="btn_overload" data-f="overload" style="background:#6d4c92">
      <b>负载突变(200%过载)</b><small>×1.92 → ≈20.2A</small>
    </button>
    <button class="fault" id="btn_stall" data-f="stall" style="background:#b71c1c">
      <b>堵转</b><small>×6 → ≈63A</small>
    </button>
    <button class="fault" id="btn_unbalance" data-f="unbalance" style="background:#6d4c41">
      <b>三相不平衡</b><small id="unbalLbl">B↓ / C↑ 滑条可调</small>
    </button>
  </div>
</div>

<script>
const DATA = __DATA__;
const wave = DATA.wave, base_rms = DATA.base_rms, refs = DATA.refs, faults = DATA.faults;
const n = DATA.n, dt = DATA.dt_s;
const Y_AMP = 95, RMAX = 70;
const SC = {STOP:"#9aa0a6",TRANSIENT:"#ff7f0e",IDLE:"#1f77b4",LOAD:"#2ca02c",STALL:"#d62728",UNKNOWN:"#9467bd"};

let playing = true, speed = 1, idx = 0, active = "normal", elapsed = 0;
let transientUntil = 0, lastBtn = null;
let yMode = "fixed", winDur = 2.0;
let unbalDepth = 0.05;
const history = [];   // {t, level, label, color}

const cvW = document.getElementById("cvWave"), cvH = document.getElementById("cvHist");
const ctxW = cvW.getContext("2d"), ctxH = cvH.getContext("2d");

function resize(cv){ const dpr = window.devicePixelRatio||1; const w = cv.clientWidth; if(cv.width !== w*dpr){ cv.width = w*dpr; cv.height = cv.height; } }
function curState(){ return (performance.now() < transientUntil) ? "TRANSIENT" : faults[active].state; }
function effFactor(){ return (active==="unbalance") ? [1.0, 1-unbalDepth, 1+unbalDepth] : faults[active].factor; }

function setFault(key){
  active = key;
  transientUntil = performance.now() + 500;   // 故障瞬间 → 短暂 TRANSIENT
  document.querySelectorAll("button.fault").forEach(b=>b.classList.toggle("active", b.dataset.f===key));
  const slider = document.getElementById("unbalRange");
  slider.disabled = (key!=="unbalance");
  if(key==="unbalance"){
    document.getElementById("unbalLbl").textContent = "B↓"+Math.round(unbalDepth*100)+"% / C↑"+Math.round(unbalDepth*100)+"%";
  }
  history.push({t: elapsed, level: base_rms[0]*effFactor()[0], label: faults[key].label, color: faults[key].color});
  if(history.length>30) history.shift();
  const ev = document.getElementById("eventLog");
  ev.textContent = history.map(h=>h.t.toFixed(1)+"s → "+h.label).join("  ·  ");
}
document.getElementById("btn_normal").onclick=()=>setFault("normal");
document.getElementById("btn_overload").onclick=()=>setFault("overload");
document.getElementById("btn_stall").onclick=()=>setFault("stall");
document.getElementById("btn_unbalance").onclick=()=>setFault("unbalance");
setFault("normal");   // 初始：正常运行

function drawWave(){
  const dpr=window.devicePixelRatio||1, W=cvW.width, H=cvW.height;
  ctxW.clearRect(0,0,W,H);
  const i = Math.floor(idx);
  const i0 = i - Math.floor(winDur/dt);
  const st = curState();
  const f = faults[active], f0 = effFactor();
  // 动态量程：自动模式按当前故障峰值自适应；固定模式恒为 ±Y_AMP
  const peak = Math.max(...base_rms.map((r,k)=>r*f0[k]))*Math.SQRT2;
  const yAmp = (yMode==="auto") ? Math.max(5, peak*1.5) : Y_AMP;
  // 背景：状态色 + 故障色
  ctxW.fillStyle=SC[st]; ctxW.globalAlpha=0.08; ctxW.fillRect(0,0,W,H);
  ctxW.fillStyle=f.color; ctxW.globalAlpha=0.12; ctxW.fillRect(0,0,W,H); ctxW.globalAlpha=1;
  const X=j=>(j-i0)/(i-i0)*W, Y=v=>H/2 - v*(H/2-10*dpr)/yAmp;
  // 当前电流幅值带：随故障跳变，高度 = 三相峰值，一眼可见幅值变化
  const peakA = Math.max(...base_rms.map((r,k)=>r*f0[k]))*Math.SQRT2;
  ctxW.fillStyle=f.color; ctxW.globalAlpha=0.07;
  ctxW.fillRect(0, Y(peakA), W, Y(-peakA)-Y(peakA)); ctxW.globalAlpha=1;
  // 参考线（超出量程的自动跳过）
  for(const [v,c] of [[refs.no_load,"#3c4043"],[refs.rated,"#5f6368"],[refs.stall,"#d62728"]]){
    if(v>=yAmp) continue;
    ctxW.strokeStyle=c; ctxW.globalAlpha=0.4; ctxW.setLineDash([4,4]); ctxW.beginPath();
    ctxW.moveTo(0,Y(v)); ctxW.lineTo(W,Y(v)); ctxW.stroke(); ctxW.setLineDash([]); ctxW.globalAlpha=1;
  }
  const chmap={a:0,b:1,c:2};
  for(const [k,c] of [["a","#1f77b4"],["b","#ff7f0e"],["c","#2ca02c"]]){
    const arr=wave[k], m=f0[chmap[k]];
    ctxW.strokeStyle=c; ctxW.lineWidth=1.2*dpr; ctxW.beginPath();
    let started=false;
    for(let j=i0;j<=i;j++){ const v=arr[j%n]*m; const x=X(j), y=Y(v);
      if(started)ctxW.lineTo(x,y); else{ctxW.moveTo(x,y);started=true;} }
    ctxW.stroke();
  }
  ctxW.fillStyle="#9aa0a6"; ctxW.font=(10*dpr)+"px sans-serif";
  ctxW.fillText("实时三相电流(A) · 纵轴 ±"+yAmp.toFixed(0)+"A · 点下方按钮注入故障", 4, H-4);
}

function drawHist(){
  const dpr=window.devicePixelRatio||1, W=cvH.width, H=cvH.height;
  ctxH.clearRect(0,0,W,H);
  const X=t=>(t - Math.max(0, elapsed-12))/(elapsed - Math.max(0, elapsed-12) || 1)*W;
  // 参考线
  for(const [v,c] of [[refs.no_load,"#9aa0a6"],[refs.rated,"#e8eaed"],[refs.stall,"#d62728"]]){
    ctxH.strokeStyle=c; ctxH.globalAlpha=0.6; ctxH.setLineDash([4,4]); ctxH.beginPath();
    ctxH.moveTo(0,H - v*(H-14*dpr)/RMAX); ctxH.lineTo(W,H - v*(H-14*dpr)/RMAX); ctxH.stroke(); ctxH.setLineDash([]); ctxH.globalAlpha=1;
  }
  // 电流水平历史（阶梯线）
  if(history.length){
    ctxH.strokeStyle="#1f77b4"; ctxH.lineWidth=1.6*dpr; ctxH.beginPath();
    let started=false;
    for(const h of history){ const x=X(h.t), y=H - Math.min(h.level,RMAX)*(H-14*dpr)/RMAX;
      if(started)ctxH.lineTo(x,y); else{ctxH.moveTo(x,y);started=true;} }
    const cur=base_rms[0]*effFactor()[0];
    ctxH.lineTo(W, H - Math.min(cur,RMAX)*(H-14*dpr)/RMAX); ctxH.stroke();
    // 事件点
    for(const h of history){ const x=X(h.t), y=H - Math.min(h.level,RMAX)*(H-14*dpr)/RMAX;
      ctxH.fillStyle=h.color; ctxH.beginPath(); ctxH.arc(x,y,3*dpr,0,7); ctxH.fill(); }
  }
  ctxH.fillStyle="#5f6368"; ctxH.font=(10*dpr)+"px sans-serif";
  ctxH.fillText("A相电流水平历史(A) · 点=故障注入时刻 · 虚线=空载/额定/堵转", 4, H-4);
}

function updateUI(){
  const f=faults[active], st=curState();
  const fac=effFactor();
  const ia=base_rms[0]*fac[0], ib=base_rms[1]*fac[1], ic=base_rms[2]*fac[2];
  const el=document.getElementById("stateLabel");
  el.textContent = st + "  ·  " + f.label;
  el.style.background = SC[st];
  document.getElementById("curLabel").textContent =
    "A相 "+ia.toFixed(1)+" A ("+Math.round(ia/refs.rated*100)+"%额定) · B相 "+ib.toFixed(1)+" · C相 "+ic.toFixed(1)+"   (额定 "+refs.rated.toFixed(1)+"A / 堵转 "+refs.stall.toFixed(0)+"A)";
  document.getElementById("curFill").style.width = Math.min(100, ia/refs.stall*100) + "%";
  document.getElementById("timeLabel").textContent = "运行 "+elapsed.toFixed(1)+" s";
}

let last=performance.now();
function loop(now){
  const d=(now-last)/1000; last=now;
  if(playing){
    idx += d*speed/dt;
    elapsed += d*speed;
    if(idx>=n) idx-=n;
  }
  resize(cvW);resize(cvH);
  drawWave();drawHist();updateUI();
  requestAnimationFrame(loop);
}
document.getElementById("btnPlay").onclick=()=>{ playing=!playing; document.getElementById("btnPlay").textContent=playing?"⏸ 暂停":"▶ 播放"; };
document.getElementById("speed").onchange=e=>{ speed=parseFloat(e.target.value); document.getElementById("speedRange").value=Math.round((Math.log10(speed)+3)*100); document.getElementById("speedVal").textContent=speed+"x"; };
document.getElementById("speedRange").oninput=e=>{ speed=Math.pow(10,-3+parseFloat(e.target.value)/100); document.getElementById("speedVal").textContent=speed.toFixed(3)+"x"; document.getElementById("speed").value="custom"; };
document.getElementById("yscale").onchange=e=>{ yMode=e.target.value; };
document.getElementById("winSel").onchange=e=>{ winDur=parseFloat(e.target.value); };
document.getElementById("unbalRange").oninput=e=>{
  unbalDepth = parseFloat(e.target.value)/100;
  document.getElementById("unbalVal").textContent = Math.round(unbalDepth*100)+"%";
  if(active==="unbalance"){
    document.getElementById("unbalLbl").textContent = "B↓"+Math.round(unbalDepth*100)+"% / C↑"+Math.round(unbalDepth*100)+"%";
  }
};
(function(){ const s=parseFloat(document.getElementById("speed").value); document.getElementById("speedRange").value=Math.round((Math.log10(s)+3)*100); document.getElementById("speedVal").textContent=s+"x"; })();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
