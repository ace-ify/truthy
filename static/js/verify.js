/* Truthy — unified verify client (dark, matches landing). One upload flow + one report renderer. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = (x) => Math.round((Number(x)||0)*100);
  const icons = () => { if (window.lucide) lucide.createIcons(); };

  const EXT = { image:["jpg","jpeg","png","webp","bmp","gif"], voice:["mp3","wav","m4a","flac","ogg","aac"], video:["mp4","webm","avi","mov","mkv"] };
  const MAXMB = { image:20, voice:50, video:50 };
  const state = { type:"image", file:null };

  const drop=$("drop"), fileIn=$("file"), go=$("go"), errEl=$("err");
  const upload=$("upload"), loading=$("loading"), result=$("result");
  const modeField=$("mode-field"), langField=$("lang-field"), consent=$("consent");

  function selectTab(t){
    state.type=t;
    document.querySelectorAll(".tab").forEach(b=>{
      const on=b.dataset.type===t;
      b.className="tab flex flex-col items-center gap-1 py-3 rounded-xl text-sm font-medium transition-all "+
        (on?"bg-white/15 text-white shadow-sm":"text-gray-400 hover:text-white hover:bg-white/5");
    });
    modeField.classList.toggle("hidden", t!=="image");
    langField.classList.toggle("hidden", t!=="voice");
    fileIn.setAttribute("accept", EXT[t].map(e=>"."+e).join(","));
    resetFile();
  }
  document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>selectTab(b.dataset.type)));

  function showErr(m){errEl.textContent=m;errEl.classList.remove("hidden");}
  function hideErr(){errEl.classList.add("hidden");}
  function setBtn(){ go.disabled=!(state.file&&consent.checked); }
  consent.addEventListener("change",setBtn);

  function resetFile(){
    state.file=null; fileIn.value="";
    $("drop-default").classList.remove("hidden"); $("drop-preview").classList.add("hidden");
    $("drop-preview").classList.remove("flex");
    setBtn(); hideErr();
  }
  function handleFile(f){
    const ext=(f.name.split(".").pop()||"").toLowerCase();
    if(!EXT[state.type].includes(ext)){showErr(`Choose a ${state.type} file (${EXT[state.type].join(", ")}).`);return;}
    if(f.size>MAXMB[state.type]*1024*1024){showErr(`File too large. Max ${MAXMB[state.type]} MB.`);return;}
    hideErr(); state.file=f;
    $("preview-name").textContent=f.name; $("preview-size").textContent=(f.size/1024).toFixed(0)+" KB";
    const img=$("preview-img"), ic=$("preview-icon");
    if(state.type==="image"){ const r=new FileReader(); r.onload=e=>{img.src=e.target.result;}; r.readAsDataURL(f);
      img.classList.remove("hidden"); ic.classList.add("hidden"); ic.classList.remove("flex"); }
    else { img.classList.add("hidden"); ic.classList.remove("hidden"); ic.classList.add("flex"); }
    $("drop-default").classList.add("hidden"); $("drop-preview").classList.remove("hidden"); $("drop-preview").classList.add("flex");
    setBtn(); icons();
  }
  drop.addEventListener("click",()=>fileIn.click());
  fileIn.addEventListener("change",e=>{if(e.target.files[0])handleFile(e.target.files[0]);});
  ["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drop-active");}));
  ["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drop-active");}));
  drop.addEventListener("drop",e=>{if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0]);});

  const LOAD={image:"Running the AI classifier, forensic signals, and provenance…",voice:"Splitting speech, running the voice-clone model…",video:"Sampling frames, checking the voice track and visual signals…"};
  go.addEventListener("click", async ()=>{
    if(!state.file||!consent.checked) return;
    upload.classList.add("hidden"); result.classList.add("hidden"); loading.classList.remove("hidden");
    $("loading-title").textContent="Analyzing your "+state.type+"…"; $("loading-sub").textContent=LOAD[state.type];
    const fd=new FormData(); fd.append("file",state.file); fd.append("consent","true");
    fd.append("mode",$("mode").value); fd.append("language",$("lang").value);
    try{
      const resp=await fetch("/api/verify",{method:"POST",body:fd});
      const data=await resp.json().catch(()=>({}));
      if(!resp.ok){let m=data.message||`Server error (${resp.status})`; if(resp.status===502||resp.status===504)m="Server was busy or timed out. Try again, Quick mode, or a smaller file."; throw new Error(m);}
      renderReport(data);
    }catch(e){ loading.classList.add("hidden"); upload.classList.remove("hidden"); showErr(e.message||"Analysis failed."); }
  });
  // RENDER_BELOW
  const RISK={
    high:{bd:"border-red-500/30",tx:"text-red-400",ib:"bg-red-500/10",ic:"octagon-alert",w:"High risk",bar:"bg-gradient-to-r from-orange-500 to-red-500",rb:"border-red-500"},
    medium:{bd:"border-amber-500/30",tx:"text-amber-400",ib:"bg-amber-500/10",ic:"help-circle",w:"Needs review",bar:"bg-gradient-to-r from-yellow-500 to-amber-500",rb:"border-amber-500"},
    low:{bd:"border-emerald-500/30",tx:"text-emerald-400",ib:"bg-emerald-500/10",ic:"shield-check",w:"Low risk",bar:"bg-gradient-to-r from-emerald-500 to-green-500",rb:"border-emerald-500"},
  };
  const VT={AI_GENERATED:"Likely fake / AI-generated",HUMAN:"Likely authentic",INCONCLUSIVE:"Not confirmed, needs review"};
  const rm=(r)=>RISK[r]||RISK.medium;
  const tierBadge=(t)=>t==="primary"?'<span class="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded bg-blue-500 text-white">primary</span>':'<span class="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded bg-white/10 text-gray-400">support</span>';

  function sigRow(s){
    const p=pct(s.score), hi=s.score>0.5, lo=s.score<0.35;
    const tc=hi?"text-red-400":lo?"text-emerald-400":"text-gray-300";
    const bc=hi?"bg-red-500":lo?"bg-emerald-500":"bg-gray-500";
    return `<div><div class="flex items-center gap-2 mb-1.5">${tierBadge(s.tier)}<span class="text-sm text-gray-200">${esc(s.display_name)}</span><span class="ml-auto text-sm font-semibold ${tc}">${p}%</span></div>
      <div class="h-1.5 bg-white/5 rounded-full overflow-hidden"><div class="h-full rounded-full sigbar ${bc}" style="--target:${p}%;width:${p}%"></div></div>
      ${s.reasoning?`<p class="text-xs text-gray-500 mt-1.5">${esc(s.reasoning)}</p>`:""}</div>`;
  }
  function evidenceHtml(env){
    const u=(env.signals||[]).filter(s=>!s.error);
    const prim=u.filter(s=>s.tier==="primary"), sup=u.filter(s=>s.tier!=="primary").sort((a,b)=>Math.abs(b.score-.5)-Math.abs(a.score-.5));
    const x=env.extras||{}; let extra="";
    if(env.mediaType==="image"&&x.heatmapBase64) extra+=`<div><p class="text-xs uppercase tracking-widest text-gray-500 mb-2">Attention heatmap · illustrative</p><img class="max-w-[300px] rounded-xl border border-white/10" src="data:image/png;base64,${x.heatmapBase64}" alt="heatmap"></div>`;
    if(env.mediaType==="video") extra+=`<div class="text-xs text-gray-400">Frames ${esc(x.totalFramesAnalyzed)} · est. BPM ${esc(x.detectedBpm??"—")} · lip-sync mismatches ${esc(x.biomechanicalViolations??0)}</div>`;
    if(env.mediaType==="voice") extra+=`<div class="text-xs text-gray-400">Language ${esc(x.language)} · speech segments ${esc(x.speechChunks)}/${esc(x.totalChunks)}</div>`;
    if(!prim.length&&!sup.length&&!extra) return "";
    return `<details class="glass rounded-3xl p-6 animate-fade-in"><summary class="flex items-center justify-between"><span class="font-semibold">Technical evidence</span><i data-lucide="chevron-down" class="w-5 h-5 text-gray-400"></i></summary>
      <div class="mt-5 space-y-5">
        ${prim.length?`<div><p class="text-xs uppercase tracking-widest text-blue-400 mb-3">Primary detector · model-backed</p><div class="space-y-4">${prim.map(sigRow).join("")}</div></div>`:""}
        ${sup.length?`<div><p class="text-xs uppercase tracking-widest text-gray-500 mb-3">Supporting signals · heuristic, cannot override the model</p><div class="space-y-4">${sup.map(sigRow).join("")}</div></div>`:""}
        ${extra}
      </div></details>`;
  }
  function provHtml(p){
    if(!p||!p.flag) return `<details class="glass rounded-3xl p-6"><summary class="flex items-center justify-between"><span class="font-semibold">Provenance &amp; origin</span><i data-lucide="chevron-down" class="w-5 h-5 text-gray-400"></i></summary><p class="text-sm text-gray-400 mt-4">No Content Credential or usable metadata found.</p></details>`;
    const c=p.c2pa||{},e=p.exif||{},q=p.quant||{};
    const tagc={ai:"bg-red-500/10 text-red-400",authentic:"bg-emerald-500/10 text-emerald-400"}[p.flag]||"bg-white/10 text-gray-400";
    const c2=c.present?(c.verified?"verified":"present, unverified"):"none found";
    const cam=e.present?[e.make,e.model].filter(Boolean).join(" ")||"—":"none (stripped by messaging apps / screenshots / AI tools)";
    const row=(k,v)=>`<div class="flex gap-3 py-2.5 border-b border-white/5 text-sm"><span class="w-44 flex-shrink-0 text-gray-500">${k}</span><span class="text-gray-200">${v}</span></div>`;
    return `<details class="glass rounded-3xl p-6"><summary class="flex items-center justify-between"><span class="font-semibold">Provenance &amp; origin</span><i data-lucide="chevron-down" class="w-5 h-5 text-gray-400"></i></summary><div class="mt-4">
      ${row("Assessment",`<span class="text-[10px] font-bold uppercase px-2 py-0.5 rounded ${tagc}">${esc(p.flag)}</span> ${esc(p.summary)}`)}
      ${row("Content Credentials",esc(c2)+(c.claim_generator?" · "+esc(c.claim_generator):""))}
      ${row("Camera EXIF",esc(cam)+(e.datetime?" · "+esc(e.datetime):"")+(e.software?" · sw: "+esc(e.software):""))}
      ${q.note?row("Compression",esc(q.note)):""}
    </div></details>`;
  }
  // RENDER2
  function renderReport(env){
    loading.classList.add("hidden"); result.classList.remove("hidden");
    const r=rm(env.riskLevel), aiP=pct(env.aiProbability);
    const calNote=env.calibrated?"calibrated confidence":"confidence not yet calibrated";
    const backed=env.modelBacked?"model-backed":"no confident model signal";
    const queued=env.reviewStatus==="queued", reviewed=env.reviewStatus==="reviewed";
    result.innerHTML=`
      <div class="glass rounded-3xl p-7 animate-fade-in border ${r.bd}">
        <div class="flex items-start gap-4">
          <div class="w-14 h-14 rounded-2xl flex items-center justify-center ${r.ib}"><i data-lucide="${r.ic}" class="w-7 h-7 ${r.tx}"></i></div>
          <div class="flex-1"><h2 class="text-2xl font-bold ${r.tx}">${esc(VT[env.verdict]||env.verdict)}</h2>
            <p class="text-sm text-gray-400 mt-1"><span class="font-semibold ${r.tx}">${r.w}</span> · ${esc(env.confidence)} confidence · ${esc(backed)}, ${esc(calNote)}</p></div>
          <div class="text-right"><div class="text-3xl font-bold ${r.tx}">${aiP}%</div><div class="text-[10px] uppercase tracking-widest text-gray-500">AI prob.</div></div>
        </div>
        <div class="mt-5"><div class="flex justify-between text-xs text-gray-500 mb-2"><span>Authentic</span><span>AI-generated</span></div>
          <div class="h-2 bg-white/5 rounded-full overflow-hidden"><div class="h-full rounded-full ${r.bar}" style="width:${aiP}%"></div></div></div>
        <div class="mt-5 rounded-xl bg-white/[0.03] border-l-2 ${r.rb} p-4"><div class="text-[11px] uppercase tracking-widest text-gray-500 mb-1">What you should do</div><p class="text-[15px] text-gray-100">${esc(env.recommendation)}</p></div>
        <p class="text-sm text-gray-400 mt-4">${esc(env.explanation)}</p>
      </div>
      ${evidenceHtml(env)}
      ${provHtml(env.provenance)}
      <div class="flex flex-wrap gap-3">
        <a href="/api/case/${encodeURIComponent(env.caseId)}/report" target="_blank" rel="noopener" class="inline-flex items-center gap-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-5 py-3 text-sm font-medium transition-all"><i data-lucide="download" class="w-4 h-4"></i> Download evidence report</a>
        <button id="escalate" ${queued||reviewed?"disabled":""} class="inline-flex items-center gap-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-5 py-3 text-sm font-medium transition-all disabled:opacity-50"><i data-lucide="flag" class="w-4 h-4"></i> ${reviewed?"Reviewed":queued?"Queued for review":"Send to human review"}</button>
        <button id="again" class="inline-flex items-center gap-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-5 py-3 text-sm font-medium transition-all"><i data-lucide="rotate-ccw" class="w-4 h-4"></i> Check another</button>
      </div>
      <div class="text-xs text-gray-600 leading-relaxed border-t border-white/10 pt-4">
        Case <b class="text-gray-400">${esc(env.caseId||"—")}</b> · ${esc(env.createdAt||"")}<br>
        SHA-256: <span class="font-mono">${esc((env.sha256||"").slice(0,32))}…</span><br>
        Truthy is decision-support, not legal proof. A detector score is not a verdict on a person. The model drives the call; heuristic signals only adjust confidence and can never override it. Media is processed transiently and not stored.
      </div>`;
    icons();
    const eb=$("escalate");
    if(eb&&!queued&&!reviewed) eb.addEventListener("click", async ()=>{
      eb.disabled=true; eb.innerHTML='Sending…';
      try{ const rr=await fetch(`/api/case/${encodeURIComponent(env.caseId)}/escalate`,{method:"POST"}); eb.innerHTML=rr.ok?"Queued for review":"Failed, retry"; }catch{eb.innerHTML="Failed, retry";}
    });
    $("again").addEventListener("click",()=>{ result.classList.add("hidden"); upload.classList.remove("hidden"); resetFile(); consent.checked=false; setBtn(); window.scrollTo({top:0,behavior:"smooth"}); });
    result.focus();
  }
  selectTab("image");
  if (document.readyState !== "loading") icons(); else document.addEventListener("DOMContentLoaded", icons);

  // Deep-link: /verify?case=TRV-XXXX opens a past case's report (read-only recall).
  const caseId = new URLSearchParams(location.search).get("case");
  if (caseId) {
    upload.classList.add("hidden");
    fetch(`/api/case/${encodeURIComponent(caseId)}`).then(r=>r.json()).then(d=>{
      if (d && d.verdict) renderReport(d); else showErr("Case not found: "+caseId);
    }).catch(()=>showErr("Could not load case "+caseId));
  }
})();
