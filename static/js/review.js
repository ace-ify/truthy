/* Truthy — reviewer queue (dark). Requires x-reviewer-key. */
(() => {
  "use strict";
  const $=(id)=>document.getElementById(id);
  const esc=(s)=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const icons=()=>{if(window.lucide)lucide.createIcons();};
  let KEY="";
  const RISKTX={high:"text-red-400",medium:"text-amber-400",low:"text-emerald-400"};
  const RISKIC={high:"octagon-alert",medium:"help-circle",low:"shield-check"};

  $("load").addEventListener("click",loadQueue);
  $("key").addEventListener("keydown",e=>{if(e.key==="Enter")loadQueue();});

  async function loadQueue(){
    KEY=$("key").value.trim(); const ae=$("autherr"); ae.classList.add("hidden");
    try{
      const r=await fetch("/api/review",{headers:{"x-reviewer-key":KEY}});
      if(!r.ok){const d=await r.json().catch(()=>({}));throw new Error(d.message||`Error ${r.status}`);}
      render((await r.json()).cases||[]);
    }catch(e){ae.textContent=e.message;ae.classList.remove("hidden");}
  }

  function render(cases){
    const q=$("queue"); q.classList.remove("hidden");
    if(!cases.length){q.innerHTML=`<div class="glass rounded-3xl p-7 text-gray-400 text-sm">Queue is empty — nothing awaiting review.</div>`;icons();return;}
    q.innerHTML=cases.map(c=>`
      <div class="glass rounded-3xl p-6 animate-fade-in" data-id="${esc(c.case_id)}">
        <div class="flex items-center gap-3">
          <i data-lucide="${RISKIC[c.risk_level]||'help-circle'}" class="w-6 h-6 ${RISKTX[c.risk_level]||'text-amber-400'}"></i>
          <div><h2 class="text-lg font-semibold">${esc(c.verdict)} · <span class="text-gray-400 font-normal">${esc(c.media_type)}</span></h2>
            <p class="text-xs text-gray-500 font-mono">${esc(c.case_id)} · ${esc(c.confidence)} confidence · ${esc(c.created_at)}</p></div>
          <a href="/api/case/${encodeURIComponent(c.case_id)}/report" target="_blank" rel="noopener" class="ml-auto text-xs text-blue-400 hover:text-blue-300 inline-flex items-center gap-1"><i data-lucide="external-link" class="w-3.5 h-3.5"></i> Report</a>
        </div>
        <input class="note w-full mt-4 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500/50" placeholder="Reviewer note — what did you decide and why?">
        <div class="flex flex-wrap gap-2 mt-3">
          <button data-d="confirm" class="dec rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-4 py-2 text-sm">Confirm verdict</button>
          <button data-d="override_authentic" class="dec rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-4 py-2 text-sm">Mark authentic</button>
          <button data-d="override_fake" class="dec rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 px-4 py-2 text-sm">Mark fake</button>
        </div>
        <p class="status text-xs text-gray-500 mt-2"></p>
      </div>`).join("");
    q.querySelectorAll("[data-id]").forEach(card=>{
      const id=card.dataset.id;
      card.querySelectorAll(".dec").forEach(b=>b.addEventListener("click",()=>decide(card,id,b.dataset.d)));
    });
    icons();
  }

  async function decide(card,id,decision){
    const st=card.querySelector(".status"); st.textContent="Saving…";
    const fd=new FormData(); fd.append("decision",decision); fd.append("note",card.querySelector(".note").value);
    try{
      const r=await fetch(`/api/case/${encodeURIComponent(id)}/review`,{method:"POST",headers:{"x-reviewer-key":KEY},body:fd});
      const d=await r.json().catch(()=>({})); if(!r.ok)throw new Error(d.message||`Error ${r.status}`);
      st.textContent=`Recorded: ${decision}. Case is now reviewed.`;
      card.querySelectorAll(".dec").forEach(b=>b.disabled=true);
    }catch(e){st.textContent="Failed: "+e.message;}
  }
})();
