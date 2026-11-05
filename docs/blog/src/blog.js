/* Qwen SM120 interactive article. All diagrams are deterministic, local and manually stepped. */
'use strict';
(() => {
  const $ = (s, root=document) => root.querySelector(s);
  const $$ = (s, root=document) => Array.from(root.querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const messages = JSON.parse(document.getElementById('diagram-messages').textContent);
  const msg = (key, ...values) => messages[key].replace(/\{(\d+)\}/g, (_, i) => values[i]);
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const colors = ['#4e86ce','#d69749','#8d74c4','#3fa495','#c7787d','#6c9eb0','#a2a04f','#b28aaf'];
  const rc = (i) => colors[i % colors.length];
  const fmt = (n,d=2) => Number(n).toFixed(d);
  const tx = (x,y,s,cls='diagram-label',attrs='') => `<text x="${x}" y="${y}" class="${cls}" ${attrs}>${esc(s)}</text>`;
  const rect = (x,y,w,h,fill='var(--panel-2)',stroke='var(--border)',r=5,attrs='') => `<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" ${attrs}/>`;
  const path = (d,active=false,id='',attrs='') => `<path d="${d}" class="diagram-edge ${active?'active flow-active':''}" ${id?`marker-end="url(#${id}-arrow)"`:''} ${attrs}/>`;
  function box(x,y,w,h,title,subtitle='',active=false,extra='') {
    return `<g ${extra}>${rect(x,y,w,h,active?'var(--accent-soft)':'var(--panel-2)',active?'var(--accent)':'var(--border)',7)}${tx(x+w/2,y+(subtitle?27:h/2+4),title,'diagram-title','text-anchor="middle"')}${subtitle?tx(x+w/2,y+46,subtitle,'diagram-small','text-anchor="middle"'):''}</g>`;
  }
  function svg(holder,inner,w,h,label,animate=false){
    const el=typeof holder==='string'?document.getElementById(holder):holder;
    const id=el.id || 'figure';
    const old=new Map();
    if(animate && !reduced()) $$('[data-morph]',el).forEach(n=>old.set(n.dataset.morph,n.getBoundingClientRect()));
    el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="${id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)"/></marker></defs>${inner}</svg>`;
    if(animate && !reduced()){
      el.firstElementChild.classList.add('scene-enter');
      $$('[data-morph]',el).forEach(n=>{const prev=old.get(n.dataset.morph);if(!prev)return;const now=n.getBoundingClientRect();const dx=prev.x-now.x,dy=prev.y-now.y;if(Math.abs(dx)+Math.abs(dy)>2)n.animate([{transform:`translate(${dx}px,${dy}px)`,opacity:.65},{transform:'translate(0,0)',opacity:1}],{duration:320,easing:'cubic-bezier(.2,.7,.3,1)'});});
    }
  }
  function steps(holder,labels,index,handler){
    const el=typeof holder==='string'?document.getElementById(holder):holder;
    const keepFocus=el.contains(document.activeElement);
    el.innerHTML=labels.map((l,i)=>`<button type="button" data-step="${i}" aria-pressed="${i===index}"><span class="step-num">${String(i+1).padStart(2,'0')}</span>${esc(l)}</button>`).join('');
    el.onclick=e=>{const b=e.target.closest('button[data-step]');if(b)handler(+b.dataset.step);};
    if(keepFocus)el.querySelector(`button[data-step="${index}"]`)?.focus({preventScroll:true});
  }
  function metrics(holder,items){document.getElementById(holder).innerHTML=items.map(([l,v,u,n])=>`<div><small>${esc(l)}</small><strong>${esc(v)}${u?`<em>${esc(u)}</em>`:''}</strong>${n?`<span>${esc(n)}</span>`:''}</div>`).join('');}
  function explanation(id,label,text){document.getElementById(id).innerHTML=`<span class="ex-label">${esc(label)}</span>${text}`;}

  // ---------- Reading UI ----------
  const themeBtn=$('#theme-toggle');
  function setTheme(theme){document.documentElement.dataset.theme=theme;$('#theme-icon').textContent=theme==='dark'?'☼':'☾';themeBtn.setAttribute('aria-label',theme==='dark'?msg('theme.light'):msg('theme.dark'));try{localStorage.setItem('qwen-theme',theme);}catch(e){}$('meta[name=theme-color]').setAttribute('content',theme==='dark'?'#11181f':'#faf9f6');}
  setTheme(document.documentElement.dataset.theme);
  themeBtn.addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
  const toc=$('#toc'),menu=$('#menu-toggle'),scrim=$('#toc-scrim');
  function menuState(open){toc.classList.toggle('menu-open',open);menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?msg('toc.close'):msg('toc.open'));scrim.hidden=!open;}
  menu.addEventListener('click',()=>menuState(!toc.classList.contains('menu-open')));
  scrim.addEventListener('click',()=>menuState(false));
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){menuState(false);$('#tooltip').hidden=true;}});
  function openAncestors(target){let n=target;if(n instanceof HTMLDetailsElement)n.open=true;while(n&&n!==document.body){if(n instanceof HTMLDetailsElement)n.open=true;n=n.parentElement;}}
  document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a)return;const target=document.getElementById(a.getAttribute('href').slice(1));if(target){openAncestors(target);menuState(false);}});
  if(location.hash){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target)openAncestors(target);}
  const tocLinks=$$('#toc nav a');
  const headings=tocLinks.map(a=>({a,el:document.getElementById(a.hash.slice(1))})).filter(x=>x.el);
  let scrolling=false;
  function scrollUI(){const denom=document.documentElement.scrollHeight-innerHeight;const pct=denom>0?Math.min(100,Math.max(0,scrollY/denom*100)):100;$('#reading-progress').style.width=pct+'%';$('#read-percent').textContent=Math.round(pct)+'%';let current=headings[0];headings.forEach(x=>{if(x.el.getClientRects().length && x.el.getBoundingClientRect().top<160)current=x;});tocLinks.forEach(a=>a.classList.toggle('active',a===current?.a));scrolling=false;}
  const languageSwitch = $('#language-switch');
  const otherPage = languageSwitch.getAttribute('href');
  function updateLanguageLink() {
    const active = tocLinks.find(a => a.classList.contains('active'));
    const hash = scrollY < 80 ? location.hash : (active?.hash || location.hash);
    languageSwitch.href = otherPage + hash;
  }
  languageSwitch.addEventListener('focus', updateLanguageLink);
  languageSwitch.addEventListener('pointerenter', updateLanguageLink);
  languageSwitch.addEventListener('click', updateLanguageLink);
  window.addEventListener('scroll',()=>{if(!scrolling){requestAnimationFrame(scrollUI);scrolling=true;}},{passive:true});
  window.addEventListener('resize',scrollUI);
  document.addEventListener('toggle',scrollUI,true);
  window.addEventListener('beforeprint',()=>$$('details').forEach(d=>{d.dataset.wasOpen=String(d.open);d.open=true;}));
  window.addEventListener('afterprint',()=>$$('details').forEach(d=>{if(d.dataset.wasOpen!==undefined){d.open=d.dataset.wasOpen==='true';delete d.dataset.wasOpen;}}));
  const tooltip=$('#tooltip');
  document.addEventListener('pointermove',e=>{const n=e.target.closest('[data-tip]');if(!n){tooltip.hidden=true;return;}tooltip.innerHTML=n.dataset.tip;tooltip.hidden=false;const b=tooltip.getBoundingClientRect();tooltip.style.left=Math.min(innerWidth-b.width-10,e.clientX+14)+'px';tooltip.style.top=Math.max(8,Math.min(innerHeight-b.height-10,e.clientY+14))+'px';});
  document.addEventListener('pointerleave',()=>tooltip.hidden=true);

  // ---------- CPP / VPP: equal-total-compute dependency model ----------
  function schedule(n,g,kind){
    const orders=[]; const nodes=new Map();
    for(let rank=0;rank<4;rank++){
      const order=[];
      if(kind==='cpp'){for(let j=0;j<n;j++)order.push([j,rank]);}
      else {const w=Math.min(7-2*rank,n);for(let j=0;j<w;j++)order.push([j,rank]);for(let j=0;j<n-w;j++){order.push([j,7-rank],[j+w,rank]);}for(let j=n-w;j<n;j++)order.push([j,7-rank]);}
      orders.push(order);
      order.forEach(([chunk,stage],k)=>{const deps=[];if(k)deps.push(orders[rank][k-1].join(':'));if(stage)deps.push([chunk,stage-1].join(':'));nodes.set([chunk,stage].join(':'),{chunk,stage,rank,duration:(1+g*chunk)*(kind==='vpp'?.5:1),deps:[...new Set(deps)]});});
    }
    const visiting=new Set();
    function end(key){const node=nodes.get(key);if(node.end!==undefined)return node.end;if(visiting.has(key))throw Error('Pipeline dependency cycle');visiting.add(key);node.start=Math.max(0,...node.deps.map(end));node.end=node.start+node.duration;visiting.delete(key);return node.end;}
    nodes.forEach((_,key)=>end(key));
    const all=[...nodes.values()];const finish=Math.max(...all.map(x=>x.end));const work=all.reduce((s,x)=>s+x.duration,0);
    return {nodes:all,orders,finish,work,idle:finish-work/4,get:(j,s)=>nodes.get(`${j}:${s}`)};
  }
  let pipeState={count:16,growth:.12,step:5,highlight:-1};
  function renderPipe(animate=false){
    const s=pipeState,cpp=schedule(s.count,s.growth,'cpp'),vpp=schedule(s.count,s.growth,'vpp');const total=Math.max(cpp.finish,vpp.finish);
    const times=[0,vpp.get(0,3).start+.2,vpp.get(0,4).start+.15,vpp.get(1,7).start+.15,vpp.get(s.count-1,0).end,total];const t=Math.min(total,times[s.step]);
    steps('pipe-steps',[msg('pipe.start'),msg('pipe.fill'),msg('pipe.turn'),msg('pipe.steady'),msg('pipe.drain'),msg('overview')],s.step,i=>{pipeState.step=i;renderPipe(true);});
    const select=$('#pipe-highlight');if(select.options.length!==s.count+1){select.innerHTML=msg('pipe.all')+Array.from({length:s.count},(_,j)=>`<option value="${j}">C${j}</option>`).join('');select.value=String(s.highlight);}
    $('#cpp-runtime').textContent=msg('pipe.finish', fmt(cpp.finish));$('#vpp-runtime').textContent=msg('pipe.finish', fmt(vpp.finish));
    function draw(data,kind,id){
      const W=960,H=207,left=77,right=24,pw=W-left-right;const xx=t=>left+t/total*pw;let a='';
      for(let i=0;i<=6;i++){const x=left+i*pw/6;a+=`<line x1="${x}" y1="26" x2="${x}" y2="177" class="grid"/>`+tx(x,194,fmt(total*i/6,1),'axis','text-anchor="middle"');}
      for(let r=0;r<4;r++){const y=37+r*35;a+=tx(15,y+15,'GPU'+r,'track-label');a+=rect(left,y,pw,25,'#1d2c3d','none',3);}
      data.nodes.forEach(e=>{const x=xx(e.start),y=37+e.rank*35,width=e.duration/total*pw,isR=kind==='vpp'&&e.stage>=4;let color=isR?'#9872c7':'#4d8fbd';const highlight=s.highlight<0||s.highlight===e.chunk;const past=e.start<t||s.step===5;const opacity=(highlight?1:.13)*(past?1:.2);const tip=msg('pipe.tip', kind.toUpperCase(), e.rank, e.chunk, e.stage, isR?'Return':'Forward', e.chunk, fmt(e.start), fmt(e.end), fmt(e.duration));
        a+=`<g opacity="${opacity}" data-tip="${esc(tip)}">${rect(x,y,Math.max(.7,width-1),25,color,'none',3,`data-morph="${kind}-${e.chunk}-${e.stage}"`)}${width>23?tx(x+width/2,y+16,`C${e.chunk}`,'block-label','text-anchor="middle"'):''}</g>`;
      });
      if(s.step!==5){a+=`<line x1="${xx(t)}" x2="${xx(t)}" y1="19" y2="179" stroke="#8fe0c1" stroke-width="1.6"/>`+tx(Math.min(W-60,xx(t)+5),15,`${fmt(t,1)} u`,'phase-label');}
      svg(id,a,W,H,msg('pipe.timeline', kind.toUpperCase()),animate);
    }
    draw(cpp,'cpp','cpp-timeline');draw(vpp,'vpp','vpp-timeline');
    const texts=[
      msg('pipe.startNote'),
      msg('pipe.fillNote'),
      msg('pipe.turnNote'),
      msg('pipe.steadyNote'),
      msg('pipe.drainNote'),
      s.growth===0?msg('pipe.uniformNote'):msg('pipe.growthNote', fmt(cpp.finish), fmt(vpp.finish))
    ];
    explanation('pipe-explanation',msg('pipe.phase', s.step+1),texts[s.step]);
    metrics('pipe-metrics',[[msg('pipe.cppIdle'),fmt(cpp.idle),'u',msg('pipe.elapsed')],[msg('pipe.vppIdle'),fmt(vpp.idle),'u',msg('pipe.equalWork')],[msg('pipe.delta'),fmt((cpp.finish-vpp.finish)/cpp.finish*100,1)+'%','','tⱼ = 1 + '+s.growth+'j']]);
  }
  $('#pipe-count').addEventListener('change',e=>{pipeState.count=+e.target.value;if(pipeState.highlight>=pipeState.count)pipeState.highlight=-1;renderPipe(true);});
  $('#pipe-growth').addEventListener('change',e=>{pipeState.growth=+e.target.value;renderPipe(true);});
  $('#pipe-highlight').addEventListener('change',e=>{pipeState.highlight=+e.target.value;renderPipe(true);});

  // ---------- Rank resources ----------
  let rankStep=0;
  function renderRank(animate=false){
    const id='rank-diagram';steps('rank-steps',[msg('overview'),msg('rank.receive'),msg('rank.compute'),msg('rank.send'),msg('rank.reuse')],rankStep,i=>{rankStep=i;renderRank(true);});
    const active=(n)=>rankStep===0||rankStep===n;let a='';
    a+=rect(193,48,565,287,'transparent','var(--border)',12)+tx(211,72,'GPU1 / RANK 1','diagram-title')+tx(740,72,'S1 · S6','diagram-small diagram-mono','text-anchor="end"');
    a+=box(15,92,150,66,msg('rank.from0'),'Forward / Edge0',active(1));
    a+=box(15,218,150,66,msg('rank.from2'),'Return / Edge5',active(1));
    a+=tx(87,185,msg('rank.edgeRings'),'diagram-small','text-anchor="middle"');
    a+=box(214,117,125,58,'H2D Stream',msg('rank.shared'),active(1));
    a+=rect(367,100,175,105,active(1)?'var(--accent-soft)':'var(--panel-2)',active(1)?'var(--accent)':'var(--border)',7);
    a+=tx(454,122,'Receive Arena','diagram-title','text-anchor="middle"');
    [msg('rank.current'),msg('rank.ahead1'),msg('rank.ahead2')].forEach((t,i)=>{a+=rect(380,132+i*22,149,17,i===0?'var(--blue-soft)':'var(--panel)','var(--border)',3)+tx(454,144+i*22,t,'diagram-small diagram-mono','text-anchor="middle"');});
    a+=box(367,249,175,61,'Compute Stream',msg('rank.interleave'),active(2));
    a+=rect(581,230,151,88,active(3)?'var(--accent-soft)':'var(--panel-2)',active(3)?'var(--accent)':'var(--border)',7);
    a+=tx(656,251,'Source Arena','diagram-title','text-anchor="middle"');
    a+=rect(592,264,62,37,'var(--panel)','var(--border)',4)+rect(660,264,62,37,'var(--panel)','var(--border)',4);
    a+=tx(623,279,'Slot 0','diagram-small diagram-mono','text-anchor="middle"')+tx(623,293,msg('rank.sendPrev'),'diagram-small','text-anchor="middle"')+tx(691,279,'Slot 1','diagram-small diagram-mono','text-anchor="middle"')+tx(691,293,msg('rank.writeNext'),'diagram-small','text-anchor="middle"');
    a+=box(581,117,151,58,'D2H Stream',msg('rank.shared'),active(3));
    a+=box(795,92,150,66,msg('rank.to2'),'Forward / Edge1',active(3));a+=box(795,218,150,66,msg('rank.to0'),'Return / Edge6',active(3));
    a+=tx(870,185,msg('rank.edgeRings'),'diagram-small','text-anchor="middle"');
    a+=path('M165 125 H190 V146 H214',active(1),id)+path('M165 250 H186 V146 H214',active(1),id);
    a+=path('M339 146 H367',active(1),id)+path('M454 205 V249',active(2),id)+path('M542 279 H581',active(2),id)+path('M656 230 V175',active(3),id);
    a+=path('M732 146 H774 V125 H795',active(3),id)+path('M732 146 H774 V251 H795',active(3),id);
    a+=box(245,369,460,62,'Host · C++ Progress Thread',msg('rank.progress'),active(4));
    a+=path('M275 369 V335 M473 369 V335 M657 369 V335',active(4),'','stroke-dasharray="4 5"');
    a+=tx(480,456,msg('rank.capacity'),'diagram-caption','text-anchor="middle"');
    svg(id,a,960,475,msg('rank.diagram'),animate);
    const desc=[
      msg('rank.overviewNote'),
      msg('rank.receiveNote'),
      msg('rank.computeNote'),
      msg('rank.sendNote'),
      msg('rank.reuseNote')
    ];explanation('rank-explanation','RANK-LOCAL DATAFLOW',desc[rankStep]);
  }

  // ---------- Segmented DMA timeline ----------
  let dmaStep=3;
  function renderDMA(animate=false){
    const id='dma-timeline';steps('dma-steps',[msg('dma.ready'),msg('dma.d2h'),msg('dma.h2d'),msg('dma.pipeline'),msg('dma.received'),msg('dma.release')],dmaStep,i=>{dmaStep=i;renderDMA(true);});
    const t=[2.8,3.0,3.08,4.55,5.7,9.2][dmaStep],W=960,H=400,left=145,pw=785,x=t=>left+t/10*pw;let a='';
    for(let i=0;i<=5;i++){a+=`<line x1="${x(i*2)}" x2="${x(i*2)}" y1="22" y2="361" class="grid"/>`+tx(x(i*2),383,String(i*2),'axis','text-anchor="middle"');}
    const rows=[['GPU0 · Compute',42],['GPU0 · CPU',86],['GPU0 · D2H',128],['Host · Ring ID',177],['GPU1 · CPU',225],['GPU1 · H2D',268],['GPU1 · Compute',322]];
    rows.forEach(([l,y])=>{a+=tx(12,y+14,l,'track-label')+rect(left,y,pw,24,'#1a2939','none',3);});
    function block(start,dur,y,label,color){a+=`<g opacity="${start<=t?1:.2}">${rect(x(start),y,dur/10*pw-1,24,color,'none',3)}${dur/10*pw>35?tx(x(start+dur/2),y+15,label,'block-label','text-anchor="middle"'):''}</g>`;}
    block(0,4,42,'S0(C0) · Source0','#4f8db9');block(4.15,4,42,'S0(C1) · Source1','#477397');
    block(5.6,3.5,322,'S1(C0)','#4f8db9');
    for(let i=0;i<10;i++){
      const d=2.9+.24*i,h=3.16+.24*i,ring=i%8;
      block(d,.22,128,'','#cc9855');block(h,.22,268,'','#48a59f');
      if(i===0||i===9){a+=tx(x(d+.11),122,`Seg${i}`,'axis','text-anchor="middle"')+tx(x(h+.11),301,`Seg${i}`,'axis','text-anchor="middle"');}
      const op=d<=t?1:.15;
      a+=`<g opacity="${op}">${rect(x(d),177,.22/10*pw,24,'#5c563e','#a68f5f',3)}${tx(x(d+.11),192,'R'+ring,'block-label','text-anchor="middle"')}</g>`;
      if(i===0||i===8){a+=`<path d="M${x(d+.22)} 152 V250 H${x(h)} V268" fill="none" stroke="#69b9b0" stroke-width="1.1" stroke-dasharray="3 3" opacity="${d<=t?.8:.16}"/>`;}
    }
    const marks=[[2.8,86,'Plane ready'],[2.91,86,'D2H + Event'],[2.96,86,'READY'],[3.02,225,'Wait + H2D'],[3.06,225,'ACK'],[5.54,225,'Recv done']];
    marks.forEach(([tm,y,l],i)=>{const yy=y+((i===2)?7:0);a+=`<circle cx="${x(tm)}" cy="${yy+10}" r="3" fill="#8ed2bb" opacity="${tm<=t?1:.2}"/>`;});
    a+=tx(x(2.8)-6,78,msg('dma.outputEvent'),'axis','text-anchor="end"');a+=tx(x(3.06)+9,241,msg('dma.wait'),'axis');
    a+=rect(x(3.02),268,(3.16-3.02)/10*pw,24,'none','#8ed2bb',2,'stroke-dasharray="2 2"');
    a+=`<path d="M${x(5.54)} 292 V313 H${x(5.6)} V322" fill="none" stroke="#8ed2bb" stroke-width="1.5"/>`;
    a+=`<line x1="${x(t)}" x2="${x(t)}" y1="15" y2="360" stroke="#8de0bd" stroke-width="1.6"/>`+tx(x(t)+4,12,fmt(t,2)+' u','phase-label');
    svg(id,a,W,H,msg('dma.diagram'),animate);
    const source=t>=5.28,ring=t>=5.54,dest=t>=9.1;
    $('#dma-credits').innerHTML=[['Source Slot0',source?msg('dma.free'):msg('dma.held'),source],['Host Ring',ring?msg('dma.read'):msg('dma.cycle'),ring],['Receive Slot',dest?msg('dma.free'):t>=5.54?msg('dma.consuming'):t>=3.02?msg('dma.writing'):msg('dma.waiting'),dest]].map(([l,v,free])=>`<div class="credit ${free?'free':''}"><strong>${l}</strong>${v}</div>`).join('');
    const desc=[
      msg('dma.readyNote'),
      msg('dma.d2hNote'),
      msg('dma.h2dNote'),
      msg('dma.pipelineNote'),
      msg('dma.receivedNote'),
      msg('dma.releaseNote')
    ];explanation('dma-explanation',`HANDOFF ${dmaStep+1} / 6`,desc[dmaStep]);
  }

  // ---------- XQA task geometry: port of task_plan() interval mode ----------
  function taskPlan(lengths,ctas){
    if(!Array.isArray(lengths)||!lengths.length||lengths.length>ctas||ctas>340||lengths.some(n=>!Number.isInteger(n)||n<0))throw Error('Invalid task geometry');
    const counts=lengths.map(n=>Math.ceil(n/256)),total=counts.reduce((a,b)=>a+b,0),budget=Math.max(2,Math.ceil(total/ctas));
    const tasks=[],requests=[],buckets=Array.from({length:ctas},()=>({begin:0,end:0}));let prefix=0;
    counts.forEach((n,r)=>{const begin=tasks.length;if(n){for(let b=Math.floor(prefix/budget);b<=Math.floor((prefix+n-1)/budget);b++){const lo=Math.max(prefix,b*budget)-prefix,hi=Math.min(prefix+n,(b+1)*budget)-prefix;if(buckets[b].begin===buckets[b].end)buckets[b].begin=tasks.length;tasks.push({request:r,begin:lo,end:hi,stride:1,bucket:b,slot:tasks.length});buckets[b].end=tasks.length;}}requests.push({begin,count:tasks.length-begin});prefix+=n;});
    const loads=buckets.map(b=>tasks.slice(b.begin,b.end).reduce((s,t)=>s+t.end-t.begin,0));
    return {counts,total,budget,tasks,requests,buckets,loads,active:loads.filter(n=>n>0).length,ctas,lengths};
  }
  function nativePlan(lengths,ctas){const counts=lengths.map(n=>Math.ceil(n/256)),max=Math.max(...counts,1);const splits=Math.min(Math.max(1,Math.floor(ctas/lengths.length)),max);const tasks=[];lengths.forEach((_,r)=>{for(let s=0;s<splits;s++)tasks.push({request:r,begin:s,end:counts[r],stride:splits,count:Math.max(0,Math.ceil((counts[r]-s)/splits))});});const loads=Array.from({length:ctas},(_,i)=>tasks[i]?.count||0);return {tasks,loads,splits,active:loads.filter(n=>n>0).length};}
  const presets={skew32:[130816,...Array(31).fill(4096)],skew9:[130816,...Array(8).fill(4096),...Array(7).fill(0)],uniform:Array(32).fill(8064)};
  let loadPreset='skew32';
  function renderLoad(animate=false){const lengths=presets[loadPreset],p=taskPlan(lengths,170),n=nativePlan(lengths,170);const max=Math.max(...n.loads,...p.loads),W=960,left=52,pw=877;let a='';
    [[n,msg('load.native'),48,'#99a9b7'],[p,'Global KV Bucket',217,'var(--accent)']].forEach(([d,name,top,color])=>{
      a+=tx(left,top-18,name,'diagram-title')+tx(930,top-18,msg('load.active', d.active),'diagram-small diagram-mono','text-anchor="end"');
      [0,.5,1].forEach(f=>{const y=top+105-105*f;a+=`<line x1="${left}" x2="${left+pw}" y1="${y}" y2="${y}" stroke="var(--border)" stroke-dasharray="3 4"/>`+tx(41,y+3,Math.round(max*f),'diagram-small diagram-mono','text-anchor="end"');});
      d.loads.forEach((v,i)=>{const x=left+i/170*pw,h=v/max*105;const tip=`<strong>CTA ${i} · ${name}</strong>${v} KV Tiles${d===n&&n.tasks[i]?`<br>Request ${n.tasks[i].request} · ${n.splits} Splits / request`:''}${d===p?msg('load.tasks', p.buckets[i].end-p.buckets[i].begin):''}`;a+=rect(x,top+105-h,Math.max(1,pw/170-1),Math.max(v?2:1,h),v?color:'var(--border)','none',1,`data-tip="${esc(tip)}" data-morph="${name}-${i}"`);});
      [0,40,80,120,169].forEach(i=>a+=tx(left+(i+.5)/170*pw,top+124,i,'diagram-small diagram-mono','text-anchor="middle"'));
      if(d===p)a+=tx(left+pw/2,top+145,'CTA ID','diagram-small diagram-mono','text-anchor="middle"');
    });
    svg('load-chart',a,W,378,msg('load.diagram'),animate);
    metrics('load-metrics',[[msg('load.nativeMax'),Math.max(...n.loads),'Tiles',`${n.splits} Splits / padded request`],[msg('load.bucketMax'),Math.max(...p.loads),'Tiles',`Budget = ${p.budget}`],[msg('load.batch'),p.total,'Tiles',msg('load.counts', p.tasks.length, p.active)]]);
    explanation('load-explanation','WORK DISTRIBUTION',loadPreset==='uniform'?msg('load.uniformStart')+Math.max(...n.loads)+msg('load.uniformEnd'):msg('load.skewNote', Math.max(...n.loads), p.budget));
  }
  $('#load-preset').addEventListener('change',e=>{loadPreset=e.target.value;renderLoad(true);});

  // ---------- Bucket step-through ----------
  let bucketState={preset:'teaching',step:2,selected:4};
  function bucketData(){return bucketState.preset==='teaching'?taskPlan([2304,768,512,256],8):taskPlan(presets.skew32,170);}
  function renderBucket(animate=false){
    const s=bucketState,p=bucketData(),small=s.preset==='teaching',id='bucket-diagram',W=960;let a='',H=340;
    steps('bucket-steps',[msg('bucket.requests'),'KV Tiles',msg('bucket.pack'),msg('bucket.run'),msg('bucket.partials')],s.step,i=>{s.step=i;renderBucket(true);});
    const sel=$('#bucket-selected');if(sel.options.length!==p.ctas){sel.innerHTML=Array.from({length:p.ctas},(_,i)=>`<option value="${i}">CTA ${i}</option>`).join('');s.selected=Math.min(s.selected,p.ctas-1);}sel.value=String(s.selected);
    const selected=p.buckets[s.selected],st=p.tasks.slice(selected.begin,selected.end);
    if(s.step<=1 && small){
      a+=tx(26,23,s.step===0?msg('bucket.cost'):msg('bucket.tileSize'),'diagram-title');
      p.counts.forEach((n,r)=>{const y=52+r*59;a+=tx(27,y+20,`R${r}`,'diagram-label diagram-mono');
        if(s.step===0){a+=rect(75,y,n*56-5,32,rc(r),'none',4,`data-morph="request-${r}"`)+tx(90+n*56,y+21,`${p.lengths[r].toLocaleString()} Tokens`,'diagram-label');}
        else{for(let t=0;t<n;t++)a+=`<g data-tip="${esc(`<strong>R${r} / Tile ${t}</strong>KV Tokens [${t*256}, ${(t+1)*256})`)}">${rect(75+t*56,y,49,32,rc(r),'none',4,`data-morph="tile-${r}-${t}"`)}${tx(99+t*56,y+21,`T${t}`,'','fill="#fff" font-size="11" font-family="var(--mono)" text-anchor="middle"')}</g>`;a+=tx(90+n*56,y+21,`${n} Tiles`,'diagram-label');}
      });a+=tx(28,314,msg('bucket.summary', p.total, p.ctas, p.budget),'diagram-caption');
    } else if(s.step<=1){
      H=360;a+=tx(26,25,s.step===0?msg('bucket.skew'):msg('bucket.tileAxis'),'diagram-title');
      if(s.step===0){p.counts.forEach((n,r)=>{const col=r<16?0:1,row=r%16,x=25+col*460,y=46+row*17;const w=n/511*300;a+=tx(x,y+8,`R${r}`,'diagram-small diagram-mono')+rect(x+35,y,Math.max(3,w),10,rc(r),'none',2)+tx(x+351,y+8,`${p.lengths[r]}`,'diagram-small diagram-mono');});}
      else {let offset=0;p.counts.forEach((n,r)=>{for(let t=0;t<n;t++){const global=offset+t,x=25+global/p.total*910;a+=rect(x,72,Math.max(.6,910/p.total-.15),62,rc(r),'none',0,`data-tip="${esc(`<strong>Global ${global}</strong>R${r} / Tile ${t}`)}"`);}offset+=n;});a+=tx(25,157,'0','diagram-small diagram-mono')+tx(935,157,`${p.total} Tiles`,'diagram-small diagram-mono','text-anchor="end"');a+=box(90,205,320,75,msg('bucket.budgetTitle'),msg('bucket.budgetCalc', p.total, p.budget),true);a+=box(510,205,320,75,msg('bucket.identity'),msg('bucket.boundary'),false);}
    } else if(s.step===2 && small){
      H=335;a+=tx(25,25,msg('bucket.example'),'diagram-title');
      p.buckets.forEach((b,i)=>{const x=24+i%4*234,y=46+Math.floor(i/4)*124;const selected=i===s.selected;const tasks=p.tasks.slice(b.begin,b.end);a+=rect(x,y,212,104,selected?'var(--accent-soft)':'var(--panel-2)',selected?'var(--accent)':'var(--border)',7,`data-cta="${i}" style="cursor:pointer"`)+tx(x+12,y+20,`Bucket ${i} / CTA${i}`,'diagram-label diagram-mono');let local=0;tasks.forEach(t=>{const count=t.end-t.begin,start=local;for(let j=t.begin;j<t.end;j++){const xx=x+12+local*92;a+=rect(xx,y+37,85,31,rc(t.request),'none',4,`data-morph="tile-${t.request}-${j}" data-cta="${i}" style="cursor:pointer"`)+tx(xx+42,y+57,`R${t.request} · T${j}`,'','fill="#fff" font-size="10" font-family="var(--mono)" text-anchor="middle"');local++;}a+=tx(x+12+(start+count/2)*92-4,y+86,`Task ${t.slot}`,'diagram-small diagram-mono','text-anchor="middle"');});});
      a+=tx(25,320,`CTA ${s.selected}：`+st.map(t=>`Task${t.slot} = R${t.request} [${t.begin},${t.end})`).join('  →  '),'diagram-caption');
    } else if(s.step===2){
      H=385;a+=tx(24,24,msg('bucket.select'),'diagram-title');
      p.buckets.forEach((b,i)=>{const x=24+i%17*54,y=45+Math.floor(i/17)*27,active=i===s.selected;a+=rect(x,y,48,22,'var(--panel-2)',active?'var(--accent)':'var(--border)',3,`stroke-width="${active?2:1}" data-cta="${i}" style="cursor:pointer"`);let off=0;const tasks=p.tasks.slice(b.begin,b.end);tasks.forEach(t=>{const w=(t.end-t.begin)/p.budget*46;a+=rect(x+1+off,y+1,w,20,rc(t.request),'none',2,`data-cta="${i}" style="cursor:pointer" data-tip="${esc(`<strong>CTA ${i} / Task${t.slot}</strong>R${t.request} · KV Tile [${t.begin},${t.end})`)}"`);off+=w;});});
      a+=tx(24,337,`CTA ${s.selected} · ${p.loads[s.selected]} Tiles · ${st.length} Tasks`,'diagram-title');a+=tx(24,361,st.map(t=>`Task${t.slot}: R${t.request} [${t.begin},${t.end})`).join('  →  ')||msg('bucket.empty'),'diagram-small diagram-mono');
    } else if(s.step===3){
      H=small?335:350;a+=tx(24,25,msg('bucket.ctaPath'),'diagram-title');
      if(small){p.buckets.forEach((b,i)=>{const y=48+i*29;const active=i===s.selected;a+=tx(25,y+15,`CTA${i}`,'diagram-label diagram-mono')+rect(90,y,392,22,'var(--panel-2)','none',3);let offset=0;p.tasks.slice(b.begin,b.end).forEach(t=>{const w=(t.end-t.begin)*180;a+=rect(90+offset,y,w-2,22,rc(t.request),active?'var(--accent)':'none',3)+tx(90+offset+w/2,y+15,`Task${t.slot} / R${t.request}`,'','fill="#fff" font-size="10" text-anchor="middle"');offset+=w;});});a+=box(537,67,383,83,msg('bucket.order', s.selected),st.map(t=>`Task${t.slot}`).join(' → ')||msg('bucket.empty'),true);a+=tx(554,184,msg('bucket.inside'),'diagram-title');[msg('bucket.readQ'),msg('bucket.readTiles'),msg('bucket.softmax'),msg('bucket.write')].forEach((t,i)=>a+=tx(554,213+i*25,t,'diagram-label'));}
      else {const max=Math.max(...p.loads,1),left=30,pw=895;p.loads.forEach((v,i)=>{const x=left+i/p.ctas*pw,h=v/max*150;a+=rect(x,210-h,pw/p.ctas-1,h,'var(--accent)','none',1,`data-cta="${i}" data-tip="${esc(`<strong>CTA ${i}</strong>${v} Tiles`)}"`);});a+=tx(30,236,msg('bucket.max'),'diagram-small diagram-mono');a+=box(30,263,890,66,msg('bucket.sequence', s.selected),st.map(t=>`Task${t.slot}: R${t.request} [${t.begin},${t.end})`).join(' → ')||msg('bucket.empty'),true);}
    } else {
      H=small?337:370;a+=tx(24,25,msg('bucket.mergeTitle'),'diagram-title');
      const show=small?[0,1,2,3]:[0,1,2,31];
      show.forEach((r,k)=>{const y=48+k*69,range=p.requests[r];a+=box(25,y,128,48,`R${r}`,msg('bucket.taskCount', range.count),false);a+=path(`M153 ${y+24} H207`,true,id);const total=range.count;const showparts=Math.min(total,7);for(let j=0;j<showparts;j++)a+=rect(212+j*49,y+7,42,34,rc(r),'none',4)+tx(233+j*49,y+29,j===showparts-1&&total>7?'…':`P${j}`,'','fill="#fff" font-family="var(--mono)" font-size="10" text-anchor="middle"');a+=path(`M565 ${y+24} H627`,true,id)+box(633,y,135,48,'Last CTA',msg('bucket.byRequest'),true)+path(`M768 ${y+24} H814`,true,id)+box(820,y,112,48,`O_R${r}`,'6 × 256',true);});if(!small)a+=tx(26,353,msg('bucket.other'),'diagram-small');
    }
    svg(id,a,W,H,msg('bucket.diagram'),animate);
    $('#bucket-diagram').onclick=e=>{const n=e.target.closest('[data-cta]');if(n){s.selected=+n.dataset.cta;renderBucket(true);}};
    metrics('bucket-stats',[[msg('bucket.total'),p.total,'Tiles',msg('bucket.reqCount', p.lengths.filter(x=>x>0).length)],[msg('bucket.budget'),p.budget,'Tiles',msg('bucket.active', p.active)],[msg('bucket.slots'),p.tasks.length,'',msg('bucket.count', p.ctas)]]);
    const cross=st.length>1;
    const desc=[
      msg('bucket.requestsNote'),
      msg('bucket.tilesNote'),
      msg('bucket.packNote', p.budget, s.selected, st.length, cross?msg('bucket.cross'):msg('bucket.same')),
      msg('bucket.runNote'),
      msg('bucket.mergeNote')
    ];explanation('bucket-explanation',`ASSIGN / COMPUTE · ${s.step+1} / 5`,desc[s.step]);
    $('#bucket-task-table').innerHTML=msg('bucket.table')+p.tasks.map(t=>`<tr><td>T${t.slot}</td><td>R${t.request}</td><td>[${t.begin}, ${t.end})</td><td>${t.end-t.begin}</td><td>${t.bucket}</td></tr>`).join('')+'</tbody></table>';
  }
  $('#bucket-preset').addEventListener('change',e=>{bucketState.preset=e.target.value;bucketState.selected=bucketState.preset==='teaching'?4:85;renderBucket(true);});
  $('#bucket-selected').addEventListener('change',e=>{bucketState.selected=+e.target.value;renderBucket(true);});

  // ---------- Last-CTA reduction ----------
  let mergeStep=5;
  function renderMerge(animate=false){const id='merge-diagram',order=[2,0,4,3,1],done=new Set(order.slice(0,mergeStep));steps('merge-steps',[msg('merge.start'),msg('merge.t2'),msg('merge.t0'),msg('merge.t4'),msg('merge.t3'),'T1 → Merge'],mergeStep,i=>{mergeStep=i;renderMerge(true);});let a='';
    a+=tx(35,24,msg('merge.tasks'),'diagram-title')+tx(272,24,'Global Scratch','diagram-title')+tx(627,24,'Request R0','diagram-title');
    for(let i=0;i<5;i++){const y=43+i*47,ok=done.has(i);a+=rect(26,y,148,33,ok?'var(--accent-soft)':'var(--panel-2)',ok?'var(--accent)':'var(--border)',5)+tx(100,y+21,`Task ${i} · ${ok?msg('merge.done'):msg('merge.running')}`,'diagram-label diagram-mono','text-anchor="middle"');a+=path(`M174 ${y+17} H253`,ok,id);a+=rect(260,y,202,33,ok?'var(--blue-soft)':'var(--panel-2)',ok?'var(--blue)':'var(--border)',5)+tx(361,y+21,ok?`O${i} · m${i} · l${i}`:msg('merge.wait'),'diagram-label diagram-mono','text-anchor="middle"');if(ok)a+=path(`M462 ${y+17} H540 V145 H600`,mergeStep===5,id);}
    a+=box(603,115,160,62,mergeStep===5?'CTA of Task 1':msg('merge.counter'),mergeStep===5?msg('merge.winner'):`${mergeStep} / 5`,mergeStep===5);a+=path('M763 146 H798',mergeStep===5,id);a+=box(803,115,132,62,'O_R0',mergeStep===5?msg('merge.output'):msg('merge.pending'),mergeStep===5);
    a+=tx(608,211,mergeStep===5?msg('merge.reweight'):msg('merge.order'),'diagram-small');a+=tx(608,239,mergeStep===5?msg('merge.reset'):msg('merge.publish'),'diagram-small');
    svg(id,a,960,300,msg('merge.diagram'),animate);
    const texts=[msg('merge.startNote'),msg('merge.t2Note'),msg('merge.t0Note'),msg('merge.t4Note'),msg('merge.t3Note'),msg('merge.t1Note')];
    explanation('merge-explanation',`COMPLETE ${mergeStep} / 5`,texts[mergeStep]);
  }

  // ---------- Three consumer-format fusions ----------
  const fusionStates={norm:{mode:'after',step:2},front:{mode:'after',step:2},tail:{mode:'after',step:2}};
  function renderFusion(kind,animate=false){
    const holder=$(`#${kind}-widget`),state=fusionStates[kind],after=state.mode==='after',step=state.step;
    const stageNames=kind==='norm'?[msg('fusion.readInput'),msg('fusion.normQuant'),msg('fusion.gemm')]:kind==='front'?[msg('fusion.readProj'),msg('fusion.qkv'),msg('fusion.store')]:[msg('fusion.final'),msg('fusion.gateQuant'),msg('fusion.project')];
    steps($('.fusion-steps',holder),stageNames,step,i=>{state.step=i;renderFusion(kind,true);});
    $$('.fusion-mode button',holder).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode)));
    const container=$('.fusion-diagram',holder);container.id=kind+'-flow';const id=container.id;let a='';const inputActive=step===0,processActive=step===1,outputActive=step===2;
    const mem=(x,y,w,h,title,sub,active)=>`<g>${rect(x,y,w,h,'color-mix(in srgb,var(--orange) 12%,var(--panel))','var(--orange)',6,active?'stroke-width="2"':'')}${tx(x+w/2,y+25,title,'diagram-title','text-anchor="middle"')}${tx(x+w/2,y+45,sub,'diagram-small','text-anchor="middle"')}</g>`;
    if(kind==='norm'){
      a+=box(20,89,143,64,'x + Residual','BF16 / H5120',inputActive,'data-morph="norm-in"');
      if(after){a+=box(259,75,352,91,'Residual + RMSNorm + NVFP4',msg('norm.reducePack'),processActive,'data-morph="norm-process"');a+=path('M163 120 H259',step>=1,id)+path('M611 120 H767',step>=2,id);a+=tx(686,107,'FP4 + Scale','diagram-small diagram-mono','text-anchor="middle"');a+=path('M435 166 V219',step>=2,id);a+=box(345,219,180,57,'BF16 Residual',msg('norm.residual'),outputActive,'data-morph="norm-residual"');}
      else {a+=box(202,89,178,64,'Add + Gemma RMSNorm',msg('norm.fp32'),processActive,'data-morph="norm-process"');a+=mem(414,89,134,64,'BF16 y',msg('fusion.roundtrip'),step>=1);a+=box(581,89,146,64,'NVFP4 Pack',msg('norm.scale'),processActive);a+=path('M163 120 H202',step>=1,id)+path('M380 120 H414',step>=1,id)+path('M548 120 H581',step>=2,id)+path('M727 120 H767',step>=2,id);a+=path('M290 153 V247 H345',step>=1,id);a+=box(345,219,180,57,'BF16 Residual',msg('norm.residual'),outputActive,'data-morph="norm-residual"');}
      a+=box(767,89,173,64,'NVFP4 GEMM',msg('norm.layout'),outputActive,'data-morph="norm-out"');a+=tx(25,312,after?msg('norm.registers'):msg('norm.spill'),'diagram-caption');
    }else if(kind==='front'){
      a+=box(20,120,144,66,'QKV Projection',msg('front.gate'),inputActive,'data-morph="front-in"');
      if(after){a+=box(259,78,354,128,'Fused Attention Front','Q / K: Norm → RoPE',processActive,'data-morph="front-process"');a+=tx(436,150,msg('front.value'),'diagram-label','text-anchor="middle"')+tx(436,177,msg('front.cache'),'diagram-small','text-anchor="middle"');a+=path('M164 152 H259',step>=1,id);a+=path('M613 110 H687 V82 H770',step>=2,id)+path('M613 178 H687 V243 H770',step>=2,id);}
      else {a+=box(211,62,190,65,'QK Norm + RoPE',msg('front.heads'),processActive,'data-morph="front-process"');a+=path('M164 142 H185 V94 H211',step>=1,id);a+=path('M401 94 H690 V82 H770',step>=2,id);a+=mem(436,165,135,64,'BF16 K',msg('front.key'),step>=1);a+=path('M304 127 V196 H436',step>=1,id);a+=box(613,210,133,64,'Quant + Append',msg('front.append'),processActive);a+=path('M571 196 H593 V242 H613',step>=2,id);a+=path('M90 186 V290 H588 V254 H613',step>=1,id)+tx(290,282,msg('front.valuePath'),'diagram-small','text-anchor="middle"');a+=path('M746 243 H770',step>=2,id);}
      a+=box(770,50,169,64,'BF16 Q + Gate',msg('front.query'),outputActive,'data-morph="front-q"');
      a+=rect(770,205,169,83,outputActive?'var(--accent-soft)':'var(--panel-2)',outputActive?'var(--accent)':'var(--border)',7,'data-morph="front-kv"')+tx(854,229,'Paged KV Cache','diagram-title','text-anchor="middle"');
      ['K / V','Scale','Page32'].forEach((v,i)=>{a+=rect(779+i*52,242,47,25,i===0?'var(--blue-soft)':'var(--panel)','var(--border)',3)+tx(802+i*52,258,v,'diagram-small diagram-mono','text-anchor="middle"');});a+=tx(25,329,msg('front.caption'),'diagram-caption');
    }else {
      a+=box(20,113,181,68,'Attention / Merge',msg('tail.stats'),inputActive,'data-morph="tail-in"');
      if(after){a+=box(277,96,342,105,'Final Store + Gate + FP8',msg('tail.epilogue'),processActive,'data-morph="tail-process"');a+=path('M201 148 H277',step>=1,id)+path('M619 148 H766',step>=2,id)+tx(689,133,'FP8','diagram-small diagram-mono','text-anchor="middle"');a+=box(355,14,190,48,'Gate + Static Scale','',inputActive,'data-morph="tail-gate"')+path('M449 62 V96',step>=1,id);}
      else {a+=mem(245,113,144,68,'BF16 O',msg('fusion.roundtrip'),step>=1);a+=box(454,98,235,101,'Gate + FP8',msg('tail.compiled'),processActive,'data-morph="tail-process"');a+=path('M201 148 H245',step>=1,id)+path('M389 148 H454',step>=1,id)+path('M689 148 H766',step>=2,id);a+=box(478,14,190,48,'Gate + Static Scale','',inputActive,'data-morph="tail-gate"')+path('M573 62 V98',step>=1,id);}
      a+=box(766,113,174,68,'Output Projection',msg('tail.layout'),outputActive,'data-morph="tail-out"');a+=rect(211,244,542,39,'var(--panel-2)','var(--border)',5)+tx(482,269,msg('tail.once'),'diagram-label','text-anchor="middle"');a+=tx(25,322,msg('tail.caption'),'diagram-caption');
    }
    svg(container,a,960,kind==='front'?350:340,msg('fusion.diagram', kind, after?msg('fusion.fused'):msg('fusion.native')),animate);
    const texts={
      norm:[
        msg('norm.inputNote'),
        after?msg('norm.fusedNote'):msg('norm.nativeNote'),
        after?msg('norm.outNote'):msg('norm.unfusedOut')
      ],front:[
        msg('front.inputNote'),
        after?msg('front.fusedNote'):msg('front.nativeNote'),
        msg('front.outNote')
      ],tail:[
        msg('tail.inputNote'),
        after?msg('tail.fusedNote'):msg('tail.nativeNote'),
        after?msg('tail.outNote'):msg('tail.unfusedOut')
      ]
    };
    $('.fusion-explanation',holder).innerHTML=`<span class="ex-label">${after?'FUSED':'NATIVE BOUNDARY'} · ${step+1} / 3</span>${texts[kind][step]}`;
  }
  $$('.fusion-widget').forEach(h=>{const kind=h.dataset.kind;$('.fusion-mode',h).addEventListener('click',e=>{const b=e.target.closest('button[data-mode]');if(b){fusionStates[kind].mode=b.dataset.mode;renderFusion(kind,true);}});});

  // ---------- Measured benchmark charts ----------
  const data=JSON.parse($('#benchmark-data').textContent);let prefillMetric='relative';
  function barline(name,value,width,baseline){return `<div class="barline"><span class="barname">${name}</span><div class="bartrack"><div class="barfill ${baseline?'baseline':''}" style="width:${Math.min(100,width)}%"></div></div><span class="barvalue">${value}</span></div>`;}
  function renderPrefill(){const max=Math.max(...data.prefill.map(r=>+r.cpp_ms));$('#prefill-chart').innerHTML=data.prefill.map(r=>{const c=+r.cpp_ms,v=+r.vpp_ms,rel=prefillMetric==='relative',scale=rel?c:max;return msg('chart.prefillRow', r.input_tokens==='131071'?'128K':(+r.input_tokens/1024)+'K', fmt((c-v)/c*100), barline('CPP',rel?'100.0%':fmt(c/1000,3)+' s',c/scale*100,true), barline('VPP',rel?fmt(v/c*100,1)+'%':fmt(v/1000,3)+' s',v/scale*100,false));}).join('');$$('#prefill-metric button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===prefillMetric)));}
  $('#prefill-metric').addEventListener('click',e=>{const b=e.target.closest('button[data-value]');if(b){prefillMetric=b.dataset.value;renderPrefill();}});
  function renderDecode(){const max=Math.max(...data.decode.map(r=>+r.native_ms));const labels=['B1 · 512',msg('chart.uniform'),'B9 · 128K + 8 × 4K','B32 · 128K + 31 × 4K'];$('#decode-chart').innerHTML=data.decode.map((r,i)=>{const n=+r.native_ms,b=+r.bucket_ms,g=(n-b)/n*100;return `<div class="bench-row"><div class="bench-row-title"><span>${labels[i]}</span><span class="bench-gain ${g<0?'neutral':''}">${g>=0?'−':'+'}${fmt(Math.abs(g))}%</span></div>${barline('Native',fmt(n,2)+' ms',n/max*100,true)}${barline('Bucket',fmt(b,2)+' ms',b/max*100,false)}</div>`;}).join('');}

  renderPipe();renderRank();renderDMA();renderLoad();renderBucket();renderMerge();['norm','front','tail'].forEach(k=>renderFusion(k));renderPrefill();renderDecode();scrollUI();
  // Deterministic model functions are exposed for reproducibility and automated checks.
  window.BlogModel=Object.freeze({schedule,taskPlan,nativePlan,benchmarkData:data});
})();
