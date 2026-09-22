const $=s=>document.querySelector(s);let timer=null;
const api=async(path,method='GET')=>(await fetch(path,{method})).json();
function render(s){
  const obs=new Set(s.obstacles.map(c=>c.join(',')));$('#grid').innerHTML='';
  for(let y=0;y<s.size[1];y++)for(let x=0;x<s.size[0];x++){
    const d=document.createElement('div');d.className='cell';
    if(obs.has(`${x},${y}`))d.classList.add('wall');
    if(x===s.goal[0]&&y===s.goal[1])d.classList.add('goal');
    if(x===s.agent[0]&&y===s.agent[1])d.classList.add('agent','pulse');
    $('#grid').appendChild(d);
  }
  const m=s.metrics;$('#metrics').innerHTML=[['Tick',s.tick],['Spikes',m.snn_spikes],['Published',m.published_events],['Local energy',m.local_energy],['Global energy',m.global_energy]].map(([k,v])=>`<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#status').textContent=s.done?'GOAL REACHED':`POS ${s.agent.join(':')}`;$('#sparsity').textContent=`${Math.round(m.event_sparsity*100)}% FILTERED`;
  if(s.decision){
    $('#winner').textContent=s.decision.winner;$('#decision').className='decision';$('#decision').innerHTML=`<strong>${s.decision.action}</strong><small>${s.decision.reason} · score ${s.decision.score}</small>`;
    $('#proposals').innerHTML=s.decision.proposals.map(p=>`<div class="proposal"><span class="arm">${p.arm}</span><div>${p.reason}<div class="bar"><i style="width:${Math.round(p.confidence*100)}%"></i></div></div><b>${p.action}</b></div>`).join('');
  }else{$('#winner').textContent='IDLE';$('#decision').className='decision empty';$('#decision').textContent='Waiting for proposals.';$('#proposals').innerHTML='';}
  const ev=s.transition?.salient_events||s.attention||[];$('#events').innerHTML=ev.length?ev.map(e=>`<div class="event"><b>${e.source}/${e.kind}</b><span>${e.salience.toFixed(2)}</span><small>${JSON.stringify(e.payload)}</small></div>`).join(''):'<div class="empty">No salient events this tick.</div>';
  if(s.done&&timer)toggleRun();
}
async function step(){render(await api('/api/step','POST'))}async function reset(){render(await api('/api/reset','POST'))}
function toggleRun(){if(timer){clearInterval(timer);timer=null;$('#run').textContent='Run';$('#run').classList.remove('active')}else{timer=setInterval(step,450);$('#run').textContent='Pause';$('#run').classList.add('active')}}
$('#step').onclick=step;$('#run').onclick=toggleRun;$('#reset').onclick=reset;api('/api/state').then(render);

