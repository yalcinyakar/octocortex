const $ = selector => document.querySelector(selector);
const MOVES = {UP:[0,-1], DOWN:[0,1], LEFT:[-1,0], RIGHT:[1,0]};
const key = cell => cell.join(',');

class LIFNeuron {
  constructor(threshold=1, leak=.6){this.threshold=threshold;this.leak=leak;this.potential=0;this.spikes=0}
  step(current){this.potential=this.potential*this.leak+current;if(this.potential<this.threshold)return false;this.potential=0;this.spikes++;return true}
}

class SparseEventBus {
  constructor(threshold=.5){this.threshold=threshold;this.pending=[];this.observations=0;this.published=0}
  observe(event){this.observations++;if(event.salience<this.threshold)return false;this.pending.push(event);this.published++;return true}
  drain(){const events=this.pending;this.pending=[];return events}
  get sparsity(){return this.observations?1-this.published/this.observations:1}
}

class OctoSimulation {
  constructor(){this.reset()}
  reset(){
    this.tick=0;this.agent=[0,0];this.goal=[11,7];
    this.obstacles=[[2,0],[2,1],[2,2],[4,2],[5,2],[6,2],[6,3],[6,4],[8,4],[9,4],[10,4],[8,6]];
    this.obstacleSet=new Set(this.obstacles.map(key));this.collisions=0;this.done=false;this.lastDecision=null;
    this.bus=new SparseEventBus();this.neuron=new LIFNeuron();this.memory=new Map();
    this.localEnergy=0;this.globalEnergy=0;this.attention=[];return this.snapshot();
  }
  event(source,kind,salience,payload={}){return {source,kind,salience,payload,tick:this.tick}}
  proposal(arm,action,confidence,values={}){return {arm,action,confidence,expected_reward:0,risk:0,urgency:0,energy_cost:0,reason:'',...values}}
  nextCells(){return Object.fromEntries(Object.entries(MOVES).map(([action,[dx,dy]])=>[action,[this.agent[0]+dx,this.agent[1]+dy]]))}
  navigation(){
    const cells=this.nextCells();
    const valid=Object.entries(cells).filter(([,cell])=>cell[0]>=0&&cell[0]<12&&cell[1]>=0&&cell[1]<8&&!this.obstacleSet.has(key(cell)));
    this.localEnergy+=.035+.008*valid.length;
    if(!valid.length)return {events:[this.event('navigation','trapped',1)],proposals:[this.proposal('navigation','WAIT',1,{risk:1,urgency:1,reason:'no valid move'})],cells};
    const distance=cell=>Math.abs(cell[0]-this.goal[0])+Math.abs(cell[1]-this.goal[1]);
    valid.sort((a,b)=>distance(a[1])-distance(b[1])||a[0].localeCompare(b[0]));
    const [action,cell]=valid[0], nextDistance=distance(cell), currentDistance=distance(this.agent), progress=currentDistance-nextDistance;
    return {events:[this.event('navigation','route_proposal',.42,{action,target:cell,distance:nextDistance})],proposals:[this.proposal('navigation',action,progress>0?.78:.58,{expected_reward:progress>0?.82:.25,risk:.05,urgency:.22,energy_cost:.08,reason:`${action.toLowerCase()} reduces goal distance to ${nextDistance}`})],cells};
  }
  vision(){
    const adjacent=Object.values(this.nextCells()).filter(cell=>this.obstacleSet.has(key(cell))).length;
    const danger=adjacent/2, spike=this.neuron.step(danger);this.localEnergy+=.055+.035*danger;
    const events=[this.event('vision',spike?'danger_spike':'visual_scan',Math.min(1,danger),{adjacent_obstacles:adjacent,spike,potential:+this.neuron.potential.toFixed(3)})];
    const proposals=spike&&adjacent?[this.proposal('vision','WAIT',.86,{risk:Math.min(1,.72+danger/3),urgency:.86,energy_cost:.05,reason:'local danger neuron spiked'})]:[];
    return {events,proposals,snn:{danger,spike,potential:this.neuron.potential}};
  }
  recall(cells){
    this.localEnergy+=.025;
    const risky=Object.entries(cells).map(([action,cell])=>[action,cell,this.memory.get(key(cell))||0]).filter(item=>item[2]);
    if(!risky.length)return {events:[this.event('memory','memory_quiet',.15)],proposals:[]};
    risky.sort((a,b)=>b[2]-a[2]);const [action,cell,count]=risky[0];
    return {events:[this.event('memory','collision_recall',Math.min(1,.55+.12*count),{cell,count,avoid_action:action})],proposals:[this.proposal('memory','WAIT',Math.min(.95,.55+.1*count),{risk:Math.min(.98,.6+.1*count),urgency:.55,energy_cost:.02,reason:`recalled ${count} collision(s) near ${cell}`})]};
  }
  score(p){return .34*p.confidence+.24*p.expected_reward+.28*p.risk+.22*p.urgency-.08*p.energy_cost}
  arbitrate(proposals){
    if(!proposals.length)return {action:'WAIT',winner:'workspace',score:0,reason:'No arm proposed an action',proposals};
    const vetoes=proposals.filter(p=>p.risk>=.9&&p.urgency>=.75), candidates=vetoes.length?vetoes:proposals;
    const winner=candidates.reduce((best,p)=>this.score(p)>this.score(best)?p:best);
    return {action:winner.action,winner:winner.arm,score:+this.score(winner).toFixed(4),reason:`${vetoes.length?'Safety veto':'Highest arbitration score'}: ${winner.reason}`,proposals};
  }
  step(){
    if(this.done)return this.snapshot();this.tick++;
    const nav=this.navigation(), visual=this.vision(), memory=this.recall(nav.cells);
    [...nav.events,...visual.events,...memory.events].forEach(event=>this.bus.observe(event));
    const salient=this.bus.drain();this.attention=[...salient].sort((a,b)=>b.salience-a.salience).slice(0,5);
    const proposals=[...nav.proposals,...visual.proposals,...memory.proposals];this.globalEnergy+=.1+.16*salient.length+.04*proposals.length;
    this.lastDecision=this.arbitrate(proposals);let collision=false;
    if(MOVES[this.lastDecision.action]){
      const [dx,dy]=MOVES[this.lastDecision.action], target=[this.agent[0]+dx,this.agent[1]+dy];
      if(target[0]<0||target[0]>=12||target[1]<0||target[1]>=8||this.obstacleSet.has(key(target))){collision=true;this.collisions++;this.memory.set(key(target),(this.memory.get(key(target))||0)+1);this.bus.observe(this.event('body','collision',1,{cell:target}))}else this.agent=target;
    }
    if(key(this.agent)===key(this.goal)){this.done=true;this.bus.observe(this.event('body','goal_reached',1))}
    return {...this.snapshot(),transition:{collision,snn:visual.snn,salient_events:salient}};
  }
  snapshot(){return {size:[12,8],agent:[...this.agent],goal:[...this.goal],obstacles:this.obstacles.map(cell=>[...cell]),tick:this.tick,done:this.done,collisions:this.collisions,decision:this.lastDecision,attention:this.attention,metrics:{observations:this.bus.observations,published_events:this.bus.published,event_sparsity:+this.bus.sparsity.toFixed(3),snn_spikes:this.neuron.spikes,local_energy:+this.localEnergy.toFixed(3),global_energy:+this.globalEnergy.toFixed(3)}}}
}

const simulation=new OctoSimulation();let timer=null;
function render(s){
  const obs=new Set(s.obstacles.map(key));$('#grid').innerHTML='';
  for(let y=0;y<s.size[1];y++)for(let x=0;x<s.size[0];x++){
    const d=document.createElement('div');d.className='cell';
    if(obs.has(`${x},${y}`))d.classList.add('wall');if(x===s.goal[0]&&y===s.goal[1])d.classList.add('goal');if(x===s.agent[0]&&y===s.agent[1])d.classList.add('agent','pulse');$('#grid').appendChild(d);
  }
  const m=s.metrics;$('#metrics').innerHTML=[['Tick',s.tick],['Spikes',m.snn_spikes],['Published',m.published_events],['Local energy',m.local_energy],['Global energy',m.global_energy]].map(([k,v])=>`<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#status').textContent=s.done?'GOAL REACHED':`POS ${s.agent.join(':')}`;$('#sparsity').textContent=`${Math.round(m.event_sparsity*100)}% FILTERED`;
  if(s.decision){$('#winner').textContent=s.decision.winner;$('#decision').className='decision';$('#decision').innerHTML=`<strong>${s.decision.action}</strong><small>${s.decision.reason} · score ${s.decision.score}</small>`;$('#proposals').innerHTML=s.decision.proposals.map(p=>`<div class="proposal"><span class="arm">${p.arm}</span><div>${p.reason}<div class="bar"><i style="width:${Math.round(p.confidence*100)}%"></i></div></div><b>${p.action}</b></div>`).join('')}else{$('#winner').textContent='IDLE';$('#decision').className='decision empty';$('#decision').textContent='Waiting for proposals.';$('#proposals').innerHTML=''}
  const ev=s.transition?.salient_events||s.attention||[];$('#events').innerHTML=ev.length?ev.map(e=>`<div class="event"><b>${e.source}/${e.kind}</b><span>${e.salience.toFixed(2)}</span><small>${JSON.stringify(e.payload)}</small></div>`).join(''):'<div class="empty">No salient events this tick.</div>';if(s.done&&timer)toggleRun();
}
function step(){render(simulation.step())}function reset(){render(simulation.reset())}
function toggleRun(){if(timer){clearInterval(timer);timer=null;$('#run').textContent='Run';$('#run').classList.remove('active')}else{timer=setInterval(step,450);$('#run').textContent='Pause';$('#run').classList.add('active')}}
if(typeof module!=='undefined')module.exports={LIFNeuron,SparseEventBus,OctoSimulation};
if(typeof document!=='undefined'){$('#step').onclick=step;$('#run').onclick=toggleRun;$('#reset').onclick=reset;render(simulation.snapshot())}
