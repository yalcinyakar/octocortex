const $ = selector => document.querySelector(selector);
const UNIT={SYSTEM:0,PERCEPTION:1,MEMORY:2,PREDICTION:3,PLANNING:4,SAFETY:5,EXECUTION:6,WORKSPACE:7};
const CONCEPT={VISUAL_SCAN:1,DANGER_SPIKE:2,ROUTE_PROPOSAL:3,TRAPPED:4,MEMORY_QUIET:5,COLLISION_RECALL:6,COLLISION:7,GOAL_REACHED:8,WORLD_MODEL_UPDATE:9};
const ACTION={WAIT:0,UP:1,DOWN:2,LEFT:3,RIGHT:4};
const REASON={NONE:0,GOAL_PROGRESS:1,DANGER_SPIKE:2,COLLISION_MEMORY:3,NO_VALID_MOVE:4};
const UNIT_NAME=['system','perception','memory','prediction','planning','safety','execution','workspace'];
const CONCEPT_NAME=['','visual_scan','danger_spike','route_proposal','trapped','memory_quiet','collision_recall','collision','goal_reached','world_model_update'];
const ACTION_NAME=['WAIT','UP','DOWN','LEFT','RIGHT'];
const MOVES={[ACTION.UP]:[0,-1],[ACTION.DOWN]:[0,1],[ACTION.LEFT]:[-1,0],[ACTION.RIGHT]:[1,0]};
const BENCHMARK_RESULTS=[
  ['clean','centralized',1,18,0,11.160],['clean','distributed',1,18,0,12.600],['clean','octocortex',1,18,0,5.176],
  ['sensor noise','centralized',1,22.38,.06,13.876],['sensor noise','distributed',1,22.38,.06,15.666],['sensor noise','octocortex',.99,22.30,.01,8.416],
  ['component failure','centralized',.52,18,0,29.611],['component failure','distributed',1,18,0,10.595],['component failure','octocortex',1,18,0,4.822],
  ['combined','centralized',.52,22.54,.05,31.074],['combined','distributed',1,22.38,.06,13.199],['combined','octocortex',.98,22.14,.03,8.074],
];
const GENERALIZATION_RESULTS=[['memory',100,.98,18.10,.02,.99],['perception',100,.98,18.20,.02,.99]];
const WORLD_MODEL_RESULTS=[['local only',100,0,null,0,0],['octocortex world model',100,1,27.4,0,.636]];
const DYNAMIC_RESULTS=[['persistent static belief',100,.58,18.83,0],['adaptive dynamic belief',100,.96,17.96,27.72]];
const BACKUP_WEIGHTS={
  [UNIT.MEMORY]:{[ACTION.UP]:[-.00475917,.94922877,.45330814,-.22344666,1.26311756],[ACTION.DOWN]:[.00760531,.72212238,.50200843,.00926497,1.32305712],[ACTION.LEFT]:[-.22875162,.72646324,.34690525,-.54242422,1.60620198],[ACTION.RIGHT]:[.2259934,.48892679,.20003376,-.23381379,1.20483094]},
  [UNIT.PERCEPTION]:{[ACTION.UP]:[.10154207,1.08784897,.61909066,-.24149871,1.44686587],[ACTION.DOWN]:[.13171805,.83812757,.64659756,-.01923372,1.55729555],[ACTION.LEFT]:[-.36510501,.85409155,.40854602,-.64978733,1.91736273],[ACTION.RIGHT]:[.10704752,.37403501,.32614213,-.36496567,1.66382972]},
};
const CAPABILITY={PLAN_ROUTE:1};
const key = cell => cell.join(',');
const reasonText=(code,value=0)=>code===REASON.GOAL_PROGRESS?`move reduces goal distance to ${value}`:code===REASON.DANGER_SPIKE?'local danger neuron spiked':code===REASON.COLLISION_MEMORY?`recalled ${value} collision(s)`:code===REASON.NO_VALID_MOVE?'no valid move':'no semantic reason code';
const packetTrace=p=>({source:UNIT_NAME[p.source],kind:CONCEPT_NAME[p.concept],salience:p.salience,payload:{concept_id:p.concept,semantic_code:p.semantic_code,quantization_error:p.quantization_error,confidence:p.confidence,uncertainty:p.uncertainty,urgency:p.urgency,risk:p.risk,state_delta:p.state_delta,latent:p.latent},tick:p.tick});
const proposalTrace=p=>({...p,arm:UNIT_NAME[p.arm],action:ACTION_NAME[p.action],reason:reasonText(p.reason_code,p.reason_value)});
const decisionTrace=d=>({action:ACTION_NAME[d.action],winner:UNIT_NAME[d.winner],score:d.score,reason:`${d.safety_veto?'Safety veto':'Highest arbitration score'}: ${reasonText(d.reason_code,d.reason_value)}`,reason_code:d.reason_code,proposals:d.proposals.map(proposalTrace)});
const shortLatent=latent=>latent.map(value=>Number(value).toFixed(3));

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

class OnlineVectorQuantizer {
  constructor(dimension=4,codebookSize=8,learningRate=.08,seed=29){
    this.dimension=dimension;this.codebookSize=codebookSize;this.learningRate=learningRate;this.steps=0;this.lastError=0;this.usage=Array(codebookSize).fill(0);
    let state=seed>>>0;const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
    this.codebook=Array.from({length:codebookSize},()=>Array.from({length:dimension},()=>random()-.5));
  }
  quantize(values,learn=true){
    if(values.length!==this.dimension)throw new Error(`expected ${this.dimension} latent values, got ${values.length}`);
    const distances=this.codebook.map(centroid=>centroid.reduce((sum,value,index)=>sum+(values[index]-value)**2,0));
    const code=distances.reduce((best,value,index)=>value<distances[best]?index:best,0), error=Math.sqrt(distances[code]/this.dimension);
    if(learn){this.codebook[code]=this.codebook[code].map((value,index)=>value+this.learningRate*(values[index]-value));this.usage[code]++;this.steps++;this.lastError=error}
    return {code:code+1,vector:[...this.codebook[code]],error};
  }
  get codesUsed(){return this.usage.filter(Boolean).length}
}

class CapabilityRouter {
  constructor(){this.routes={[CAPABILITY.PLAN_ROUTE]:[UNIT.PLANNING,UNIT.MEMORY,UNIT.PERCEPTION]};this.assignments=new Map();this.handoffs=0}
  resolve(capability,available){const chain=this.routes[capability],owner=chain.find(unit=>available.has(unit));if(owner===undefined)return null;const previous=this.assignments.has(capability)?this.assignments.get(capability):chain[0];if(owner!==previous)this.handoffs++;this.assignments.set(capability,owner);return {capability,owner,primary:chain[0],delegated:owner!==chain[0]}}
}

class DistilledBackupPlanner {
  constructor(owner){this.owner=owner;this.weights=Object.fromEntries(Object.entries(BACKUP_WEIGHTS[owner]).map(([action,values])=>[action,[...values]]));this.trainingAccuracy=owner===UNIT.MEMORY?.9996:1;this.energy=0}
  cells(agent){return Object.fromEntries(Object.entries(MOVES).map(([action,[dx,dy]])=>[Number(action),[agent[0]+dx,agent[1]+dy]]))}
  distance(cell,goal){return Math.abs(cell[0]-goal[0])+Math.abs(cell[1]-goal[1])}
  features(state,action){const cell=this.cells(state.agent)[action],valid=cell[0]>=0&&cell[0]<state.size[0]&&cell[1]>=0&&cell[1]<state.size[1]&&!state.obstacles.has(key(cell)),progress=(this.distance(state.agent,state.goal)-this.distance(cell,state.goal))/(state.size[0]+state.size[1]-2),clearance=Object.values(this.cells(cell)).filter(next=>next[0]>=0&&next[0]<state.size[0]&&next[1]>=0&&next[1]<state.size[1]&&!state.obstacles.has(key(next))).length/4,[dx,dy]=MOVES[action],alignment=(dx*Math.sign(state.goal[0]-state.agent[0])+dy*Math.sign(state.goal[1]-state.agent[1]))/2;return [1,valid?1:-1,progress,clearance,alignment]}
  score(action,features){return this.weights[action].reduce((sum,weight,index)=>sum+weight*features[index],0)}
  choose(state){return Object.keys(MOVES).map(Number).reduce((best,action)=>this.score(action,this.features(state,action))>this.score(best,this.features(state,best))?action:best,ACTION.UP)}
  teacher(state){const valid=Object.entries(this.cells(state.agent)).filter(([,cell])=>cell[0]>=0&&cell[0]<state.size[0]&&cell[1]>=0&&cell[1]<state.size[1]&&!state.obstacles.has(key(cell)));if(!valid.length)return ACTION.WAIT;valid.sort((a,b)=>this.distance(a[1],state.goal)-this.distance(b[1],state.goal)||Number(a[0])-Number(b[0]));return Number(valid[0][0])}
}

class SemanticAdapter {
  constructor(inputDim=8,latentDim=4,activeLatents=2,learningRate=.04,seed=17){
    this.inputDim=inputDim;this.latentDim=latentDim;this.activeLatents=activeLatents;this.learningRate=learningRate;this.steps=0;this.totalLoss=0;this.lastLoss=0;
    let state=seed>>>0;const random=()=>((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);const scale=1/Math.sqrt(inputDim);
    this.encoder=Array.from({length:latentDim},()=>Array.from({length:inputDim},()=>((random()*2-1)*scale)));
    this.decoder=Array.from({length:inputDim},()=>Array.from({length:latentDim},()=>((random()*2-1)*scale)));
    this.encoderBias=Array(latentDim).fill(0);this.decoderBias=Array(inputDim).fill(0);
    this.quantizer=new OnlineVectorQuantizer(latentDim);
  }
  semanticVector(source,concept,features=[]){const values=[concept/8,source/7,...features.slice(0,this.inputDim-2).map(value=>Math.max(-1,Math.min(1,Number(value))))];while(values.length<this.inputDim)values.push(0);return values}
  encodeAndLearn(vector){
    if(vector.length!==this.inputDim)throw new Error(`expected ${this.inputDim} adapter inputs, got ${vector.length}`);
    const values=vector.map(value=>Math.max(-1,Math.min(1,Number(value))));
    const hidden=this.encoder.map((row,index)=>Math.tanh(row.reduce((sum,weight,i)=>sum+weight*values[i],this.encoderBias[index])));
    const active=[...hidden.keys()].sort((a,b)=>Math.abs(hidden[b])-Math.abs(hidden[a])).slice(0,this.activeLatents);const latent=hidden.map((value,index)=>active.includes(index)?value:0);
    const reconstruction=this.decoder.map((row,index)=>row.reduce((sum,weight,i)=>sum+weight*latent[i],this.decoderBias[index]));const errors=reconstruction.map((value,index)=>value-values[index]);const loss=errors.reduce((sum,error)=>sum+error*error,0)/this.inputDim;
    const gradient=hidden.map((value,latentIndex)=>latent[latentIndex]?this.decoder.reduce((sum,row,outIndex)=>sum+row[latentIndex]*errors[outIndex],0)*(1-value*value):0);const rate=this.learningRate;
    this.decoder.forEach((row,outIndex)=>{row.forEach((_,latentIndex)=>row[latentIndex]-=rate*errors[outIndex]*latent[latentIndex]);this.decoderBias[outIndex]-=rate*errors[outIndex]});
    this.encoder.forEach((row,latentIndex)=>{row.forEach((_,inputIndex)=>row[inputIndex]-=rate*gradient[latentIndex]*values[inputIndex]);this.encoderBias[latentIndex]-=rate*gradient[latentIndex]});
    const quantized=this.quantizer.quantize(latent);this.steps++;this.lastLoss=loss;this.totalLoss+=loss;return {latent,quantizedLatent:quantized.vector,semanticCode:quantized.code,quantizationError:quantized.error,reconstruction,loss};
  }
  encodeSemantic(source,concept,features=[]){return this.encodeAndLearn(this.semanticVector(source,concept,features))}
  get meanLoss(){return this.steps?this.totalLoss/this.steps:0}
}

class OctoSimulation {
  constructor(){this.disabledUnits=new Set();this.reset()}
  reset(){
    this.tick=0;this.agent=[0,0];this.goal=[11,0];
    this.obstacles=[[5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6]];
    this.obstacleSet=new Set(this.obstacles.map(key));this.knownObstacles=new Set();this.knownFree=new Set();this.visibleSet=new Set();this.visited=new Set([key(this.agent)]);this.collisions=0;this.done=false;this.lastDecision=null;
    this.bus=new SparseEventBus();this.neuron=new LIFNeuron();this.memory=new Map();this.adapter=new SemanticAdapter();this.capabilities=new CapabilityRouter();
    this.backupPlanners=new Map([[UNIT.MEMORY,new DistilledBackupPlanner(UNIT.MEMORY)],[UNIT.PERCEPTION,new DistilledBackupPlanner(UNIT.PERCEPTION)]]);
    this.localEnergy=0;this.globalEnergy=0;this.attention=[];this.observeWorld(false);return this.snapshot();
  }
  worldState(){return {size:[12,8],agent:[...this.agent],goal:[...this.goal],obstacles:new Set(this.knownObstacles)}}
  packet(source,concept,salience,values={}){return {source,target:UNIT.WORKSPACE,concept,tick:this.tick,semantic_code:0,quantization_error:0,confidence:1,uncertainty:0,salience,urgency:0,risk:0,expected_reward:0,ttl_ms:0,state_delta:[],latent:[],flags:0,...values}}
  proposal(arm,action,confidence,values={}){return {arm,action,confidence,expected_reward:0,risk:0,urgency:0,energy_cost:0,reason_code:REASON.NONE,reason_value:0,...values}}
  nextCells(){return Object.fromEntries(Object.entries(MOVES).map(([action,[dx,dy]])=>[Number(action),[this.agent[0]+dx,this.agent[1]+dy]]))}
  observeWorld(emit=true){this.visibleSet=new Set();let fresh=0;for(let y=Math.max(0,this.agent[1]-1);y<=Math.min(7,this.agent[1]+1);y++)for(let x=Math.max(0,this.agent[0]-1);x<=Math.min(11,this.agent[0]+1);x++){const k=`${x},${y}`;this.visibleSet.add(k);if(!this.knownObstacles.has(k)&&!this.knownFree.has(k))fresh++;if(this.obstacleSet.has(k)){this.knownObstacles.add(k);this.knownFree.delete(k)}else{this.knownFree.add(k);this.knownObstacles.delete(k)}}if(!emit)return {events:[]};this.localEnergy+=.03;const coverage=(this.knownObstacles.size+this.knownFree.size)/96,learned=this.adapter.encodeSemantic(UNIT.PREDICTION,CONCEPT.WORLD_MODEL_UPDATE,[coverage,1-coverage,fresh/9,this.knownObstacles.size/96]);return {events:[this.packet(UNIT.PREDICTION,CONCEPT.WORLD_MODEL_UPDATE,Math.min(.8,.2+fresh/9),{confidence:coverage,uncertainty:1-coverage,ttl_ms:300,state_delta:[fresh,this.knownObstacles.size+this.knownFree.size,this.knownObstacles.size],latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})],fresh,coverage}}
  routeAction(obstacles){const start=key(this.agent),goal=key(this.goal),queue=[this.agent],parents=new Map(),actions=new Map(),seen=new Set([start]);while(queue.length){const cell=queue.shift(),cellKey=key(cell);if(cellKey===goal){let cursor=cellKey;while(parents.get(cursor)!==start)cursor=parents.get(cursor);return actions.get(cursor)}for(const [raw,[dx,dy]] of Object.entries(MOVES)){const next=[cell[0]+dx,cell[1]+dy],nextKey=key(next);if(next[0]<0||next[0]>=12||next[1]<0||next[1]>=8||obstacles.has(nextKey)||seen.has(nextKey))continue;seen.add(nextKey);parents.set(nextKey,cellKey);actions.set(nextKey,Number(raw));queue.push(next)}}return ACTION.WAIT}
  navigation(){
    const cells=this.nextCells();
    const valid=Object.entries(cells).filter(([,cell])=>cell[0]>=0&&cell[0]<12&&cell[1]>=0&&cell[1]<8&&!this.knownObstacles.has(key(cell)));
    this.localEnergy+=.035+.008*valid.length;
    if(!valid.length){const learned=this.adapter.encodeSemantic(UNIT.PLANNING,CONCEPT.TRAPPED,[1,1]);return {events:[this.packet(UNIT.PLANNING,CONCEPT.TRAPPED,1,{risk:1,urgency:1,ttl_ms:250,latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})],proposals:[this.proposal(UNIT.PLANNING,ACTION.WAIT,1,{risk:1,urgency:1,reason_code:REASON.NO_VALID_MOVE})],cells}}
    const distance=cell=>Math.abs(cell[0]-this.goal[0])+Math.abs(cell[1]-this.goal[1]);
    let action=this.routeAction(this.knownObstacles);if(!valid.some(([raw])=>Number(raw)===action))action=Number(valid[0][0]);const cell=cells[action],nextDistance=distance(cell),currentDistance=distance(this.agent),progress=currentDistance-nextDistance,confidence=progress>0?.78:.64,reward=progress>0?.82:.55;
    const learned=this.adapter.encodeSemantic(UNIT.PLANNING,CONCEPT.ROUTE_PROPOSAL,[action/4,cell[0]/11,cell[1]/7,nextDistance/18,confidence,reward]);
    return {events:[this.packet(UNIT.PLANNING,CONCEPT.ROUTE_PROPOSAL,.42,{confidence,uncertainty:1-confidence,risk:.05,urgency:.22,expected_reward:reward,ttl_ms:250,state_delta:[action,cell[0],cell[1],nextDistance],latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})],proposals:[this.proposal(UNIT.PLANNING,action,confidence,{expected_reward:reward,risk:.05,urgency:.22,energy_cost:.08,reason_code:REASON.GOAL_PROGRESS,reason_value:nextDistance})],cells};
  }
  backupNavigation(owner){const planner=this.backupPlanners.get(owner),state=this.worldState();for(const cell of this.memory.keys())state.obstacles.add(cell);const initial=planner.cells(this.agent),hasUnvisited=Object.values(initial).some(cell=>cell[0]>=0&&cell[0]<12&&cell[1]>=0&&cell[1]<8&&!state.obstacles.has(key(cell))&&!this.visited.has(key(cell)));if(hasUnvisited)for(const cell of this.visited)if(cell!==key(this.agent)&&cell!==key(this.goal))state.obstacles.add(cell);const cells=planner.cells(this.agent),valid=Object.keys(MOVES).map(Number).filter(candidate=>{const c=cells[candidate];return c[0]>=0&&c[0]<12&&c[1]>=0&&c[1]<8&&!state.obstacles.has(key(c))});let action=planner.choose(state);if(!valid.includes(action))action=valid.reduce((best,candidate)=>planner.score(candidate,planner.features(state,candidate))>planner.score(best,planner.features(state,best))?candidate:best);const ranked=valid.map(candidate=>planner.score(candidate,planner.features(state,candidate))).sort((a,b)=>a-b),margin=ranked.length>1?ranked.at(-1)-ranked.at(-2):0,confidence=Math.min(.92,.62+.2*Math.tanh(Math.max(0,margin))),cell=cells[action],distance=planner.distance(cell,this.goal);planner.energy+=.022;const learned=this.adapter.encodeSemantic(owner,CONCEPT.ROUTE_PROPOSAL,[action/4,cell[0]/11,cell[1]/7,distance/18,confidence,.76]);return {events:[this.packet(owner,CONCEPT.ROUTE_PROPOSAL,.46,{confidence,uncertainty:1-confidence,risk:.08,urgency:.3,expected_reward:.76,ttl_ms:250,state_delta:[action,cell[0],cell[1],distance],latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError,flags:2})],proposals:[this.proposal(owner,action,confidence,{expected_reward:.76,risk:.08,urgency:.3,energy_cost:.04,reason_code:REASON.GOAL_PROGRESS,reason_value:distance})],cells}}
  vision(){
    const adjacent=Object.values(this.nextCells()).filter(cell=>this.obstacleSet.has(key(cell))).length;
    const danger=adjacent/2, spike=this.neuron.step(danger);this.localEnergy+=.055+.035*danger;
    const concept=spike?CONCEPT.DANGER_SPIKE:CONCEPT.VISUAL_SCAN, learned=this.adapter.encodeSemantic(UNIT.PERCEPTION,concept,[adjacent/4,danger,this.neuron.potential,spike?1:0]);
    const events=[this.packet(UNIT.PERCEPTION,concept,Math.min(1,danger),{confidence:Math.min(1,.55+danger),uncertainty:Math.max(0,.45-danger/2),urgency:spike?.86:0,risk:Math.min(1,danger),ttl_ms:180,state_delta:[adjacent],latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError,flags:spike?1:0})];
    const proposals=spike&&adjacent===4?[this.proposal(UNIT.PERCEPTION,ACTION.WAIT,.86,{risk:Math.min(1,.72+danger/3),urgency:.86,energy_cost:.05,reason_code:REASON.DANGER_SPIKE})]:[];
    return {events,proposals,snn:{danger,spike,potential:this.neuron.potential}};
  }
  recall(cells){
    this.localEnergy+=.025;
    const risky=Object.entries(cells).map(([action,cell])=>[action,cell,this.memory.get(key(cell))||0]).filter(item=>item[2]);
    if(!risky.length){const learned=this.adapter.encodeSemantic(UNIT.MEMORY,CONCEPT.MEMORY_QUIET);return {events:[this.packet(UNIT.MEMORY,CONCEPT.MEMORY_QUIET,.15,{confidence:.8,ttl_ms:100,latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})],proposals:[]}}
    risky.sort((a,b)=>b[2]-a[2]);const [action,cell,count]=risky[0];
    const learned=this.adapter.encodeSemantic(UNIT.MEMORY,CONCEPT.COLLISION_RECALL,[cell[0]/11,cell[1]/7,Math.min(1,count/5),Number(action)/4]);
    return {events:[this.packet(UNIT.MEMORY,CONCEPT.COLLISION_RECALL,Math.min(1,.55+.12*count),{confidence:Math.min(.95,.55+.1*count),risk:Math.min(.98,.6+.1*count),urgency:.55,ttl_ms:500,state_delta:[cell[0],cell[1],count,Number(action)],latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})],proposals:[this.proposal(UNIT.MEMORY,ACTION.WAIT,Math.min(.95,.55+.1*count),{risk:Math.min(.98,.6+.1*count),urgency:.55,energy_cost:.02,reason_code:REASON.COLLISION_MEMORY,reason_value:count})]};
  }
  score(p){return .34*p.confidence+.24*p.expected_reward+.28*p.risk+.22*p.urgency-.08*p.energy_cost}
  arbitrate(proposals){
    if(!proposals.length)return {action:ACTION.WAIT,winner:UNIT.WORKSPACE,score:0,reason_code:REASON.NONE,reason_value:0,safety_veto:false,proposals};
    const vetoes=proposals.filter(p=>p.risk>=.9&&p.urgency>=.75), candidates=vetoes.length?vetoes:proposals;
    const winner=candidates.reduce((best,p)=>this.score(p)>this.score(best)?p:best);
    return {action:winner.action,winner:winner.arm,score:+this.score(winner).toFixed(4),reason_code:winner.reason_code,reason_value:winner.reason_value,safety_veto:Boolean(vetoes.length),proposals};
  }
  step(){
    if(this.done)return this.snapshot();this.tick++;
    const prediction=this.observeWorld(true);
    const available=new Set([UNIT.PLANNING,UNIT.MEMORY,UNIT.PERCEPTION].filter(unit=>!this.disabledUnits.has(unit))), planningLease=this.capabilities.resolve(CAPABILITY.PLAN_ROUTE,available);
    const nav=!planningLease?{events:[],proposals:[],cells:this.nextCells()}:planningLease.owner===UNIT.PLANNING?this.navigation():this.backupNavigation(planningLease.owner);
    const visual=this.disabledUnits.has(UNIT.PERCEPTION)?{events:[],proposals:[],snn:{danger:0,spike:false,potential:this.neuron.potential}}:this.vision();
    const memory=this.disabledUnits.has(UNIT.MEMORY)?{events:[],proposals:[]}:this.recall(nav.cells);
    const rawEvents=[...prediction.events,...nav.events,...visual.events,...memory.events];
    const observed=rawEvents.map(event=>({...packetTrace(event),published:this.bus.observe(event)}));
    const salient=this.bus.drain();this.attention=[...salient].sort((a,b)=>b.salience-a.salience).slice(0,5);
    const proposals=[...nav.proposals,...visual.proposals,...memory.proposals];this.globalEnergy+=.1+.16*salient.length+.04*proposals.length;
    this.lastDecision=this.arbitrate(proposals);let collision=false,executionEvent=null;
    if(MOVES[this.lastDecision.action]){
      const [dx,dy]=MOVES[this.lastDecision.action], target=[this.agent[0]+dx,this.agent[1]+dy];
      if(target[0]<0||target[0]>=12||target[1]<0||target[1]>=8||this.obstacleSet.has(key(target))){collision=true;this.collisions++;this.memory.set(key(target),(this.memory.get(key(target))||0)+1);const learned=this.adapter.encodeSemantic(UNIT.EXECUTION,CONCEPT.COLLISION,[target[0]/11,target[1]/7,1,1]);executionEvent=this.packet(UNIT.EXECUTION,CONCEPT.COLLISION,1,{risk:1,urgency:1,ttl_ms:500,state_delta:target,latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})}else{this.agent=target;this.visited.add(key(this.agent))}
    }
    if(key(this.agent)===key(this.goal)){this.done=true;const learned=this.adapter.encodeSemantic(UNIT.EXECUTION,CONCEPT.GOAL_REACHED,[this.goal[0]/11,this.goal[1]/7,1]);executionEvent=this.packet(UNIT.EXECUTION,CONCEPT.GOAL_REACHED,1,{expected_reward:1,ttl_ms:1000,state_delta:this.goal,latent:learned.quantizedLatent,semantic_code:learned.semanticCode,quantization_error:learned.quantizationError})}
    if(executionEvent)observed.push({...packetTrace(executionEvent),published:this.bus.observe(executionEvent)});
    const activity=(unit,label,events,unitProposals,detail)=>({unit:UNIT_NAME[unit],label,status:unitProposals.length&&this.lastDecision.winner===unit?'winner':unitProposals.length?'proposing':'active',detail,events:events.map(event=>observed.find(item=>item.source===UNIT_NAME[unit]&&item.kind===CONCEPT_NAME[event.concept])),proposals:unitProposals.map(proposalTrace)});
    const executionEvents=executionEvent?[executionEvent]:[];
    const unitActivity=[
      activity(UNIT.PREDICTION,'Prediction',prediction.events,[],`${this.knownObstacles.size+this.knownFree.size}/96 cells reconstructed`),
      activity(planningLease?.owner??UNIT.PLANNING,planningLease?.delegated?`Planning → ${UNIT_NAME[planningLease.owner]}`:'Planning',nav.events,nav.proposals,planningLease?`${Object.values(nav.cells).length} candidate cells evaluated`:'no healthy planning-capable unit'),
      activity(UNIT.PERCEPTION,'Perception',visual.events,visual.proposals,`danger ${visual.snn.danger.toFixed(2)} · membrane ${visual.snn.potential.toFixed(2)} · spike ${visual.snn.spike?'yes':'no'}`),
      activity(UNIT.MEMORY,'Memory',memory.events,memory.proposals,`${this.memory.size} collision cells stored`),
      activity(UNIT.EXECUTION,'Execution',executionEvents,[],collision?'movement blocked':this.done?'goal confirmed':`position committed to ${this.agent.join(':')}`),
    ];
    return {...this.snapshot(),transition:{collision,snn:visual.snn,planning_owner:planningLease?UNIT_NAME[planningLease.owner]:null,planning_delegated:Boolean(planningLease?.delegated),all_events:observed,salient_events:salient.map(packetTrace),unit_activity:unitActivity,pipeline:{observed:observed.length,published:observed.filter(event=>event.published).length,filtered:observed.filter(event=>!event.published).length,proposals:proposals.length,adapter_updates:this.adapter.steps}}};
  }
  snapshot(){const backupEnergy=[...this.backupPlanners.values()].reduce((sum,planner)=>sum+planner.energy,0),known=this.knownObstacles.size+this.knownFree.size,visible=[];for(let y=Math.max(0,this.agent[1]-1);y<=Math.min(7,this.agent[1]+1);y++)for(let x=Math.max(0,this.agent[0]-1);x<=Math.min(11,this.agent[0]+1);x++)visible.push([x,y]);return {size:[12,8],agent:[...this.agent],goal:[...this.goal],obstacles:this.obstacles.map(cell=>[...cell]),belief_obstacles:[...this.knownObstacles].map(value=>value.split(',').map(Number)),belief_free:[...this.knownFree].map(value=>value.split(',').map(Number)),visible,tick:this.tick,done:this.done,collisions:this.collisions,decision:this.lastDecision?decisionTrace(this.lastDecision):null,attention:this.attention.map(packetTrace),metrics:{observations:this.bus.observations,published_events:this.bus.published,event_sparsity:+this.bus.sparsity.toFixed(3),snn_spikes:this.neuron.spikes,local_energy:+(this.localEnergy+backupEnergy).toFixed(3),global_energy:+this.globalEnergy.toFixed(3),adapter_loss:+this.adapter.lastLoss.toFixed(6),adapter_mean_loss:+this.adapter.meanLoss.toFixed(6),adapter_steps:this.adapter.steps,semantic_codes_used:this.adapter.quantizer.codesUsed,quantization_error:+this.adapter.quantizer.lastError.toFixed(6),capability_handoffs:this.capabilities.handoffs,planning_owner:UNIT_NAME[this.capabilities.assignments.get(CAPABILITY.PLAN_ROUTE)??UNIT.PLANNING],backup_accuracy:+Math.max(...[...this.backupPlanners.values()].map(planner=>planner.trainingAccuracy)).toFixed(3),world_model_cells:known,world_model_coverage:known/96}}}
}

const simulation=new OctoSimulation();let timer=null;
function render(s){
  const obs=new Set(s.obstacles.map(key)),visible=new Set(s.visible.map(key)),beliefObs=new Set(s.belief_obstacles.map(key)),beliefFree=new Set(s.belief_free.map(key));$('#grid').innerHTML='';$('#belief-grid').innerHTML='';
  for(let y=0;y<s.size[1];y++)for(let x=0;x<s.size[0];x++){
    const d=document.createElement('div');d.className='cell';
    const k=`${x},${y}`;if(obs.has(k))d.classList.add('wall');if(visible.has(k))d.classList.add('sensed');if(x===s.goal[0]&&y===s.goal[1])d.classList.add('goal');if(x===s.agent[0]&&y===s.agent[1])d.classList.add('agent','pulse');$('#grid').appendChild(d);
    const b=document.createElement('div');b.className='cell';if(beliefObs.has(k))b.classList.add('wall','known');else if(beliefFree.has(k))b.classList.add('known');else b.classList.add('unknown');if(x===s.goal[0]&&y===s.goal[1])b.classList.add('goal');if(x===s.agent[0]&&y===s.agent[1])b.classList.add('agent','pulse');$('#belief-grid').appendChild(b);
  }
  const m=s.metrics;$('#metrics').innerHTML=[['Tick',s.tick],['Spikes',m.snn_spikes],['Published',m.published_events],['Adapter loss',m.adapter_loss],['VQ error',m.quantization_error],['Codes used',`${m.semantic_codes_used}/8`],['Handoffs',m.capability_handoffs],['Local energy',m.local_energy],['Global energy',m.global_energy]].map(([k,v])=>`<div class="metric"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#status').textContent=s.done?'HEDEFE ULAŞTI':`KONUM ${s.agent.join(':')}`;$('#model-status').textContent=`%${Math.round(m.world_model_coverage*100)} BİLİNİYOR`;$('#sparsity').textContent=`%${Math.round(m.event_sparsity*100)} FİLTRELENDİ`;
  if(s.decision){const tr={UP:'YUKARI',DOWN:'AŞAĞI',LEFT:'SOL',RIGHT:'SAĞ',WAIT:'BEKLE'},away=(s.decision.action==='DOWN'&&s.goal[1]===0)||(s.decision.action==='UP'&&s.goal[1]===7);$('#winner').textContent=s.decision.winner.toUpperCase();$('#decision').className='decision';$('#decision').innerHTML=`<strong>${tr[s.decision.action]}</strong><small>${s.decision.winner} kolunun önerisi seçildi · güven skoru ${s.decision.score}</small>`;$('#story').textContent=s.done?'Ajan hedefe ulaştı. Bunu tam haritayı hiç görmeden, parça parça kurduğu iç modelle yaptı.':away?'Ajan duvarın çevresini dolaşmak için bilinçli olarak hedeften uzaklaşıyor. Bu, basit “hedefe doğru git” davranışı değil.':`Prediction kolu ${m.world_model_cells} hücreyi hatırlıyor. Global Workspace şimdi ${tr[s.decision.action].toLowerCase()} hareketini seçti.`;$('#proposals').innerHTML=s.decision.proposals.map(p=>`<div class="proposal"><span class="arm">${p.arm}</span><div>güven ${Math.round(p.confidence*100)}%<div class="bar"><i style="width:${Math.round(p.confidence*100)}%"></i></div></div><b>${tr[p.action]}</b></div>`).join('')}else{$('#winner').textContent='BEKLİYOR';$('#decision').className='decision empty';$('#decision').textContent='Henüz karar verilmedi.';$('#proposals').innerHTML=''}
  const activity=s.transition?.unit_activity||[];$('#activities').innerHTML=activity.length?activity.map(a=>{const event=a.events[0],proposal=a.proposals[0];return `<article class="activity ${a.status}"><header><b>${a.label}</b><span>${a.status}</span></header><p>${a.detail}</p><dl><div><dt>event</dt><dd>${event?event.kind:'none'}</dd></div><div><dt>semantic code</dt><dd>${event?`C${event.payload.semantic_code} · ε ${event.payload.quantization_error.toFixed(3)}`:'—'}</dd></div><div><dt>route</dt><dd>${event?(event.published?'published':'filtered'):'none'}</dd></div><div><dt>proposal</dt><dd>${proposal?proposal.action:'none'}</dd></div><div><dt>quantized latent</dt><dd>${event?shortLatent(event.payload.latent).join(' · '):'—'}</dd></div></dl></article>`}).join(''):'<div class="empty">Step the simulation to inspect local work.</div>';
  const ev=s.transition?.all_events||s.attention||[];$('#events').innerHTML=ev.length?ev.map(e=>`<div class="event ${e.published===false?'filtered':'published'}"><b>${e.source}/${e.kind}</b><span>${e.published===false?'FILTERED':'PUBLISHED'} · ${e.salience.toFixed(2)}</span><small>code C${e.payload.semantic_code} · VQ ε ${e.payload.quantization_error.toFixed(3)} · latent [${shortLatent(e.payload.latent).join(', ')}] · Δ [${e.payload.state_delta.join(', ')}] · risk ${e.payload.risk.toFixed(2)} · urgency ${e.payload.urgency.toFixed(2)}</small></div>`).join(''):'<div class="empty">No observations this tick.</div>';if(s.done&&timer)toggleRun();
}
function step(){render(simulation.step())}function reset(){render(simulation.reset())}
function togglePlanningFault(){if(simulation.disabledUnits.has(UNIT.PLANNING)){simulation.disabledUnits.delete(UNIT.PLANNING);$('#fault').textContent='Planlamayı boz';$('#fault').classList.remove('active')}else{simulation.disabledUnits.add(UNIT.PLANNING);$('#fault').textContent='Planlamayı geri getir';$('#fault').classList.add('active')}render(simulation.snapshot())}
function toggleRun(){if(timer){clearInterval(timer);timer=null;$('#run').textContent='Otomatik';$('#run').classList.remove('active')}else{timer=setInterval(step,450);$('#run').textContent='Duraklat';$('#run').classList.add('active')}}
function renderBenchmark(){const target=$('#benchmark');if(!target)return;target.innerHTML=`<table><thead><tr><th>Scenario</th><th>Architecture</th><th>Success</th><th>Steps*</th><th>Collisions</th><th>Energy</th></tr></thead><tbody>${BENCHMARK_RESULTS.map(([scenario,architecture,success,steps,collisions,energy])=>`<tr class="${architecture==='octocortex'?'ours':''}"><td>${scenario}</td><td>${architecture}</td><td>${Math.round(success*100)}%</td><td>${steps}</td><td>${collisions}</td><td>${energy.toFixed(3)}</td></tr>`).join('')}</tbody></table><small>* Steps are averaged over successful episodes. Component outage probability: 35%; sensor noise: 15%.</small>`}
function renderGeneralization(){const target=$('#generalization');if(!target)return;target.innerHTML=`<table><thead><tr><th>Backup owner</th><th>Unseen worlds</th><th>Success</th><th>Steps*</th><th>Collisions</th><th>Action match</th></tr></thead><tbody>${GENERALIZATION_RESULTS.map(([owner,worlds,success,steps,collisions,agreement])=>`<tr class="ours"><td>${owner}</td><td>${worlds}</td><td>${Math.round(success*100)}%</td><td>${steps}</td><td>${collisions.toFixed(2)}</td><td>${Math.round(agreement*100)}%</td></tr>`).join('')}</tbody></table><small>32 procedural training worlds and 100 disjoint test worlds; zero map overlap. Route-existence is constrained, maze search is not yet tested.</small>`}
function renderWorldModel(){const target=$('#world-model');if(!target)return;target.innerHTML=`<table><thead><tr><th>Condition</th><th>Hard worlds</th><th>Success</th><th>Steps*</th><th>Collisions</th><th>Map coverage</th></tr></thead><tbody>${WORLD_MODEL_RESULTS.map(([condition,worlds,success,steps,collisions,coverage])=>`<tr class="${condition.startsWith('octocortex')?'ours':''}"><td>${condition}</td><td>${worlds}</td><td>${Math.round(success*100)}%</td><td>${steps??'—'}</td><td>${collisions.toFixed(2)}</td><td>${Math.round(coverage*100)}%</td></tr>`).join('')}</tbody></table><small>Only a 3×3 local sensor window is available. Every held-out world requires a temporary move away from the goal.</small>`}
function renderDynamicModel(){const target=$('#dynamic-model');if(!target)return;target.innerHTML=`<table><thead><tr><th>Belief policy</th><th>Seeded trials</th><th>Success</th><th>Steps*</th><th>Expired cells</th></tr></thead><tbody>${DYNAMIC_RESULTS.map(([condition,worlds,success,steps,expired])=>`<tr class="${condition.startsWith('adaptive')?'ours':''}"><td>${condition}</td><td>${worlds}</td><td>${Math.round(success*100)}%</td><td>${steps}</td><td>${expired}</td></tr>`).join('')}</tbody></table><small>Gate position, route orientation, change timing, and sensor noise vary. Adaptive belief expires stale cells; static belief keeps obsolete obstacles.</small>`}
if(typeof module!=='undefined')module.exports={UNIT,CONCEPT,ACTION,REASON,CAPABILITY,BENCHMARK_RESULTS,GENERALIZATION_RESULTS,WORLD_MODEL_RESULTS,DYNAMIC_RESULTS,LIFNeuron,SparseEventBus,CapabilityRouter,DistilledBackupPlanner,OnlineVectorQuantizer,SemanticAdapter,OctoSimulation};
if(typeof document!=='undefined'){$('#step').onclick=step;$('#run').onclick=toggleRun;$('#fault').onclick=togglePlanningFault;$('#reset').onclick=reset;render(simulation.snapshot());renderBenchmark();renderGeneralization();renderWorldModel();renderDynamicModel()}
