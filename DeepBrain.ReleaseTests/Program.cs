using System.Reflection;
using System.Text.Json;
using DeepBrain.Host.BrainLife;
using DeepBrain.Host.BrainLife.Ml;
using DeepBrain.Experiments;
using ML.Host.Services;
using Server=ML.Shared.Protocol;
using Client=DeepBrain.Shared.MlBridge;
var root=Path.Combine(Path.GetTempPath(),"db-release-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(root);
var old=Directory.GetCurrentDirectory();Directory.SetCurrentDirectory(root);
try {
 var cfg=BrainConfig.Default with {Ml=BrainConfig.Default.Ml with {Enable=true,Backend="remote",CheckpointPath=Path.Combine(root,"model"),TrainEveryTicks=1,BatchSize=1},Memory=MemoryConfig.Default with {Enable=false},Habits=HabitConfig.Default with {Enable=false,ImportPath=""},Evaluation=BrainConfig.Default.Evaluation with {SaveReports=true},Episode=BrainConfig.Default.Episode with {MaxSteps=40}};
 var fake=new CaptureBackend();var advisor=new MlPolicyAdvisor(cfg.Ml,_=>{},fake);
 Check(MlMath.NormalizeMask(new float[8])!.All(x=>x==0),"empty mask remains empty");
 advisor.Observe(new float[38],0,0,new float[38],false,new float[8],cfg.Ml,1);
 Check(fake.Transitions.Single().ActionMask.All(x=>x==0),"empty mask reaches backend");fake.Transitions.Clear();
 var file=Path.Combine(root,"config.json");File.WriteAllText(file,JsonSerializer.Serialize(cfg));
 var loop=new LifeLoop(new WorldSim(1337),new HomeostasisEngine(),new InstinctEngine(),new EmotionEngine(),new ActionSelector(),new Actuator(),new RewardEngine(),new LearningEngine(),new BrainConfigLoader(file,_=>{}),(_,_)=>{},(_,_)=>{},(_,_)=>{},_=>{},advisor);
 var step=typeof(LifeLoop).GetMethod("TickOnce",BindingFlags.Instance|BindingFlags.NonPublic)!;
 for(int i=0;i<30;i++)step.Invoke(loop,new object[]{0.2,CancellationToken.None});
 Check(fake.Infers.Count==30 && fake.Transitions.Count==29,"transition waits for next decision");
 Check(fake.Transitions.Select((t,i)=>t.NextState.SequenceEqual(fake.Infers[i+1].State)&&t.ActionMask.SequenceEqual(fake.Infers[i+1].Mask)).All(x=>x),"training uses actual next inference state and mask");
 for(int i=0;i<15;i++)step.Invoke(loop,new object[]{0.2,CancellationToken.None});
 Check(fake.Transitions.Any(t=>t.Done&&t.ActionMask.All(v=>v==0)),"terminal transition has no bootstrap");
 loop.FlushPersistentState();Check(fake.Saves>0,"shutdown saves ML");
 Check(Directory.GetFiles(root,"episodes.jsonl",SearchOption.AllDirectories).Any(),"shutdown drains reports");
 var backend=new SerializedBackend(cfg.Ml,_=>{});var real=new MlPolicyAdvisor(cfg.Ml,_=>{},backend);
 await backend.InferAsync(new float[38],Enumerable.Repeat(1f,8).ToArray(),cfg.Ml,CancellationToken.None);real.TrySave(cfg.Ml.CheckpointPath,12);
 using(var w=JsonDocument.Parse(File.ReadAllText(cfg.Ml.CheckpointPath))){var remote=w.RootElement.GetProperty("WeightsPath").GetString()!;Check(remote!=cfg.Ml.CheckpointPath&&File.Exists(remote+".service"),"absolute paths do not collide");using var meta=JsonDocument.Parse(File.ReadAllText(remote+".service"));Check(File.Exists(meta.RootElement.GetProperty("weightsPath").GetString()),"metadata points to committed weights");}
 var fresh=new MlPolicyAdvisor(cfg.Ml,_=>{},new SerializedBackend(cfg.Ml,_=>{}));Check(fresh.TryLoad(cfg.Ml.CheckpointPath,out var ep)&&ep==12,"checkpoint round trip");
 var service=new BrainMlService();var req=new Client.MlInferRequest(new float[]{1,0},new[]{true,true,true},2,3){Seed=2027,LearningRate=0.002};
 var decoded=JsonSerializer.Deserialize<Server.MlInferRequest>(JsonSerializer.Serialize(req),new JsonSerializerOptions{PropertyNameCaseInsensitive=true})!;
 var first=service.Infer(decoded);Check(first.Ok,"initialization via JSON succeeds");
 var saved=service.Save(new Server.MlCheckpointRequest(Path.Combine(root,"server")));using(var meta=JsonDocument.Parse(saved.Meta!))Check(meta.RootElement.GetProperty("seed").GetInt32()==2027&&meta.RootElement.GetProperty("learningRate").GetDouble()==0.002,"configured seed and learning rate honored");
 Check(!service.Infer(decoded with {InputDim=4,State=new float[4]}).Ok,"dimension mismatch rejected");Check(service.Infer(decoded).QValues.SequenceEqual(first.QValues),"rejected request preserves model");
 Check(!new BrainMlService().Save(new Server.MlCheckpointRequest(Path.Combine(root,"empty"))).Ok,"uninitialized model is not saved");
 Check(service.Reset(new Server.MlResetRequest(2027,0.002)).Ok,"service reset succeeds");
 Check(!service.Save(new Server.MlCheckpointRequest(Path.Combine(root,"reset-empty"))).Ok,"reset removes old network");
 Check(service.Infer(decoded).QValues.SequenceEqual(first.QValues),"reset reproduces seeded fresh network");
 Console.WriteLine("RELEASE ACCEPTANCE PASSED");
}finally{Directory.SetCurrentDirectory(old);Directory.Delete(root,true);}
static void Check(bool b,string name){if(!b)throw new Exception("FAIL: "+name);Console.WriteLine("PASS: "+name);}
sealed class CaptureBackend:IBrainMlBackend {
 public List<(float[] State,float[] Mask)> Infers=new();public List<Transition> Transitions=new();public int Saves;
 public bool IsAvailable=>true;public string Kind=>"remote";public bool IsConnected=>true;public string? LastError=>null;public double LastRttMs=>0;public int InputDim=>38;public int ActionCount=>8;public int BufferSize=>Transitions.Count;public int BufferCapacity=>20000;public double LastLoss=>0;public double AvgLoss100=>0;public long TrainSteps=>Transitions.Count;public double AvgQ=>0;
 public Task<MlInferResult> InferAsync(float[] s,float[] m,MlConfig c,CancellationToken ct){Infers.Add((s.ToArray(),m.ToArray()));return Task.FromResult(new MlInferResult(true,new double[8],null,0,0));}
 public Task<MlTrainResult> TrainAsync(Transition t,MlConfig c,long tick,CancellationToken ct){Transitions.Add(t);return Task.FromResult(new MlTrainResult(true,0,0,Transitions.Count,false));}
 public bool TryLoad(string p,out int ep){ep=0;return false;}public void TrySave(string p,int ep){Saves++;}public void Reset(MlConfig c){}public void ResetCounters(){}public void UpdateConfig(MlConfig c){}public bool TryConnect()=>true;public void Disconnect(){}public ValueTask DisposeAsync()=>ValueTask.CompletedTask;
}
