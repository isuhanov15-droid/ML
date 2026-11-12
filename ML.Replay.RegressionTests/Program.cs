using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using ML.Host.Services;
using ML.Shared.Protocol;

var root=Path.Combine(Path.GetTempPath(),"replay-tests-"+Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
int passed=0;
void Check(bool value,string name) { if(!value) throw new Exception("FAIL "+name); passed++;Console.WriteLine("PASS "+name); }
MlTrainResponse Add(BrainMlService s,int tick,int capacity=1024,int batch=int.MaxValue) => s.Train(new MlTrainRequest(
    1,tick,null,null,null,null,null,null,null,null,
    new MlTransitionDto(new[]{tick/10000f,0.5f},tick%3,tick/100000f,new[]{0.1f,tick/10000f},tick%7==0,null,
        tick%5==0?null:tick%2==0?new[]{false,false,false}:new[]{true,false,true}),
    new MlTrainConfigDto(capacity,batch,1,1,200,0.98,1,1337),2,3));
JsonObject Meta(string path)=>JsonNode.Parse(File.ReadAllText(path))!.AsObject();
string Replay(string path) {using var f=File.OpenRead(Meta(path)["replayPath"]!.GetValue<string>());using var z=new GZipStream(f,CompressionMode.Decompress);using var r=new StreamReader(z);return r.ReadToEnd();}
double[] Q(BrainMlService s)=>s.Infer(new MlInferRequest(null,null,new[]{0.2f,0.3f},new[]{true,true,true},null,null,null,2,3)).QValues;
try {
 foreach(var count in new[]{0,137,1120}) {
  var service=new BrainMlService();Q(service);for(int i=0;i<count;i++)if(!Add(service,i).Ok)throw new Exception("Transition rejected");
  var path=Path.Combine(root,"source"+count);Check(service.Save(new MlCheckpointRequest(path)).Ok,"save "+count);
  var original=Replay(path);var q=Q(service);var restored=new BrainMlService();Check(restored.Load(new MlCheckpointRequest(path)).Ok,"load into new service "+count);
  Check(Q(restored).SequenceEqual(q),"network unchanged "+count);
  var copy=path+"-copy";Check(restored.Save(new MlCheckpointRequest(copy)).Ok,"save restored "+count);
  Check(Replay(copy)==original,"every transition, mask and ring index preserved "+count);
  var a=Add(service,1200);var b=Add(restored,1200);Check(a.BufferSize==b.BufferSize,"next insertion count "+count);
  Check(service.Save(new MlCheckpointRequest(path)).Ok&&restored.Save(new MlCheckpointRequest(copy)).Ok,"save after insertion "+count);
  Check(Replay(path)==Replay(copy),"same oldest slot replaced "+count);
  if(count>0)Check(Add(restored,1201,batch:16).Trained,"training resumes without refilling "+count);
 }
 var live=new BrainMlService();for(int i=0;i<137;i++)Add(live,i);
 var file=Path.Combine(root,"protected");Check(live.Save(new MlCheckpointRequest(file)).Ok,"baseline");var meta=Meta(file);var baseline=Replay(file);var qBefore=Q(live);
 var replayPath=meta["replayPath"]!.GetValue<string>();var bytes=File.ReadAllBytes(replayPath);
 File.WriteAllBytes(replayPath,bytes[..(bytes.Length/2)]);
 Check(!live.Load(new MlCheckpointRequest(file)).Ok,"truncated replay fails closed");
 Check(Q(live).SequenceEqual(qBefore),"failed load leaves network live");
 File.WriteAllBytes(replayPath,bytes);
 var bad=JsonNode.Parse(baseline)!.AsObject();bad["nextIndex"]=999;
 using(var f=File.Create(replayPath))using(var z=new GZipStream(f,CompressionLevel.Fastest))JsonSerializer.Serialize(z,bad);
 meta["replaySha256"]=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(replayPath)));File.WriteAllText(file,meta.ToJsonString());
 Check(!live.Load(new MlCheckpointRequest(file)).Ok,"valid checksum cannot hide invalid ring metadata");
 var preserved=Path.Combine(root,"preserved");Check(live.Save(new MlCheckpointRequest(preserved)).Ok,"save live service after failed loads");
 Check(Replay(preserved)==baseline,"failed loads leave every replay entry unchanged");
 var stable=File.ReadAllText(preserved);var refused=live.Save(new MlCheckpointRequest(preserved){ExpectedInstanceId="wrong-instance"});
 Check(!refused.Ok&&File.ReadAllText(preserved)==stable,"failed save leaves published checkpoint unchanged");
 var missingVersion=Meta(preserved);missingVersion.Remove("schemaVersion");File.WriteAllText(file,missingVersion.ToJsonString());
 Check(!live.Load(new MlCheckpointRequest(file)).Ok,"missing schema cannot silently downgrade to weights-only");
 var old=Path.Combine(root,"legacy");var m=Meta(preserved);File.WriteAllText(old,new JsonObject { ["weightsPath"]=m["weightsPath"]!.GetValue<string>(),["trainSteps"]=77 }.ToJsonString());
 var legacy=new BrainMlService();Check(legacy.Load(new MlCheckpointRequest(old)).Ok,"legacy weights-only checkpoint supported");
 var trained=Add(legacy,1);Check(trained.BufferSize==1&&trained.TrainSteps==77,"legacy import keeps counters");
 Check(Add(live,2000,2048).BufferSize==138,"capacity increase retains all experience");
 for(int i=0;i<1100;i++)Add(live,3000+i,2048);
 Check(Add(live,5000,1024).BufferSize==1024,"capacity shrink retains newest experience");
 Console.WriteLine($"REPLAY REGRESSION PASSED ({passed} checks)");
} finally {Directory.Delete(root,true);}
