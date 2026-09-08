using DeepBrain.Host.BrainLife.Ml;
namespace DeepBrain.Experiments;

public static class DiagnosticTests
{
    public static void Run()
    {
        var names = ActionCatalog.Actions;
        var allowed = new[] {names[0],names[1]};
        var scores = new Dictionary<string,double> {{names[0],1},{names[1],0.5}};
        var mask = new float[ActionCatalog.Count]; mask[0]=mask[1]=1;
        var q = Enumerable.Repeat(100000.0,ActionCatalog.Count).ToArray(); q[0]=0.1;q[1]=0.2;
        var choice = new PolicyDecision(names[0],0.6,0,0,"net",true,false,false);
        var p = DiagnosticAdvisor.Analyze(1,scores,allowed,mask,q,choice)!;
        Require(Math.Abs(p.QRange-0.1)<1e-9 && p.LegalCount==2 && p.NetworkAction==names[1] && p.HeuristicAction==names[0],"forbidden Q excluded; same-state argmax differs");
        Require(p.ProbabilitySpread>0 && p.ProbabilitySpread<0.1 && p.AdvisorAction==names[0],"small probability spread need not change chosen action");
        q[1]=q[0]; p=DiagnosticAdvisor.Analyze(2,scores,allowed,mask,q,choice)!;
        Require(p.TopTwoQGap==0 && p.ProbabilitySpread==0 && p.MaxProbability==0.5,"tied legal values");
        Require(DiagnosticAdvisor.Analyze(3,scores,allowed,new float[ActionCatalog.Count],q,choice) is null,"no legal action produces no fabricated diagnostic");
        var sequence = SequenceDiagnostics.Compare(new[]{"a","sleep","b"},new[]{"a","a","c"});
        Require(sequence.DifferentActions==2 && sequence.FirstDifferentTick==1 && sequence.BothAwakeTicks==2 && sequence.DifferentActionsBothAwake==1,"sequence comparison separates sleep");
        Console.WriteLine("DIAGNOSTICS: 5/5 passed");
    }
    private static void Require(bool ok,string name)
    {
        if(!ok) throw new InvalidOperationException("FAIL: "+name);
        Console.WriteLine("PASS: "+name);
    }
}
