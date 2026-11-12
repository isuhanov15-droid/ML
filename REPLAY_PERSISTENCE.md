# DeepBrain replay persistence

ML.Host checkpoint schema 2 stores the online network, target network and replay
buffer as one generation. Replay includes both states, action, reward, terminal
flag, next-action mask, capacity, count and physical ring write index. JSON is
compressed with gzip. Metadata contains SHA-256 hashes of all three files.

Files are immutable and flushed before atomically replacing metadata. A failed
save leaves the published checkpoint intact. Loading validates all components
before changing the live service. Missing, truncated, mismatched or invalid replay
is an error; it never silently becomes an empty buffer. Existing weights-only
checkpoints still load explicitly in legacy mode. Buffer resizing retains newest
experience instead of dropping the entire buffer.

The existing DeepBrain save schedule (every 500 ticks and graceful shutdown) now
includes replay. An abrupt power loss can still lose work since the last completed
checkpoint. Adam optimizer moments and the sampling RNG position are not included;
this is experience retention, not bit-identical training continuation. The target
network is preserved in schema 2; migration from an old weights-only checkpoint
initializes the target from the saved online network.

Validation commands:

```
dotnet run --project ML.Replay.RegressionTests -c Release
dotnet run --project ML.Bridge.RegressionTests -c Release
dotnet run --project DeepBrain.ReleaseTests -c Release
```

Regression coverage includes empty/partial/wrapped ring round trips, every field
and next overwrite position, immediate continued training, corrupt/truncated
files, invalid ring metadata despite a valid checksum, failed-load state isolation,
failed-save publication, legacy checkpoints and preserving experience during resize.

Production migration must export the currently running old service's buffer before
stopping that service: it cannot save replay itself. Stop the decision host first
to flush weights, export the stable ML heap, validate all entries and compare its
training counter with the checkpoint. Validate the constructed schema 2 checkpoint
in an isolated new ML process and compare the complete replay after re-saving.
Only then switch production. Back up the full memory/checkpoint tree. Never roll
back to the old weights-only writer against a schema 2 checkpoint, because its next
save would remove the replay references. Keep the previous full checkpoint backup
for recovery instead of overwriting newer experience with old data.
