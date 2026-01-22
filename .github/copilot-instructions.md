# AI Coding Guidelines for ML Framework

## Architecture Overview
This is a modular C# machine learning framework built on .NET 8. Core components:
- **ML.Core**: Pure library with abstractions (ILayer, IModel, ILoss, IOptimizer) and implementations (Layers/, Losses/, Optimizers/). No external dependencies.
- **ML.Examples**: Console app running preset examples (XOR, AND, Threshold, Emotion) via `ExampleRegistry`.
- **ML.Gui**: Avalonia-based desktop UI for training visualization and inference monitoring.
- **ML.Host**: Server component (less documented).
- **ML.Models**: Directory for saved models (JSON format) and demo data.

Data flows: Datasets (CSV/JSON) → Trainer (epochs/batches) → Network (layers) → ModelStore (save/load).

## Key Patterns
- **Network Composition**: `Network` aggregates `ILayer`s; forward/backward passes chain through layers.
- **Training Loop**: `Trainer.Train()` with `TrainOptions` (epochs, batchSize, callbacks); uses `CancellationToken` for soft stops.
- **Serialization**: Models saved as JSON DTOs (NetworkDto, LayerDto) in `ML/Models/`. Use `ModelStore.Save/Load`.
- **Inference**: `InferenceSession` for real-time predictions; integrates with GUI monitoring.
- **Datasets**: Tuples `(double[] x, int y)`; loaded via `DatasetLoader` (CSV: last column label; JSON: `{x:[], y:0}`).
- **Layers**: Factory-based creation; common: `LinearLayer`, `ActivationLayer` (ReLU/Sigmoid), `SoftmaxLayer`.

## Workflows
- **Build**: `dotnet build <project>.csproj` (e.g., `dotnet build ML.Core/ML.Core.csproj`).
- **Run Examples**: `dotnet run --project ML.Examples` (runs Emotion by default).
- **Run GUI**: `dotnet restore && dotnet run --project ML.Gui` (requires Avalonia 11 + LiveChartsCore 2.0.0-rc2).
- **Debug Training**: Use `TrainingHost` in GUI for state management (Idle/Running/Stopping); callbacks for epoch metrics.
- **Add Layer**: Implement `ILayer`, add to `Network.Add()`; serialize in `ModelStore` switch-case.

## Conventions
- **Naming**: PascalCase classes/interfaces (e.g., `ILayer`, `Network`); folders mirror namespaces.
- **Error Handling**: Argument validation in constructors/methods; throw `ArgumentException` for invalid inputs.
- **Dependencies**: Core is dependency-free; GUI adds Avalonia/ReactiveUI. Pin versions (e.g., Avalonia 11.0.10).
- **Code Style**: Implicit usings, nullable enabled; comments in Russian for domain-specific logic.
- **Model Paths**: Relative to `ML/Models/`; resolve via `ModelStore.ResolveModelsDir()`.

## Examples
- Create network: `var net = new Network(); net.Add(new LinearLayer(2, 3)); net.Add(new ActivationLayer(Activation.ReLU));`
- Train: `new Trainer(net, new AdamOptimizer(), new CrossEntropyLoss()).Train(dataset, new TrainOptions { Epochs = 100 });`
- Save: `net.Save("myModel");` (creates `ML/Models/myModel.json`).