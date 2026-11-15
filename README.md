# ML для DeepBrain 1.12

Библиотека нейросетей, ML.Host, протоколы и настольные инструменты. Совместимая версия ядра и Web: [DeepBrain 1.12](https://github.com/isuhanov15-droid/DeepBrain/tree/DeepBrain-v1.12.0-release).

Эта ветка содержит действующую реализацию сохранения replay-буфера и двух сетей из `feature/replay-persistence`. Номер 1.12 обозначает совместимый комплект DeepBrain; алгоритм ML не изменён при публикации.

## Сборка и проверки

```bash
dotnet build ML.Host/ML.Host.csproj -c Release
dotnet run --project ML.Replay.RegressionTests -c Release
dotnet run --project ML.Bridge.RegressionTests -c Release
dotnet run --project DeepBrain.ReleaseTests -c Release
```

Нужен .NET SDK 8+. [Формат и гарантии сохранения опыта](REPLAY_PERSISTENCE.md).

Веса обученной Лады, её replay-буфер, переписка и рабочие контрольные точки не входят в исходники. Сохраняйте их отдельно и восстанавливайте согласованным комплектом. Примерные модели и наборы данных, существовавшие в репозитории, остаются примерами.
