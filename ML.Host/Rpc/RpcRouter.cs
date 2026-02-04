using System.Text.Json;
using ML.Host.Networking;
using ML.Shared.Protocol;

namespace ML.Host.Rpc;

public sealed class RpcRouter
{
    private readonly Dictionary<string, Func<RpcRequest, Task<object?>>> _handlers = new(StringComparer.OrdinalIgnoreCase);

    public string[] Methods => _handlers.Keys.OrderBy(x => x, StringComparer.OrdinalIgnoreCase).ToArray();

    public void Register(string method, Func<RpcRequest, Task<object?>> handler)
    {
        _handlers[method] = handler;
    }

    public async Task HandleAsync(ClientConnection conn, RpcRequest req)
    {
        try
        {
            if (!_handlers.TryGetValue(req.method, out var handler))
            {
                await conn.SendResponseAsync(new RpcResponse(req.v, req.id, false, null,
                    new RpcError("rpc.method_not_found", $"Unknown method: {req.method}. Try ml.status", null)));
                return;
            }

            object? result = await handler(req);
            await conn.SendResponseAsync(new RpcResponse(req.v, req.id, true, result, null));
        }
        catch (Exception ex)
        {
            await conn.SendResponseAsync(new RpcResponse(req.v, req.id, false, null,
                new RpcError("rpc.handler_error", ex.Message, ex.StackTrace)));
        }
    }

    public static JsonElement? GetParamsAsJsonElement(RpcRequest req)
    {
        if (req.@params is JsonElement el)
            return el;
        return null;
    }
}
