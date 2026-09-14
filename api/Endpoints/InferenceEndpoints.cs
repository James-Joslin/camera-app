using CameraSoftware.Api.Inference;

namespace CameraSoftware.Api.Endpoints;

public static class InferenceEndpoints
{
    public static void MapInferenceEndpoints(this WebApplication app)
    {
        app.MapGet("/api/inference/status", (InferenceService service) => service.Status());
        app.MapGet("/api/camera/models", (InferenceService service) => new {
            modelDirectory = service.ModelDirectory,
            models = Directory.Exists(service.ModelDirectory)
                ? Directory.GetFiles(service.ModelDirectory).Select(Path.GetFileName).Order().ToArray() : [],
            active = service.Status()
        });
        app.MapPost("/api/inference/detect", (HttpRequest request, InferenceService service,
            CancellationToken token, float threshold = .5f) => service.PredictAsync(request, threshold, token))
            .DisableAntiforgery(); // Preserve the existing unauthenticated multipart endpoint contract.
    }
}
