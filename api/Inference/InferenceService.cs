using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

namespace CameraSoftware.Api.Inference;

public sealed class InferenceService : IDisposable
{
    private sealed class Worker(IntPtr handle)
    {
        public readonly IntPtr Handle = handle;
        public readonly float[] Boxes = new float[500];
        public readonly int[] Dimensions = new int[2];
        public readonly double[] Stages = new double[4];
        public readonly StringBuilder Error = new(2048);
    }
    private readonly ConcurrentQueue<Worker> workers = new();
    private readonly SemaphoreSlim slots;
    private readonly int capacity, queueLimit, topK, threads;
    private readonly ILogger<InferenceService> logger;
    private readonly object initialization = new();
    private IntPtr engine;
    private int admitted;
    private string? error;
    public string ModelDirectory { get; }
    public string ModelPath { get; private set; }
    public string? ReleaseId { get; private set; }
    public string? ReleaseStatus { get; private set; }
    private readonly IConfiguration config;
    public ModelRefreshStatus Refresh { get; set; } = new(null, null, null);
    public int MaxUploadBytes { get; }

    public InferenceService(IConfiguration config, ILogger<InferenceService> logger)
    {
        this.logger = logger;
        this.config = config;
        ModelDirectory = config["MODEL_DIR"] ?? "/models";
        ModelPath = config["MODEL_PATH"] ?? Path.Combine(ModelDirectory, "person_detector_int8.xml");
        capacity = ReadInt(config, "INFERENCE_REQUESTS", 1, 1, 32);
        queueLimit = ReadInt(config, "INFERENCE_QUEUE_LIMIT", 2, 0, 128);
        topK = ReadInt(config, "MODEL_PRE_NMS_TOPK", 1000, 1, 100000);
        threads = ReadInt(config, "OPENVINO_THREADS", 1, 1, 256);
        MaxUploadBytes = ReadInt(config, "INFERENCE_MAX_UPLOAD_BYTES", 20*1024*1024, 1, 100*1024*1024);
        slots = new SemaphoreSlim(capacity, capacity);
    }
    private static int ReadInt(IConfiguration config, string key, int fallback, int min, int max)
    {
        var value = config.GetValue<int?>(key) ?? fallback;
        return value >= min && value <= max ? value : throw new ArgumentOutOfRangeException(key);
    }
    public object Status()
    {
        lock (initialization) return new {
            ready = (engine != IntPtr.Zero || (File.Exists(ModelPath) && File.Exists(Path.ChangeExtension(ModelPath,".bin")))) && error is null,
            loaded = engine != IntPtr.Zero, model = Path.GetFileName(ModelPath), device = "CPU", error,
            backend = "csharp-openvino", requests = capacity, queueLimit,
            releaseId = ReleaseId, releaseStatus = ReleaseStatus, refresh = Refresh
        };
    }

    // Compile separately while the old model keeps serving. Drain all worker slots
    // before transferring ownership so no in-flight native handle can be destroyed.
    public async Task ReplaceModelAsync(string path, string releaseId, string releaseStatus, CancellationToken token)
    {
        var candidateConfig = new ConfigurationBuilder().AddConfiguration(config)
            .AddInMemoryCollection(new Dictionary<string, string?> { ["MODEL_PATH"] = path }).Build();
        using var candidate = new InferenceService(candidateConfig, logger);
        await Task.Run(() => {
            candidate.EnsureLoaded();
            if (!candidate.workers.TryPeek(out var probe)) throw new InvalidOperationException("Candidate has no worker");
            // A tiny PPM exercises decode, preprocessing, inference and postprocessing
            // before the candidate is allowed to replace the working release.
            byte[] image = [.. Encoding.ASCII.GetBytes("P6\n1 1\n255\n"), 0, 0, 0];
            if (NativeDetector.detector_predict(probe.Handle, image, image.Length, .5f, .45f, topK,
                    probe.Boxes, 100, probe.Dimensions, probe.Stages, probe.Error, probe.Error.Capacity) < 0)
                throw new InvalidOperationException("Candidate inference validation failed: " + probe.Error);
        }, token);
        int acquired = 0;
        try
        {
            for (; acquired < capacity; acquired++) await slots.WaitAsync(token);
            token.ThrowIfCancellationRequested();
            lock (initialization)
            {
                ReleaseNative();
                engine = candidate.engine;
                candidate.engine = IntPtr.Zero;
                while (candidate.workers.TryDequeue(out var worker)) workers.Enqueue(worker);
                ModelPath = path;
                ReleaseId = releaseId;
                ReleaseStatus = releaseStatus;
                error = null;
            }
        }
        finally { if (acquired > 0) slots.Release(acquired); }
    }
    private void EnsureLoaded()
    {
        lock (initialization)
        {
            if (engine != IntPtr.Zero) return;
            try
            {
                var message = new StringBuilder(2048);
                engine = NativeDetector.detector_create(ModelPath, threads, message, message.Capacity);
                if (engine == IntPtr.Zero) throw new InvalidOperationException(message.ToString());
                for (var i=0; i<capacity; i++)
                {
                    var handle = NativeDetector.detector_worker_create(engine, message, message.Capacity);
                    if (handle == IntPtr.Zero) throw new InvalidOperationException(message.ToString());
                    workers.Enqueue(new Worker(handle));
                }
                error = null;
            }
            catch (Exception ex)
            {
                ReleaseNative();
                error = "Model initialization failed. Check API logs and model artifacts.";
                logger.LogError(ex, "Unable to initialize inference model {Model}", ModelPath);
                throw new InvalidOperationException(error, ex);
            }
        }
    }
    public async Task<IResult> PredictAsync(HttpRequest request, float threshold, CancellationToken token)
    {
        var total = Stopwatch.StartNew();
        if (!float.IsFinite(threshold) || threshold < .01f || threshold > .99f)
            return Results.BadRequest(new { detail = "threshold must be between 0.01 and 0.99" });
        if (!request.HasFormContentType)
            return Results.BadRequest(new { detail = "Expected a multipart image upload" });
        if (request.ContentLength > MaxUploadBytes + 1024L*1024)
            return Results.Json(new { detail = "Image upload exceeds the size limit" }, statusCode:413);
        if (Interlocked.Increment(ref admitted) > capacity + queueLimit)
        {
            Interlocked.Decrement(ref admitted);
            return Results.Json(new { detail = "Inference queue is full; retry with a newer frame" }, statusCode:429);
        }
        bool acquired = false;
        byte[]? buffer = null;
        Worker? worker = null;
        try
        {
            var watch = Stopwatch.StartNew();
            await slots.WaitAsync(token); acquired = true;
            double queueMs = watch.Elapsed.TotalMilliseconds;
            watch.Restart();
            // ASP.NET executes synchronous native work on a thread-pool thread;
            // there is no Python event-loop thread or unbounded task queue.
            EnsureLoaded();
            double initializationMs = watch.Elapsed.TotalMilliseconds;
            workers.TryDequeue(out worker);
            if (worker is null) throw new InvalidOperationException("Inference worker unavailable");
            watch.Restart();
            var form = await request.ReadFormAsync(token);
            var image = form.Files.GetFile("image");
            if (image is null) return Results.BadRequest(new { detail = "An image file is required" });
            if (image.Length < 1 || image.Length > MaxUploadBytes)
                return Results.Json(new { detail = "Image upload is empty or exceeds the size limit" }, statusCode:413);
            int length = checked((int)image.Length);
            buffer = ArrayPool<byte>.Shared.Rent(length);
            await using (var source = image.OpenReadStream())
                await source.ReadExactlyAsync(buffer.AsMemory(0,length), token);
            double readMs = watch.Elapsed.TotalMilliseconds;
            token.ThrowIfCancellationRequested();
            worker.Error.Clear();
            int count = NativeDetector.detector_predict(worker.Handle,buffer,length,threshold,.45f,topK,
                worker.Boxes,100,worker.Dimensions,worker.Stages,worker.Error,worker.Error.Capacity);
            if (count == -2) return Results.BadRequest(new { detail = "The uploaded file is not a supported image." });
            if (count < 0) throw new InvalidOperationException(worker.Error.ToString());
            var detections = Enumerable.Range(0,count).Select(i => new {
                label = "person", classId = 1, confidence = worker.Boxes[5*i+4],
                box = new[] { (int)worker.Boxes[5*i], (int)worker.Boxes[5*i+1],
                              (int)worker.Boxes[5*i+2], (int)worker.Boxes[5*i+3] }
            }).ToArray();
            return Results.Ok(new {
                model = Path.GetFileName(ModelPath), releaseId = ReleaseId, image = new { width = worker.Dimensions[0], height = worker.Dimensions[1] },
                inferenceMs = Math.Round(worker.Stages[2],2), detections,
                timings = new { queueMs, initializationMs, readMs, decodeMs = worker.Stages[0],
                    preprocessMs = worker.Stages[1], inferenceMs = worker.Stages[2],
                    postprocessMs = worker.Stages[3], processingMs = total.Elapsed.TotalMilliseconds }
            });
        }
        catch (InvalidDataException)
        {
            return Results.BadRequest(new { detail = "Invalid or oversized multipart upload" });
        }
        catch (OperationCanceledException) when (token.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            logger.LogError(ex, "Inference request failed");
            return Results.Json(new { detail = "Inference failed. Check model readiness and API logs." }, statusCode:503);
        }
        finally
        {
            if (buffer is not null) ArrayPool<byte>.Shared.Return(buffer);
            if (worker is not null) workers.Enqueue(worker);
            if (acquired) slots.Release();
            Interlocked.Decrement(ref admitted);
        }
    }
    private void ReleaseNative()
    {
        while (workers.TryDequeue(out var worker)) NativeDetector.detector_worker_destroy(worker.Handle);
        if (engine != IntPtr.Zero) NativeDetector.detector_destroy(engine);
        engine = IntPtr.Zero;
    }
    public void Dispose() { ReleaseNative(); slots.Dispose(); }
}
