using System.Runtime.InteropServices;
using System.Text;

namespace CameraSoftware.Api.Inference;

internal static class NativeDetector
{
    private const string Library = "camera_inference";
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr detector_create([MarshalAs(UnmanagedType.LPUTF8Str)] string path, int threads, StringBuilder error, int capacity);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void detector_destroy(IntPtr engine);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr detector_worker_create(IntPtr engine, StringBuilder error, int capacity);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void detector_worker_destroy(IntPtr worker);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int detector_predict(IntPtr worker, byte[] encoded, int length, float threshold,
        float nms, int topk, [Out] float[] output, int limit, [Out] int[] dimensions,
        [Out] double[] stages, StringBuilder error, int capacity);
}
