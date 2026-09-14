#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void error(char* out, int capacity, const char* message) {
    if (capacity > 0) { std::strncpy(out, message, capacity - 1); out[capacity - 1] = 0; }
}
struct Box { float x1, y1, x2, y2, score; size_t index; };
struct Engine {
    ov::CompiledModel model;
    int height, width, logits_port = -1, boxes_port = -1;
    explicit Engine(const char* path, int threads) {
        cv::setNumThreads(1);
        ov::Core core;
        auto graph = core.read_model(path);
        if (graph->inputs().size() != 1 || graph->outputs().size() != 2)
            throw std::runtime_error("Expected one input and two detector outputs");
        auto shape = graph->input().get_shape();
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[3] <= shape[2] ||
            graph->input().get_element_type() != ov::element::f32)
            throw std::runtime_error("Expected batch-one FP32 NCHW landscape input");
        height = int(shape[2]); width = int(shape[3]);
        size_t points = 0;
        for (size_t i = 0; i < 2; ++i) {
            auto port = graph->output(i); auto dims = port.get_shape();
            if (dims.size() != 3 || dims[0] != 1 || port.get_element_type() != ov::element::f32)
                throw std::runtime_error("Unexpected detector output shape/type");
            if (dims[2] == 1) { logits_port = int(i); points = dims[1]; }
            if (dims[2] == 4 && port.get_names().count("boxes_xyxy_pixels")) boxes_port = int(i);
        }
        if (logits_port < 0 || boxes_port < 0 || graph->output(boxes_port).get_shape()[1] != points)
            throw std::runtime_error("Requires clean_ltrb decoded boxes_xyxy_pixels and one person logit");
        model = core.compile_model(graph, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::num_streams(1), ov::inference_num_threads(threads), ov::hint::inference_precision(ov::element::f32));
    }
};
struct Worker {
    Engine* engine;
    ov::InferRequest request;
    cv::Mat decoded, resized;
    std::vector<Box> boxes, kept;
    explicit Worker(Engine* e): engine(e), request(e->model.create_infer_request()) {}
};
float overlap(const Box& a, const Box& b) {
    float intersection = std::max(0.f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                         std::max(0.f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    return intersection / ((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-intersection+1e-6f);
}
}
// C ABI: exceptions never cross the managed/native boundary. Handles have explicit ownership.
extern "C" {
void* detector_create(const char* path, int threads, char* err, int cap) {
    try { return new Engine(path, threads); } catch (const std::exception& e) { error(err,cap,e.what()); return nullptr; }
}
void detector_destroy(void* p) { delete static_cast<Engine*>(p); }
void* detector_worker_create(void* engine, char* err, int cap) {
    try { return new Worker(static_cast<Engine*>(engine)); } catch (const std::exception& e) { error(err,cap,e.what()); return nullptr; }
}
void detector_worker_destroy(void* p) { delete static_cast<Worker*>(p); }
// Output rows: x1,y1,x2,y2,confidence. Stages: decode, preprocess, inference, postprocess.
// Return -2 for malformed image, -1 for runtime failure, otherwise detection count.
int detector_predict(void* handle, const unsigned char* encoded, int length, float threshold,
                     float nms, int topk, float* output, int limit, int* dimensions,
                     double* stages, char* err, int cap) {
    try {
        auto& w = *static_cast<Worker*>(handle); auto& e = *w.engine;
        auto start = Clock::now();
        try {
            // Do not use the dst overload: OpenCV 4.6 can retain the previous
            // image on malformed input. Never infer on a stale frame.
            w.decoded = cv::imdecode(cv::Mat(1,length,CV_8U,const_cast<unsigned char*>(encoded)), cv::IMREAD_COLOR);
        } catch (const cv::Exception&) { error(err,cap,"Invalid encoded image"); return -2; }
        if (w.decoded.empty()) { error(err,cap,"Invalid encoded image"); return -2; }
        dimensions[0] = w.decoded.cols; dimensions[1] = w.decoded.rows;
        stages[0] = ms(start); start = Clock::now();
        double scale = std::min(double(e.width)/w.decoded.cols, double(e.height)/w.decoded.rows);
        int rw = std::max(1,int(std::nearbyint(w.decoded.cols*scale)));
        int rh = std::max(1,int(std::nearbyint(w.decoded.rows*scale)));
        int px = (e.width-rw)/2, py = (e.height-rh)/2;
        cv::resize(w.decoded,w.resized,cv::Size(rw,rh),0,0,cv::INTER_LINEAR);
        // Write straight into persistent OpenVINO input storage. No RGB image,
        // float canvas, normalization temporaries or interop input-tensor copy.
        float* tensor = w.request.get_input_tensor().data<float>();
        constexpr float mean[] = {.485f,.456f,.406f}, stddev[] = {.229f,.224f,.225f};
        size_t plane = size_t(e.width)*e.height;
        for (int c=0;c<3;++c) {
            float* dest = tensor+c*plane;
            std::fill(dest,dest+plane,-mean[c]/stddev[c]);
            for (int y=0;y<rh;++y) {
                const auto* row = w.resized.ptr<unsigned char>(y);
                for (int x=0;x<rw;++x)
                    dest[size_t(y+py)*e.width+x+px] = (float(row[3*x+2-c])/255.f-mean[c])/stddev[c];
            }
        }
        stages[1] = ms(start); start = Clock::now();
        w.request.infer();
        stages[2] = ms(start); start = Clock::now();
        auto logits = w.request.get_output_tensor(e.logits_port);
        const float* scores = logits.data<float>();
        const float* coords = w.request.get_output_tensor(e.boxes_port).data<float>();
        w.boxes.clear(); w.kept.clear();
        for (size_t i=0;i<logits.get_shape()[1];++i) {
            float score = 1.f/(1.f+std::exp(-std::clamp(scores[i],-80.f,80.f)));
            if (score >= threshold) w.boxes.push_back({coords[4*i],coords[4*i+1],coords[4*i+2],coords[4*i+3],score,i});
        }
        auto order = [](const Box& a,const Box& b) { return a.score == b.score ? a.index < b.index : a.score > b.score; };
        if (w.boxes.size() > size_t(topk)) {
            std::nth_element(w.boxes.begin(),w.boxes.begin()+topk,w.boxes.end(),order);
            w.boxes.resize(topk);
        }
        std::sort(w.boxes.begin(),w.boxes.end(),order);
        for (auto b : w.boxes) {
            if (!std::isfinite(b.x1)||!std::isfinite(b.y1)||!std::isfinite(b.x2)||!std::isfinite(b.y2)) continue;
            b.x1=std::clamp(b.x1,0.f,float(e.width)); b.x2=std::clamp(b.x2,0.f,float(e.width));
            b.y1=std::clamp(b.y1,0.f,float(e.height)); b.y2=std::clamp(b.y2,0.f,float(e.height));
            if (b.x2<=b.x1 || b.y2<=b.y1) continue;
            if (std::any_of(w.kept.begin(),w.kept.end(),[&](const Box& k){return overlap(b,k)>nms;})) continue;
            w.kept.push_back(b);
            if (w.kept.size() == size_t(limit)) break;
        }
        for (size_t i=0;i<w.kept.size();++i) {
            const auto& b=w.kept[i]; float* row=output+5*i;
            row[0]=float(std::clamp((b.x1-px)/scale,0.,double(w.decoded.cols)));
            row[1]=float(std::clamp((b.y1-py)/scale,0.,double(w.decoded.rows)));
            row[2]=float(std::clamp((b.x2-px)/scale,0.,double(w.decoded.cols)));
            row[3]=float(std::clamp((b.y2-py)/scale,0.,double(w.decoded.rows))); row[4]=b.score;
        }
        stages[3]=ms(start); return int(w.kept.size());
    } catch (const std::exception& e) { error(err,cap,e.what()); return -1; }
}
}
