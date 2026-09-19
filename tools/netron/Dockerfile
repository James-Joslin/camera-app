FROM python:3.12-slim
RUN pip install --no-cache-dir netron==9.2.9
USER 65534:65534
EXPOSE 8080
CMD ["netron", "/models/person_detector_int8.xml", "--host", "0.0.0.0", "--port", "8080"]
