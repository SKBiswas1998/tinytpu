"""Object detection wrapper."""
class ObjectDetector:
    """YOLO-based object detector. Uses Model internally."""
    def __init__(self, model_name="yolov8n", **kwargs):
        from tinytpu.inference.model_zoo import Model
        self._model = Model(model_name, **kwargs)

    def detect(self, frame):
        return self._model.predict(frame)
