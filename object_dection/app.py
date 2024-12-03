import cv2
import numpy as np
import tensorflow.lite as tflite

class ObjectDetector:
    def __init__(self, model_path="D:\object_dection\object_dection\yolov8x.pt", 
                 labels_path="d:/object_dection/object_dection/labels.txt"):
        # Initialize TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def preprocess_image(self, image):
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1:3]
        
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize and preprocess
        image = cv2.resize(image, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def detect(self, image, conf_threshold=0.25):
        # Preprocess and run inference
        input_data = self.preprocess_image(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Process detections
        boxes, class_ids, scores = [], [], []
        img_height, img_width = image.shape[:2]

        for detection in output_data:
            score = float(detection[4] * detection[5:].max())
            if score > conf_threshold:
                class_id = int(np.argmax(detection[5:]))
                x_center, y_center, w, h = detection[:4]
                
                # Convert to pixel coordinates
                x1 = int((x_center - w/2) * img_width)
                y1 = int((y_center - h/2) * img_height)
                x2 = int((x_center + w/2) * img_width)
                y2 = int((y_center + h/2) * img_height)
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
                scores.append(score)

        # Apply NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
            if len(indices) > 0:
                indices = indices.flatten()
                return np.array(boxes)[indices], np.array(class_ids)[indices], np.array(scores)[indices]
        
        return np.array([]), np.array([]), np.array([])

    def draw_detections(self, image, boxes, class_ids, scores):
        for box, class_id, score in zip(boxes, class_ids, scores):
            # Draw box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{self.labels[class_id]}: {score:.2f}"
            cv2.putText(image, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw
        boxes, classes, scores = detector.detect(frame)
        frame = detector.draw_detections(frame, boxes, classes, scores)
        
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
