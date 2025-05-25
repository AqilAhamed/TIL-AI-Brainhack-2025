import os
import io
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from pycocotools.coco import COCO

class CVManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths from environment variables or defaults
        ann_path = os.getenv("ANNOTATIONS_PATH", "/app/annotations.json")
        weights_path = os.getenv("WEIGHTS_PATH", "/app/weights/fasterrcnn_novice.pth")
        resnet_weights_path = os.getenv("RESNET_WEIGHTS_PATH", "/app/resnet50-0676ba61.pth")

        # Load COCO categories
        coco = COCO(ann_path)
        #num_classes = len(coco.getCatIds()) + 1  # +1 for background
        num_classes = 18

        # Load ResNet50 classification weights and filter out fc layers
        resnet_state_dict = torch.load(resnet_weights_path, map_location=self.device)
        filtered_state_dict = {k: v for k, v in resnet_state_dict.items() if not k.startswith("fc.")}

        # Create backbone and load filtered weights
        backbone = resnet_fpn_backbone('resnet50', weights=None)
        backbone.body.load_state_dict(filtered_state_dict)

        # Create detection model using the custom backbone
        model = FasterRCNN(backbone, num_classes=num_classes)

        # Load fine-tuned Faster R-CNN weights
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)

        # Finalize model
        model.to(self.device).eval()
        self.model = model
        self.transform = transforms.Compose([transforms.ToTensor()])

    def cv(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            out = self.model([tensor])[0]

        results = []
        boxes  = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        for (x1, y1, x2, y2), lbl, scr in zip(boxes, labels, scores):
            if scr < 0.05:
                continue
            results.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_id": int(lbl)
            })
        return results
