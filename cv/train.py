import argparse
import os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # 1) Dataset & DataLoader
    dataset = CocoDetection(
        root=os.path.join(args.data_root, "images"),
        annFile=os.path.join(args.data_root, "annotations.json"),
        transform=get_transform()
    )
    coco = dataset.coco
    cat_ids = coco.getCatIds()
    num_classes = max(cat_ids) + 1
    print(f"[INFO] {len(dataset)} images, {num_classes-1} classes (plus background)")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    total_batches = len(data_loader)
    print(f"[INFO] {total_batches} batches per epoch (batch_size={args.batch_size})")

    # 2) Build the model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # 3) Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 4) Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Starting epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(data_loader, start=1):
            images = [img.to(device) for img in images]

            # Reformat COCO annotations → {"boxes": Tensor, "labels": Tensor}
            formatted = []
            for ann_list in targets:
                boxes, labels = [], []
                for ann in ann_list:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])
                if boxes:
                    boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
                    labels = torch.tensor(labels, dtype=torch.int64,   device=device)
                else:
                    boxes  = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    labels = torch.zeros((0,),   dtype=torch.int64,   device=device)
                formatted.append({"boxes": boxes, "labels": labels})

            loss_dict = model(images, formatted)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # log every log_interval batches
            if batch_idx % args.log_interval == 0 or batch_idx == total_batches:
                avg_batch_loss = epoch_loss / batch_idx
                print(f"  [Epoch {epoch}] Batch {batch_idx}/{total_batches}  "
                      f"batch_loss={losses.item():.4f}  avg_loss={avg_batch_loss:.4f}")

        # end of epoch
        avg_loss = epoch_loss / total_batches
        print(f"[INFO] Epoch {epoch} complete — avg_loss: {avg_loss:.4f}")

    # 5) Save weights
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "fasterrcnn_novice.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[INFO] Training complete, model saved to:\n  {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",
                        default="/home/jupyter/novice/cv",
                        help="Root directory containing images/ and annotations.json")
    parser.add_argument("--output_dir",
                        default="/home/jupyter/turings-brainchildren/cv/weights",
                        help="Where to save the trained weights")
    parser.add_argument("--epochs",
                        type=int, default=12,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",
                        type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr",
                        type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--device",
                        default="cuda",
                        help="Device to train on (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--num_workers",
                        type=int, default=0,
                        help="DataLoader workers (0 for no multiprocessing)")
    parser.add_argument("--log_interval",
                        type=int, default=100,
                        help="Print training status every N batches")
    args = parser.parse_args()
    main(args)
