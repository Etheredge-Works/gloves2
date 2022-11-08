from model import SiameseNet
import torch
import onnx
# import onnx.optimizer


if __name__ == "__main__":
    model = SiameseNet.load_from_checkpoint("checkpoints/best_model-v9.ckpt")
    model.eval()
    input_sample = torch.randn(1, 3, 224, 224)
    input_sample = input_sample, input_sample
    model.to_onnx("model.onnx", input_sample, export_params=True)

    # model = onnx.load("model.onnx")
    # model = onnx.optimizer.optimize(model)
    # onnx.save(model, "model2.onnx")