import torch
import torch.jit
import argparse
import onnxruntime as ort
import numpy as np

from src.lightning_module import PlanetModule


def check_model_quality(model_path, scripted_model):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    test_data = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
    scripted_pred = scripted_model(torch.from_numpy(test_data))
    onnx_pred = session.run(None, {'input': test_data})[0]

    max_diff = np.abs(onnx_pred - scripted_pred.cpu().detach().numpy()).max()

    print(f'Max difference: {max_diff}')

    assert max_diff < 1e-5, 'Too different answers!'


def convert_model(input_model: str, output_model: str):
    module = PlanetModule.load_from_checkpoint(input_model, map_location='cpu')

    model = module._model

    model.eval()

    scripted_model = torch.jit.script(model)

    torch.onnx.export(
        model=scripted_model, 
        args=torch.randn(1, 3, 256, 256), 
        f=output_model,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={'input': [0], 'output': [0]}
    )

    check_model_quality(output_model, scripted_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model')
    parser.add_argument('output_model')

    args = parser.parse_args()

    convert_model(args.input_model, args.output_model)
