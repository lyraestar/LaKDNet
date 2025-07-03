import os
import torch
import torchvision.utils as vutils
from util.util import read_image, crop_image
from pathlib import Path
from models.LaKDNet import LaKDNet
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description='Single Image Deblurring Test')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default='./Results_single', help='Directory to save the output image.')
    parser.add_argument('--model_type', type=str, required=True, choices=['Defocus', 'Motion'], help='Task type: Defocus or Motion.')
    parser.add_argument('--config_name', type=str, required=True, help='Specific configuration name from Test_configs.yml (e.g., train_on_dpdd_s).')
    parser.add_argument('--left_image_path', type=str, default=None, help='Path to the left view image (required for dual mode).')
    parser.add_argument('--right_image_path', type=str, default=None, help='Path to the right view image (required for dual mode).')

    args = parser.parse_args()

    # Load configuration
    config_file = './options/Test_configs.yml'
    with open(config_file, 'r') as file:
        config_all = yaml.safe_load(file)

    if args.model_type not in config_all:
        print(f"Error: Model type '{args.model_type}' not found in {config_file}")
        return

    config = config_all[args.model_type]

    if args.config_name not in config['weight']:
        print(f"Error: Config name '{args.config_name}' not found in weights for model type '{args.model_type}'.")
        print(f"Available configs: {list(config['weight'].keys())}")
        return

    net_weight_path = config['weight'][args.config_name]

    # Determine net_config_key by finding which net_configs entry corresponds to the config_name's index in test_status
    try:
        config_idx = config['test_status'].index(args.config_name)
        net_config_key = config['net_configs'][config_idx]
        net_config_params = config[net_config_key]
    except ValueError:
        print(f"Error: Config name '{args.config_name}' not found in 'test_status' for model type '{args.model_type}'.")
        print(f"Available in test_status: {config['test_status']}")
        return
    except IndexError:
        print(f"Error: Index out of bounds when trying to find net_config for '{args.config_name}'. Mismatch between 'test_status' and 'net_configs' lengths.")
        return


    is_dual_mode = 'dual' in args.config_name.lower() or (net_config_params.get('dual_pixel_task', False))


    if is_dual_mode and (not args.left_image_path or not args.right_image_path):
        print("Error: For dual mode, --left_image_path and --right_image_path are required.")
        return
    if not os.path.isfile(args.input_path):
        print(f"Error: Input image path '{args.input_path}' does not exist.")
        return
    if is_dual_mode:
        if not os.path.isfile(args.left_image_path):
            print(f"Error: Left image path '{args.left_image_path}' does not exist.")
            return
        if not os.path.isfile(args.right_image_path):
            print(f"Error: Right image path '{args.right_image_path}' does not exist.")
            return

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Image pre-processing
    # In run.py, DPDD (dual) uses 65535.0, others use 255.0
    # We'll assume dual mode implies 65535.0 for now, this might need refinement
    # based on specific dataset characteristics if non-dual datasets also use 16-bit.
    # For DPDD (which is dual), it's 65535.0. Let's check if config_name contains 'dpdd' for 16-bit for now.
    # A more robust way would be to have this in the config file per dataset.
    divisor = 65535.0 if 'dpdd' in args.config_name.lower() and is_dual_mode else 255.0

    print(f"Reading input image: {args.input_path}")
    C = read_image(args.input_path, divisor)
    C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
    C, h, w = crop_image(C, 8, return_hw=True) # pad=8 as in run.py

    if is_dual_mode:
        print(f"Reading left image: {args.left_image_path}")
        L = read_image(args.left_image_path, divisor)
        L = torch.FloatTensor(L.transpose(0, 3, 1, 2).copy()).cuda()
        L = crop_image(L, 8) # Assuming same padding

        print(f"Reading right image: {args.right_image_path}")
        R = read_image(args.right_image_path, divisor)
        R = torch.FloatTensor(R.transpose(0, 3, 1, 2).copy()).cuda()
        R = crop_image(R, 8) # Assuming same padding

    # Model loading and inference
    print(f"Loading model with config: {net_config_key}")
    print(f"Model parameters: {net_config_params}")
    print(f"Model weights: {net_weight_path}")

    network = LaKDNet(**net_config_params).cuda()
    network.load_state_dict(torch.load(net_weight_path))
    network.eval()

    with torch.no_grad():
        if not is_dual_mode:
            output = network(C)
        else:
            input_tensor = torch.cat([L, R, C], 1).cuda()
            output = network(input_tensor)

    # Post-processing and saving
    output = output[:, :, :h, :w] # Crop to original size before padding

    base_filename = os.path.basename(args.input_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_deblurred{ext}"
    save_file_path = os.path.join(args.output_dir, output_filename)

    vutils.save_image(output, save_file_path, nrow=1, padding=0, normalize=False)
    print(f"Output image saved to: {save_file_path}")

if __name__ == '__main__':
    main()
