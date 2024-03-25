import os.path
import logging
from copy import deepcopy

import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # # RFDN baseline, AIM 2020 Efficient SR Challenge winner
        # from models.team00_RFDN import RFDN
        # name, data_range = f"{model_id:02}_RFDN_baseline", 255.0
        # model_path = os.path.join('model_zoo', 'team00_rfdn.pth')
        # model = RFDN()
        # model.load_state_dict(torch.load(model_path), strict=True)

        # # SwinIR baseline, ICCVW 2021
        from models.team14_HGD import HAT
        name, data_range = f"{model_id:02}_SwinIR_baseline", 1.0
        model_path = os.path.join('model_zoo', 'team14_ensemble3_HAT.pth')
        model=HAT()
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
        param_key = 'params_ema'
        load_net = load_net[param_key]

    elif model_id == 14:
        from models.team14_HGD import EnsembleModel
        name, data_range = f"{model_id:02}_HGD", 1.0
        model_path = [
            os.path.join('model_zoo', 'DAT_x4.pth'),
            os.path.join('model_zoo', 'sr_grl_base_c3x4.ckpt'),
            os.path.join('model_zoo', 'team14_ensemble3_HAT.pth')]
        model = EnsembleModel()
        model.load_weights(model_path)
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    """
    # Important !!!
    # ALL our models's inputs are whole image in the test stage,
    # With limited GPU memory, you can break it down by forward each model, 
    # then call model_ensemble to get reproduced results.
    #
    # Besides, you can set tile=256 and overlap=64 to run in limited GPU memory.
    """
    tile = None
    # tile = 256
    ############################readme
    for k, v in model.named_parameters():
        v.requires_grad = False
    ############################
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    if mode == "test":
        path = [
            (
                os.path.join(data_dir, f"DIV2K_test_LR/{i:04}.png"),
                os.path.join(data_dir, f"DIV2K_test_HR/{i:04}.png")
            ) for i in range(901, 1001)
        ]
        # [f"DIV2K_test_LR/{i:04}.png" for i in range(901, 1001)]
    elif mode == "valid":
        path = [
            (
                # os.path.join(data_dir, f"DIV2K_valid_LR/{i:04}x4.png"), #here
                os.path.join(data_dir, f"{i:04}x4.png"),
                os.path.join(data_dir, f"{i:04}.png").replace("LR","HR")
                # os.path.join(data_dir, f"DIV2K_valid_LR_bicubic/X4/{i:04}x4.png"),
                # os.path.join(data_dir, f"DIV2K_valid_HR/{i:04}.png")
            ) for i in range(801, 901)
        ]
    elif mode == "hybrid_test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "LSDIR_DIV2K_test_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    return path


def forward(img_lq, model, tile=None, tile_overlap=64, scale=4):
    if tile is None:
        # test the image as a whole
        print(img_lq.shape)
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    logger.info("{:>16s} : {:<.2f}".format("Average PSNR", results[f"{mode}_ave_psnr"]))
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
        logger.info("{:>16s} : {:<.4f}".format("Average SSIM", results[f"{mode}_ave_ssim"]))
    logger.info("{:>16s} : {:<.3f}".format("Max Memery", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2024-ImageSRx4", log_path="NTIRE2024-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2024-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        if args.hybrid_test:
            # inference on the DIV2K and LSDIR test set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="hybrid_test")
            # record PSNR, runtime
            results[model_name] = valid_results
        else:
            print('inference on the validation set')
            # inference on the validation set
            valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")

            # record PSNR, runtime
            results[model_name] = valid_results

            if args.include_test:
                # inference on the test set
                test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
                results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        flops = get_model_flops(model, input_dim, False)
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Mem [M]")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Mem [M]")
    for k, v in results.items():
        # print(v.keys())
        if args.hybrid_test:
            val_psnr = f"{v['hybrid_test_ave_psnr']:2.2f}"
            val_time = f"{v['hybrid_test_ave_runtime']:3.2f}"
            mem = f"{v['hybrid_test_memory']:2.2f}"
        else:
            val_psnr = f"{v['valid_ave_psnr']:2.2f}"
            val_time = f"{v['valid_ave_runtime']:3.2f}"
            mem = f"{v['valid_memory']:2.2f}"
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, mem)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, mem)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2024-ImageSRx4")
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--save_dir", default="NTIRE2024_ImageSR_x4/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K test set")
    parser.add_argument("--hybrid_test", action="store_true", help="Hybrid test on DIV2K and LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
