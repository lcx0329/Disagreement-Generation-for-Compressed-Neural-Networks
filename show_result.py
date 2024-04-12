import math
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr_ssim(tensor1, tensor2):
    psnr_value = peak_signal_noise_ratio(tensor1, tensor2, data_range=1.0)
    ssim_value = structural_similarity(tensor1, tensor2, channel_axis=2)
    return psnr_value, ssim_value


def calculate_image_metrics(image_list1, image_list2):
    psnr_list = []
    ssim_list = []

    for img1, img2 in zip(image_list1, image_list2):
        psnr, ssim = calculate_psnr_ssim(img1, img2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    return psnr_list, ssim_list


from toolkit.commons import *
import math


def show_df_results(setting_name):
    args = Args(setting_name)

    dataset = get_dataset(args.dataset, batch_size=args.b)

    large = get_network(args.net, dataset.num_classes, weight=args.weight)
    pmodel = torch.load(args.prune_weight)
    qmodel = TorchQuantizer.load_model(large, args.quant_weight)
    kd1 = get_network(args.kd1, dataset.num_classes, weight=args.kd1_weight)
    kd2 = get_network(args.kd2, dataset.num_classes, weight=args.kd2_weight)

    smalls = {
        "P": pmodel,
        "KD1": kd1,
        "KD2": kd2,
        "Q": qmodel
    }

    all_images = []

    for i, (name, model) in enumerate(smalls.items()):
        device = 'cpu' if name == "Q" else "cuda"
        white_model = model
        black_model = large
        if name == "Q":
            white_model, black_model = black_model, white_model
        large.to(device)
        model.to(device)
        
        same_tensor = SameFinder.find_images(large, model, 1000, dataset.test_loader, device=device, agreement=True)

        
        df = CWDiffinder(white_model, black_model, normalization=MEAN_STDs[setting_name], c=1, steps=50)
        df.set_device(device)
        
        df_imgs = df.find(datasource=same_tensor)
        
        metric = ModelMetric(wrapper.tensor_to_loader(df_imgs, same_tensor[1]))
        metric.device = device
        dr = metric.disagree_rate(model, large)
        

    
        
        W = wrapper.get_plot_wrapper(MENAs[setting_name])
        psnr_values, ssim_values = calculate_image_metrics(W(same_tensor[0]), W(df_imgs))

        print(setting_name)
        print(name)
        filtered_psnr_values = [num for num in psnr_values if not math.isinf(num)]
        print("PSNR values:", sum(filtered_psnr_values) / len(filtered_psnr_values))
        print("SSIM values:", sum(ssim_values) / len(ssim_values))
        print("SR: {}%".format(round(dr * 100, 2)))
        # print("Time: {}s".format(times[i]))
    
    
    

show_df_results("CIFAR100")
show_df_results("CIFAR10")
show_df_results("TINYIMAGENET")