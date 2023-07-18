import torch
import numpy as np
import torch.nn.functional as F
import configs.asrf_config as asrf_cfg
import sys

sys.path.append('./backbones/asrf')
from libs.postprocess import PostProcessor
from libs.utils import entropy, seg_encode, rollout

# sys.path.append('./backbones/ASFormer')
# from eval import segment_bars_with_confidence


def predict_refiner(model, refine_net, main_backbone_name, model_dir, result_dir, features_path, vid_list_file,
                    epoch, actions_dict, device, sample_rate):

    model.eval()
    refine_net.eval()
    with torch.no_grad():
        model.to(device)
        refine_net.to(device)
        model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        refine_net.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".opt"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)

            if main_backbone_name == 'mstcn':
                mask = torch.ones(input_x.size(), device=device)
                action_pred, _ = model(input_x, mask)
                # _, predicted = torch.max(action_pred[-1].data, 1)  ## source prediction
                predicted_refine = refine_net(F.softmax(action_pred[-1], dim=1)) ##
                _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

            elif main_backbone_name == 'ASFormer':
                # mask = torch.ones(input_x.size(), device=device)
                predictions, _ = model(input_x, torch.ones(input_x.size(), device=device))
                # _, predicted = torch.max(predictions.data, 1)
                # for i in range(len(predictions)):
                #     confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                predicted_refine = refine_net(F.softmax(predictions[-1], dim=1))  ##
                # predicted_refine = refine_net(F.softmax(predictions[-1], dim=1),frames_feats[-1], mask)  ## ASFormer
                _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [
                    list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()

def predict_backbone(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device,
                     sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls, out_bound = model(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(),
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                predicted = refined_output_cls

            elif name == 'mstcn':
                predictions, _ = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)

            elif name == 'ASFormer':
                predictions, _ = model(input_x, torch.ones(input_x.size(), device=device))
                # _, predicted = torch.max(predictions.data, 1)
                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    # confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    #
                    # batch_target = batch_target.squeeze()
                    # confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    #
                    # segment_bars_with_confidence(result_dir + '/{}_stage{}.png'.format(vid, i),
                    #                              confidence.tolist(),
                    #                              batch_target.tolist(), predicted.tolist())
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [
                    list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()




