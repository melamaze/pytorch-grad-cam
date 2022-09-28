import numpy as np
import torch
from torch.autograd import Function
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(
                input_img.size()).type_as(input_img),
            input_img,
            positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(
                input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(
                    input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1),
            positive_mask_2)
        return grad_input


class GuidedBackpropReLUasModule(torch.nn.Module):
    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()

    def forward(self, input_img):
        return self.model(input_img)

    def recursive_replace_relu_with_guidedrelu(self, module_top):

        for idx, module in module_top._modules.items():
            self.recursive_replace_relu_with_guidedrelu(module)
            if module.__class__.__name__ == 'ReLU':
                module_top._modules[idx] = GuidedBackpropReLU.apply
        print("b")

    def recursive_replace_guidedrelu_with_relu(self, module_top):
        try:
            for idx, module in module_top._modules.items():
                self.recursive_replace_guidedrelu_with_relu(module)
                if module == GuidedBackpropReLU.apply:
                    module_top._modules[idx] = torch.nn.ReLU()
        except BaseException:
            pass

    def __call__(self, input_img, target_category=None):
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         GuidedBackpropReLUasModule())

        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)
        # 這邊只取 [1] 是因為印出來的 output 看起來是 macnn forward 的 6 個值，[1] 是 cnn_pred
        output = output[1]
        # print("out",output.shape)

        if target_category is None:
            # 原本的應該是只有一張圖，所以可以取最大的index
            # target_category = np.argmax(output.cpu().data.numpy())
            # 現在有 10 張，要一個一個拿出最大的index
            target_category = []
            for o in output:
                target_category.append(np.argmax(o.cpu().data.numpy()))
            # 出現 IndexError: list index out of range加上以下寫法
            # target_category = target_category//256
            # print("target_category",target_category)

        loss = output[0, target_category]

        # 出現 RuntimeError: grad can be implicitly created only for scalar outputs
        # loss.backward(retain_graph=True)
        loss.backward(torch.ones_like(loss), retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))

        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())

        return output
