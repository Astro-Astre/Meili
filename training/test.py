from astre_utils.img.prepare import *
from data.utils import *
from astre_utils.utils import *


class Test:
    def __init__(self, model_path, mode):
        init_rand_seed(1926)
        self.mode = mode
        self.model = torch.load(model_path)
        self.device_ids = [0, 1]
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        for param in self.model.parameters():
            param.requires_grad = False
        assert self.mode in ["BCNN", "FCNN"], "only support Bayesian CNN and Frequency CNN"
        if self.mode == "BCNN":
            enable_dropout(self.model)
        self.model.eval()

    def predict(self, row, T=100):
        x = torch.from_numpy(AstroImg(row.split(" ")[0]).load().astype(np.float32))
        y = self.model(x.to("cuda:0").unsqueeze(0))
        if self.mode == "FCNN":
            pred = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
            return pred
        if self.mode == "BCNN":
            output_list = []
            for i in range(T):
                output_list.append(torch.unsqueeze(torch.Tensor(
                    answer_prob(self.model(x.to("cuda:0").unsqueeze(0)).data.cpu().numpy()[0, :])), 0).numpy())
            mean = np.mean(np.array(output_list), axis=0)[0, 0, :]
            variance = np.var(np.array(output_list), axis=0)[0, 0, :]
            return mean, variance
