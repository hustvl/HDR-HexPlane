import torch
import torch.nn as nn
class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.params =  torch.nn.Parameter(torch.Tensor([1,0]))
    def forward(self, x):
        return 1/(1+torch.exp(self.params[0]*(x+self.params[1])))
class ToneMapper(torch.nn.Module):
    def __init__(self, hidden=64):
        super(ToneMapper, self).__init__()
        # self.Linear_r = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # self.Linear_g = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # self.Linear_b = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # self.Linear_r = Sigmoid()
        # self.Linear_g = Sigmoid()
        # self.Linear_b = Sigmoid()
        self.Linear_r = torch.nn.Sigmoid()
        self.Linear_g = torch.nn.Sigmoid()
        self.Linear_b = torch.nn.Sigmoid()
    def forward(self, rgb_h_ln, exps, voxelfeature=None):
        # 尝试增加一个coordinate-based query。

        # exposure_ln = self.exposure_linear(exps)
        exposure_ln = exps
        r_ln = rgb_h_ln[:, :, 0:1] + exposure_ln#[:, :, 0:1]
        g_ln = rgb_h_ln[:, :, 1:2] + exposure_ln#[:, :, 1:2]
        b_ln = rgb_h_ln[:, :, 2:3] + exposure_ln#[:, :, 2:3]
        r_l = self.Linear_r(r_ln)
        g_l = self.Linear_g(g_ln)
        b_l = self.Linear_b(b_ln)
        # r_l = r_ln
        # g_l = g_ln
        # b_l = b_ln
        rgb_l = torch.cat([r_l, g_l, b_l], -1)
        # rgb_l = torch.sigmoid(rgb_l)
        # rgb_l = torch.relu(rgb_l)
        return rgb_l

    # def zero_point_contraint(self):

    def zero_point_contraint(self, gt, query_point):
        ln_x = query_point
        ln_y = query_point
        ln_z = query_point
        x_l = self.Linear_r(ln_x)
        y_l = self.Linear_g(ln_y)
        z_l = self.Linear_b(ln_z)

        results = torch.sigmoid(torch.cat([x_l, y_l, z_l], -1)).squeeze()
        return torch.mean((results - gt) ** 2)
    def mask_weight(self,x):
        low = 0.05
        high = 0.9
        
        lower = torch.where( x < low)
        higher = torch.where( x > high)
        weight = torch.ones_like(x).to(x.device)
        weight[lower] = ((weight[lower]+low)/2*low)**2
        weight[higher] = ((2-weight[higher]/2*(1-high))**2)
        return weight

    @torch.no_grad()
    def export_mapping_function(self):
        N = 30

        exps = torch.arange(-N, N, N / 100).unsqueeze(1)
        r_l = self.Linear_r(exps)
        g_l = self.Linear_g(exps)
        b_l = self.Linear_b(exps)
        rgb_l = torch.cat([r_l, g_l, b_l], -1)
        rgb_l = torch.sigmoid(rgb_l)
        return exps, rgb_l

