import numpy as np
import torch
from scipy import interpolate


def relative_position_bias(model, checkpoint_model, key):
    if "relative_position_bias_table" in key:
        rel_pos_bias = checkpoint_model[key]
        src_num_pos, num_attn_heads = rel_pos_bias.size()
        dst_num_pos, _ = model.state_dict()[key].size()
        dst_patch_shape = model.patch_embed.patch_shape
        if dst_patch_shape[0] != dst_patch_shape[1]:
            raise NotImplementedError()
        num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
        if src_size != dst_size:
            print("Position interpolate for %s from %dx%d to %dx%d" % (
                key, src_size, src_size, dst_size, dst_size))
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

            def geometric_progression(a, r, n):
                return a * (1.0 - r ** n) / (1.0 - r)

            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src_size // 2)
                if gp > dst_size // 2:
                    right = q
                else:
                    left = q

            # if q > 1.090307:
            #     q = 1.090307

            dis = []
            cur = 1
            for i in range(src_size // 2):
                dis.append(cur)
                cur += q ** (i + 1)

            r_ids = [-_ for _ in reversed(dis)]

            x = r_ids + [0] + dis
            y = r_ids + [0] + dis

            t = dst_size // 2.0
            dx = np.arange(-t, t + 0.1, 1.0)
            dy = np.arange(-t, t + 0.1, 1.0)

            print("Original positions = %s" % str(x))
            print("Target positions = %s" % str(dx))

            all_rel_pos_bias = []

            for i in range(num_attn_heads):
                z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                f = interpolate.interp2d(x, y, z, kind='cubic')
                all_rel_pos_bias.append(
                    torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
            
            new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
            checkpoint_model[key] = new_rel_pos_bias