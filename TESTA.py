# anchor_sizes = [128, 256, 512]
# anchor_ratios = [[1, 1], [1, 2], [2, 1]]
anchor_sizes = [10]
anchor_ratios = [[1, 2.4], [2.4, 2.4]]
# for anchor_size_idx in range(len(anchor_sizes)):
#     for anchor_ratio_idx in range(n_anchratios):
#         # 长
#         anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
#         # 宽
#         anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

anchor_x = anchor_sizes[0] * anchor_ratios[0][0]
print(anchor_x)
# 宽
anchor_y = anchor_sizes[0] * anchor_ratios[0][1]
print(anchor_y)
# 10*24.0
# 24.0*24.0

