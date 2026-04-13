using LinearAlgebra
using Images, FileIO

function nmf_optimize(V, r; iterations=100)
    # 获取原始数据的行数和列数
    n, m = size(V)
    
    # 随机初始化 W (特征) 和 H (权重)，注意必须是非负数
    W = rand(n, r)
    H = rand(r, m)
    
    # 开始优化
    for i in 1:iterations
        # 更新 H 矩阵
        H = H .* (W' * V) ./ (W' * W * H .+ 1e-9)
        
        # 更新 W 矩阵
        W = W .* (V * H') ./ (W * H * H' .+ 1e-9)
    end
    
    return W, H
end