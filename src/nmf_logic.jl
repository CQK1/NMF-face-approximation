using LinearAlgebra

# 这是一个经典的 NMF 优化算法
# V 是你的原始图像矩阵，r 是你希望提取的特征数量（比如我们要提取20个局部特征）
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

# === 娱乐测试环节 ===

# 假设这是一张 100x100 像素的灰度人脸图像，我们用随机数模拟
face_image = rand(100, 100)

# 我们试着用 10 个特征来近似这张脸
for r in [10, 20, 50, 100, 200]
    println("正在使用 $r 个特征进行近似...")
    W, H = nmf_optimize(face_image, r, iterations=50)
    # 重建图像：把特征和权重乘起来
    reconstructed_face = W * H
    error = norm(face_image - reconstructed_face)
    println("当前近似的误差是: ", error)
end